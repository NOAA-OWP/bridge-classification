"""
Bridge Classification Model Training - Data Loader with Voxelization

This module provides a data loader for bridge point cloud classification with:
- Proper voxelization with feature aggregation and majority-vote labeling
- Support for HUC-organized directory structure
- Visualization tools to verify voxelization correctness
- PyTorch Lightning training integration

Data: use --train-dir for training data and optionally --val-dir for validation.
If --val-dir is not provided, the script uses --val-split to randomly split
the training directory into train/validation (val-split=0 means no validation).
Testing (gold/human-labeled) data is not used during training; use it later
for final evaluation after model selection.

Usage:
    # Test data loader (default: --train-dir ./data/ml-data/training)
    python src/train.py --train-dir ./data/ml-data/training

    # Visualize voxelization
    python src/train.py --visualize --sample-idx 0

    # Train with explicit validation directory
    python src/train.py --train --train-dir ./data/ml-data/training --val-dir ./data/ml-data/validation --epochs 50

    # Train with val-split when val-dir not provided
    python src/train.py --train --train-dir ./data/ml-data/training --val-split 0.2 --epochs 50
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

try:
    from torchview import draw_graph
    import spconv.pytorch as spconv
    HAS_TORCHVIEW = True
except ImportError:
    HAS_TORCHVIEW = False
    print("Warning: torchview not available. Network graph disabled.")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    print("Warning: pytorch_lightning not available. Training disabled.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError as e:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled. \nError: ", e)

# Import model
from model import SparseUNet


# Class label mapping for visualization
CLASS_COLORS = {
    0: 'black',    # Background/Unclassified
    1: 'orange',    # Ground/Water
    2: 'blue',      # Bridge Deck
    3: 'yellow'    # Obstacles/High Noise
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Ground/Water',
    2: 'Bridge Deck',
    3: 'Obstacles',
}


def aggregate_voxel_points(xyz: np.ndarray, features: np.ndarray, labels: np.ndarray,
                          voxel_coords: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate points within each voxel.

    For each voxel:
    - Features: Average intensity across all points
    - Coordinates: Use voxel center (discrete_coords * voxel_size + voxel_size/2)
    - Labels: Majority vote (most common label in voxel)

    Args:
        xyz: Original point coordinates (N, 3)
        features: Point features (N, 1) - intensity
        labels: Point labels (N,)
        voxel_coords: Discrete voxel coordinates (N, 3)
        voxel_size: Size of each voxel in meters

    Returns:
        Tuple of (aggregated_xyz, aggregated_features, aggregated_labels)
    """
    # Convert voxel_coords to tuple for grouping
    voxel_keys = [tuple(vc) for vc in voxel_coords]

    # Group points by voxel
    voxel_groups = {}
    for i, key in enumerate(voxel_keys):
        if key not in voxel_groups:
            voxel_groups[key] = []
        voxel_groups[key].append(i)

    # Aggregate each voxel
    aggregated_xyz = []
    aggregated_features = []
    aggregated_labels = []

    for voxel_key, point_indices in voxel_groups.items():
        # Get points in this voxel
        voxel_points = xyz[point_indices]
        voxel_features = features[point_indices]
        voxel_labels = labels[point_indices]

        # Average features (intensity)
        avg_feature = np.mean(voxel_features, axis=0)

        # Majority vote for label
        label_counts = Counter(voxel_labels)
        majority_label = label_counts.most_common(1)[0][0]

        # Use voxel center as coordinate
        # Convert discrete voxel coords back to continuous space
        voxel_center = np.array(voxel_key) * voxel_size + voxel_size / 2.0

        aggregated_xyz.append(voxel_center)
        aggregated_features.append(avg_feature)
        aggregated_labels.append(majority_label)

    return (
        np.array(aggregated_xyz, dtype=np.float32),
        np.array(aggregated_features, dtype=np.float32),
        np.array(aggregated_labels, dtype=np.int64)
    )


class BridgeDataset(Dataset):
    """
    Dataset for bridge point cloud classification with voxelization.

    Handles HUC-organized directory structure and properly aggregates
    points within voxels using majority vote for labels and averaging for features.
    """

    def __init__(self, data_dir: str, voxel_size: float = 0.05, augment: bool = False):
        """
        Args:
            data_dir: Path to directory containing .npy files (can be HUC-organized)
            voxel_size: Voxel size in meters (e.g., 0.05 for 5cm)
            augment: Whether to apply random rotations/scaling
        """
        self.data_dir = Path(data_dir)
        self.voxel_size = voxel_size
        self.augment = augment

        # Recursively find all .npy files (handles HUC folder structure)
        self.files = sorted(list(self.data_dir.rglob("*.npy")))

        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        # Define the ignore label (Background/Unclassified)
        self.ignore_label = 0

        print(f"Found {len(self.files)} bridge files in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and voxelize a single bridge sample.

        Returns:
            Tuple of (discrete_coords, features, labels)
            - discrete_coords: (N_voxels, 3) integer voxel coordinates
            - features: (N_voxels, 1) averaged intensity per voxel
            - labels: (N_voxels,) majority-vote labels per voxel
        """
        file_path = self.files[idx]

        # 1. Load Data
        data = np.load(file_path)  # Shape: (N, 5) -> [x, y, z, intensity, label]

        # Split into components
        xyz = data[:, 0:3]
        feat = data[:, 3:4]  # Intensity is the only feature for now
        labels = data[:, 4].astype(np.int64)

        # Preprocessing centers data at the mean (e.g., -50 to +50).
        # SpConv indices MUST be positive (0 to 100)?
        # We shift the min value to 0.0 for every sample.
        xyz -= xyz.min(axis=0)


        # 2. Data Augmentation (Optional)
        if self.augment:
            # Random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            xyz = xyz @ rotation_matrix

            # Random jitter
            jitter = np.random.normal(0, 0.01, size=xyz.shape)
            xyz += jitter

            # Re-shift to positive after rotation (rotation can make things negative again)
            xyz -= xyz.min(axis=0)

        # 3. Quantization (Voxelization)
        # Divide by voxel size and floor to get integer grid coordinates
        discrete_coords = np.floor(xyz / self.voxel_size).astype(np.int32)

        # 4. Aggregate points within voxels
        aggregated_xyz, aggregated_features, aggregated_labels = aggregate_voxel_points(
            xyz, feat, labels, discrete_coords, self.voxel_size
        )

        # 5. Re-quantize aggregated coordinates to get final discrete coords
        final_discrete_coords = np.floor(aggregated_xyz / self.voxel_size).astype(np.int32)

        return final_discrete_coords, aggregated_features, aggregated_labels


def sparse_collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to create a batch for sparse tensor format.

    Sparse tensors require coordinate format: [Batch_ID, X, Y, Z]

    Args:
        batch: List of (coords, features, labels) tuples

    Returns:
        Dictionary with keys:
        - coordinates: (N_total, 4) tensor [batch_id, x, y, z]
        - features: (N_total, 1) tensor
        - labels: (N_total,) tensor
    """
    batch_coords = []
    batch_feats = []
    batch_labels = []

    for batch_id, (coords, feats, labels) in enumerate(batch):
        # Append the Batch ID as the first column of the coordinates
        # Shape becomes (N, 4): [Batch_ID, X, Y, Z]
        b_idx = np.full((coords.shape[0], 1), batch_id, dtype=np.int32)
        batched_c = np.hstack([b_idx, coords])

        batch_coords.append(batched_c)
        batch_feats.append(feats)
        batch_labels.append(labels)

    # Concatenate all lists into single big tensors
    coords_tensor = torch.from_numpy(np.vstack(batch_coords)).int()
    feats_tensor = torch.from_numpy(np.vstack(batch_feats)).float()
    labels_tensor = torch.from_numpy(np.hstack(batch_labels)).long()

    return {
        "coordinates": coords_tensor,
        "features": feats_tensor,
        "labels": labels_tensor
    }


# PyTorch Lightning Components
if HAS_LIGHTNING:
    import spconv.pytorch as spconv

    class BridgeLightningModule(LightningModule):
        """
        PyTorch Lightning module for bridge classification training.
        """

        def __init__(
            self,
            input_channels=1,
            num_classes=4,
            base_channels=16,
            learning_rate=0.001,
            weight_decay=0.01,
            class_weights=None,
            monitor='val_loss',
            monitor_mode='min',
        ):
            """
            Args:
                input_channels: Number of input features (default: 1)
                num_classes: Number of output classes (default: 5)
                base_channels: Base number of channels (default: 16)
                learning_rate: Learning rate for optimizer (default: 0.001)
                weight_decay: Weight decay for optimizer (default: 0.01)
                class_weights: Class weights for loss function (default: None)
                monitor: Metric name for ReduceLROnPlateau (default: val_loss)
                monitor_mode: 'min' or 'max' for ReduceLROnPlateau (default: min)
            """
            super().__init__()
            self.save_hyperparameters()
            self.monitor = monitor
            self.monitor_mode = monitor_mode

            self.model = SparseUNet(
                input_channels=input_channels,
                num_classes=num_classes,
                base_channels=base_channels
            )

            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

            # Default class weights: [Background, Ground/ Water, Bridge Deck, Obstacle]
            # calculated weights from utils/calculate_weights.py
            if class_weights is None:
                # default training data weights
                class_weights = [6.216962881360028, 1.4907158415241706, 0.36471562073348884, 2.3448372700679068]

            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        def forward(self, x):
            """Forward pass."""
            return self.model(x)

        def _common_step(self, batch, batch_idx, prefix):
            """
            Shared logic for train and validation.

            Args:
                batch: Batch dictionary containing coordinates, features, and labels
                batch_idx: Index of batch
                prefix: Prefix for logging (train or val)
            """
            coords = batch['coordinates']  # (N, 4) -> [BatchID, X, Y, Z]
            feats = batch['features']      # (N, 1) -> Intensity
            labels = batch['labels']       # (N,)

            # Dynamic Shape Calculation
            # coords[:, 1:] gets X, Y, Z columns
            max_coords = coords[:, 1:].max(dim=0)[0]
            # limit = max_coords + 5
            # # Align to 32 if needed
            # spatial_shape = ((limit + 31) // 32 * 32).int().tolist()

            # Add small padding to be safe
            spatial_shape = (max_coords + 10).int().tolist()

            # Create SpConv Tensor
            input_sp_tensor = spconv.SparseConvTensor(
                features=feats,
                indices=coords,
                spatial_shape=spatial_shape,
                batch_size=coords[:, 0].max().item() + 1
            )

            # Forward pass
            output = self.model(input_sp_tensor)  # (N, num_classes)
            # Loss calculation
            loss = self.criterion(output, labels)

            # Calculate Metrics
            preds = torch.argmax(output, dim=1)

            # --- METRICS FOR BRIDGE DECK (CLASS 2) ---
            deck_target = (labels == 2)
            deck_pred = (preds == 2)

            # 1. Deck Recall (Accuracy on deck points)
            # "Of the real deck points, how many did we find?"
            deck_recall = 0.0
            if deck_target.sum() > 0:
                correct_deck = (preds[deck_target] == labels[deck_target]).sum().float()
                deck_recall = (correct_deck / deck_target.sum().float()) * 100.0

            # 2. Deck Precision
            # "Of the points we called 'deck', how many were actually deck?"
            deck_precision = 0.0
            if deck_pred.sum() > 0:
                true_positives = (deck_pred & deck_target).sum().float()
                deck_precision = (true_positives / deck_pred.sum().float()) * 100.0

            # 3. Deck IoU (Intersection over Union)
            # The gold standard for segmentation. Penalizes both false positives and false negatives.
            deck_iou = 0.0
            intersection = (deck_pred & deck_target).sum().float()
            union = (deck_pred | deck_target).sum().float()

            if union > 0:
                deck_iou = (intersection / union) * 100.0

            # 4. Overall Accuracy
            overall_acc = (preds == labels).float().mean() * 100.0

            # Logging
            # Loss (progress bar)
            self.log(f'{prefix}_loss', loss, on_step=(prefix=='train'), on_epoch=True, prog_bar=True)

            # Deck IoU (progress bar - this is most important metric)
            self.log(f'{prefix}_deck_iou', deck_iou, on_step=(prefix=='train'), on_epoch=True, prog_bar=True)

            # Detailed Metrics (logged but hidden from progress bar to keep it clean)
            self.log(f'{prefix}_deck_recall', deck_recall, on_step=(prefix=='train'), on_epoch=True, prog_bar=False)
            self.log(f'{prefix}_deck_precision', deck_precision, on_step=(prefix=='train'), on_epoch=True, prog_bar=False)
            self.log(f'{prefix}_overall_acc', overall_acc, on_step=(prefix=='train'), on_epoch=True, prog_bar=False)

            return loss

        def training_step(self, batch, batch_idx):
            """Training step."""
            return self._common_step(batch, batch_idx, "train")

        def validation_step(self, batch, batch_idx):
            """Validation step."""
            return self._common_step(batch, batch_idx, "val")

        def configure_optimizers(self):
            """Configure optimizer."""
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            # Add ReduceLROnPlateau
            scheduler = {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=self.monitor_mode, factor=0.5, patience=5, min_lr=1e-6
                ),
                'monitor': self.monitor,
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]


    class BridgeDataModule(LightningDataModule):
        """
        PyTorch Lightning data module for bridge dataset.
        Uses train_dir and optionally val_dir, or val_split on train_dir when val_dir not set.
        """

        def __init__(
            self,
            train_dir: str,
            val_dir: Optional[str] = None,
            voxel_size: float = 0.05,
            batch_size: int = 4,
            num_workers: int = 4,
            augment: bool = True,
            val_split: float = 0.0,
        ):
            """
            Args:
                train_dir: Path to directory containing training .npy files
                val_dir: Path to directory containing validation .npy files; if None, use val_split on train_dir
                voxel_size: Voxel size in meters (default: 0.05)
                batch_size: Batch size (default: 4)
                num_workers: Number of data loader workers (default: 4)
                augment: Whether to apply augmentation (default: True)
                val_split: Validation split ratio when val_dir is not set (default: 0.0, no validation)
            """
            super().__init__()
            self.train_dir = train_dir
            self.val_dir = val_dir
            self.voxel_size = voxel_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.augment = augment
            self.val_split = val_split

        def setup(self, stage=None):
            """Setup datasets from train_dir and val_dir, or val_split on train_dir."""
            if stage == 'fit' or stage is None:
                try:
                    full_dataset = BridgeDataset(
                        self.train_dir,
                        voxel_size=self.voxel_size,
                        augment=False
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Training directory {self.train_dir!r} has no .npy files or does not exist. {e}"
                    ) from e

                # Explicit val_dir: use it if directory exists and has .npy files
                if self.val_dir and os.path.isdir(self.val_dir):
                    npy_files = list(Path(self.val_dir).rglob("*.npy"))
                    if npy_files:
                        self.train_dataset = BridgeDataset(
                            self.train_dir,
                            voxel_size=self.voxel_size,
                            augment=self.augment
                        )
                        self.val_dataset = BridgeDataset(
                            self.val_dir,
                            voxel_size=self.voxel_size,
                            augment=False
                        )
                    else:
                        self.train_dataset = BridgeDataset(
                            self.train_dir,
                            voxel_size=self.voxel_size,
                            augment=self.augment
                        )
                        self.val_dataset = None
                elif self.val_split > 0:
                    # Split training directory into train/val by index
                    dataset_size = len(full_dataset)
                    val_size = int(dataset_size * self.val_split)
                    train_size = dataset_size - val_size
                    indices = torch.randperm(dataset_size).tolist()
                    train_indices = indices[:train_size]
                    val_indices = indices[train_size:]
                    self.train_dataset = BridgeDataset(
                        self.train_dir,
                        voxel_size=self.voxel_size,
                        augment=self.augment
                    )
                    self.train_dataset.files = [full_dataset.files[i] for i in train_indices]
                    self.val_dataset = BridgeDataset(
                        self.train_dir,
                        voxel_size=self.voxel_size,
                        augment=False
                    )
                    self.val_dataset.files = [full_dataset.files[i] for i in val_indices]
                else:
                    self.train_dataset = BridgeDataset(
                        self.train_dir,
                        voxel_size=self.voxel_size,
                        augment=self.augment
                    )
                    self.val_dataset = None

        def train_dataloader(self):
            """Create training dataloader."""
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=sparse_collate_fn,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )

        def val_dataloader(self):
            """Create validation dataloader."""
            if self.val_dataset is None:
                return None
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=sparse_collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )


def save_network_graph(model, save_dir, filename="network_architecture"):
    """
    Traces the model on GPU (required for spconv) and saves the graph as PNG.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping graph (spconv requires GPU for tracing).")
        return

    print("Generating network graph...")

    # --- 1. Define Wrapper ---
    class GraphWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features, indices):
            x = spconv.SparseConvTensor(
                features=features,
                indices=indices,
                spatial_shape=[100, 100, 100],
                batch_size=1
            )
            return self.model(x)

    # --- 2. Move Model to GPU ---
    # We must move the model to CUDA because spconv CPU implementation is incomplete
    model = model.cuda()

    # --- 3. Create Dummy Inputs on GPU ---
    # Coords: [BatchID, X, Y, Z] -> [0, 50, 50, 50]
    dummy_coords = torch.tensor([[0, 50, 50, 50]], dtype=torch.int32, device='cuda')
    # Features: [1 point, 1 channel]
    dummy_features = torch.randn(1, 1, dtype=torch.float32, device='cuda')

    try:
        wrapper = GraphWrapper(model)

        # --- 4. Draw Graph (Force device='cuda') ---
        model_graph = draw_graph(
            wrapper,
            input_data=(dummy_features, dummy_coords),
            depth=3, # Increased depth slightly to show layer details
            expand_nested=True,
            device='cuda'
        )

        # --- 5. Save ---
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        # model_graph.visual_graph.render(save_path, format='png', cleanup=True)
        model_graph.visual_graph.render(save_path, format='svg', cleanup=True)
        print(f"Network graph saved to {save_path}.png")

    except Exception as e:
        print(f"Failed to generate network graph: {e}")
    finally:
        # --- 6. CLEANUP: Move model back to CPU ---
        # Important: Move back so PyTorch Lightning can handle device placement normally
        model = model.cpu()
        torch.cuda.empty_cache()


def visualize_voxelization(data_dir: str, sample_idx: int = 0, voxel_size: float = 0.05):
    """
    Visualize original vs voxelized point cloud for a sample bridge.

    Args:
        data_dir: Path to data directory
        sample_idx: Index of sample to visualize
        voxel_size: Voxel size in meters
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        return

    dataset = BridgeDataset(data_dir, voxel_size=voxel_size, augment=False)

    if sample_idx >= len(dataset):
        print(f"Error: sample_idx {sample_idx} is out of range (dataset has {len(dataset)} samples)")
        return

    # Load original data (this logic loads NPY directly which is already pre-normalized)
    file_path = dataset.files[sample_idx]
    original_data = np.load(file_path)
    original_xyz = original_data[:, 0:3]
    # # Re-apply the shift to zero so it matches the dataset logic
    original_xyz -= original_xyz.min(axis=0)
    original_labels = original_data[:, 4].astype(int)

    # Get voxelized data from dataset
    discrete_coords, features, labels = dataset[sample_idx]

    # Convert discrete coords back to continuous space for visualization
    voxelized_xyz = discrete_coords.astype(float) * voxel_size + voxel_size / 2.0

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))

    # Add filename as figure title
    fig.suptitle(file_path.name, fontsize=14, y=0.98)

    # Original point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    for label in np.unique(original_labels):
        mask = original_labels == label
        color = CLASS_COLORS.get(label, 'gray')
        ax1.scatter(original_xyz[mask, 0], original_xyz[mask, 1], original_xyz[mask, 2],
                   c=color, label=CLASS_NAMES.get(label, f'Class {label}'), s=1, alpha=0.6)
    ax1.set_title(f'Original Point Cloud\n({len(original_xyz)} points)', fontsize=12)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # Voxelized point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    for label in np.unique(labels):
        mask = labels == label
        color = CLASS_COLORS.get(label, 'gray')
        ax2.scatter(voxelized_xyz[mask, 0], voxelized_xyz[mask, 1], voxelized_xyz[mask, 2],
                   c=color, label=CLASS_NAMES.get(label, f'Class {label}'), s=10, alpha=0.8)
    ax2.set_title(f'Voxelized Point Cloud\n({len(voxelized_xyz)} voxels, {voxel_size*100:.1f}cm voxel size)', fontsize=12)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()

    # Set equal aspect ratio for both plots
    for ax in [ax1, ax2]:
        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Calculate ranges
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        zrange = zlim[1] - zlim[0]

        # Find max range
        max_range = max(xrange, yrange, zrange)

        # Center the plot
        xcenter = (xlim[0] + xlim[1]) / 2
        ycenter = (ylim[0] + ylim[1]) / 2
        zcenter = (zlim[0] + zlim[1]) / 2

        # Set new limits
        ax.set_xlim([xcenter - max_range/2, xcenter + max_range/2])
        ax.set_ylim([ycenter - max_range/2, ycenter + max_range/2])
        ax.set_zlim([zcenter - max_range/2, zcenter + max_range/2])

    plt.tight_layout()

    # Print statistics
    compression_ratio = len(original_xyz) / len(voxelized_xyz) if len(voxelized_xyz) > 0 else 0
    print(f"\n{'='*60}")
    print(f"Visualization Statistics for Sample {sample_idx}")
    print(f"{'='*60}")
    print(f"File: {file_path.name}")
    print(f"Original points: {len(original_xyz):,}")
    print(f"Voxelized points: {len(voxelized_xyz):,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Voxel size: {voxel_size*100:.1f} cm")
    print(f"\nClass distribution (original):")
    for label in sorted(np.unique(original_labels)):
        count = np.sum(original_labels == label)
        pct = 100 * count / len(original_labels)
        print(f"  {CLASS_NAMES.get(label, f'Class {label}')}: {count:,} ({pct:.1f}%)")
    print(f"\nClass distribution (voxelized):")
    for label in sorted(np.unique(labels)):
        count = np.sum(labels == label)
        pct = 100 * count / len(labels) if len(labels) > 0 else 0
        print(f"  {CLASS_NAMES.get(label, f'Class {label}')}: {count:,} ({pct:.1f}%)")
    print(f"{'='*60}\n")

    plt.show()


def main():
    """Main entry point with command-line argument parsing."""
    if HAS_LIGHTNING:
        pl.seed_everything(27, workers=True)

    parser = argparse.ArgumentParser(
        description='Bridge point cloud data loader with voxelization and visualization'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/ml-data',
        help='Data directory for test loader and visualize when not using --train-dir (default: ./data/ml-data)'
    )

    parser.add_argument(
        '--train-dir',
        type=str,
        default='./data/ml-data/training',
        help='Path to directory containing training .npy files (default: ./data/ml-data/training)'
    )

    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help='Path to directory containing validation .npy files; if not set, validation uses --val-split on training data'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.0,
        help='Validation split ratio when --val-dir is not provided (0 = no validation, e.g. 0.2 for 20%% val)'
    )

    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.05,
        help='Voxel size in meters (default: 0.05 = 5cm)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Run visualization on a sample file'
    )

    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Index of sample to visualize (default: 0)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for testing loader (default: 4)'
    )

    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation (random rotation and jitter)'
    )

    # Training arguments
    parser.add_argument(
        '--train',
        action='store_true',
        help='Enable training mode'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )

    parser.add_argument(
        '--base-channels',
        type=int,
        default=16,
        help='Base number of channels in model (default: 16)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers (default: 4)'
    )

    parser.add_argument(
        '--exp-name',
        type=str,
        default='bridge_classify_base',
        help='Name of experiment for logging'
    )

    parser.add_argument(
        '--class-weights',
        type=str,
        default=None,
        help='Path to JSON file with "weights" list from calculate_weights.py --output. If not set, uses built-in default weights.',
    )

    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (0 for CPU, >0 for GPU, None for auto-detect).'
    )

    parser.add_argument(
        '--early-stopping',
        action='store_true',
        default=False,
        help='Stop training when the monitored metric (see --monitor) does not improve for --early-stopping-patience epochs (requires validation).',
    )

    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Number of epochs with no improvement after which to stop (used only if --early-stopping).',
    )

    parser.add_argument(
        '--monitor',
        type=str,
        default='val_deck_iou',
        help='Metric to monitor for checkpointing and early stopping (default: val_deck_iou). Use val_deck_iou for best deck IoU, val_loss for validation loss. Ignored when no validation data (train_loss used).',
    )

    args = parser.parse_args()
    print(f"Using args: {args}")

    if args.train:
        if not HAS_LIGHTNING:
            print("Pytorch Lightning not installed")
            return

        train_dir = args.train_dir
        val_dir = args.val_dir
        class_weights_list: Optional[List[float]] = None

        if args.class_weights is not None:
            cw_path = Path(args.class_weights).expanduser().resolve()
            if not cw_path.exists():
                raise SystemExit(f"Error: --class-weights file not found: {cw_path}")
            try:
                with open(cw_path, "r") as f:
                    cw_obj = json.load(f)
                weights = cw_obj.get("weights")
                if not isinstance(weights, list):
                    raise SystemExit(
                        f'Error: --class-weights JSON must contain a "weights" list. Got: {type(weights)}'
                    )
                if len(weights) != 4:
                    raise SystemExit(
                        f"Error: --class-weights must have 4 values for classes 0-3; got {len(weights)}"
                    )
                class_weights_list = [float(x) for x in weights]
            except json.JSONDecodeError as e:
                raise SystemExit(f"Error: invalid JSON in --class-weights file: {cw_path}\n{e}") from e

        has_validation = (
            (val_dir and os.path.isdir(val_dir) and len(list(Path(val_dir).rglob("*.npy"))) > 0)
            or args.val_split > 0
        )
        effective_monitor = args.monitor if has_validation else 'train_loss'
        monitor_mode = 'max' if 'iou' in effective_monitor.lower() else 'min'

        if val_dir is None or (not os.path.isdir(val_dir)):
            if args.val_split > 0:
                print("Note: --val-dir not provided; validation will use --val-split on training data.")
            else:
                print("Note: No validation (--val-dir not provided and --val-split is 0).")
        else:
            npy_in_val = list(Path(val_dir).rglob("*.npy"))
            if not npy_in_val:
                if args.val_split > 0:
                    print("Note: --val-dir has no .npy files; validation will use --val-split on training data.")
                else:
                    print("Note: No validation (--val-dir has no .npy files and --val-split is 0).")

        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Train dir: {train_dir}")
        print(f"Val dir: {val_dir}")
        print(f"Val split (when val-dir not used): {args.val_split}")
        print(f"Voxel size: {args.voxel_size*100:.1f} cm")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Augmentation: {args.augment}")
        print(
            "Class weights: "
            + (str(Path(args.class_weights).expanduser().resolve()) if args.class_weights else "default (built-in)")
        )
        print(f"Experiment name: {args.exp_name}")
        if args.early_stopping:
            print(f"Early stopping: patience={args.early_stopping_patience} ({effective_monitor}).")
        print("=" * 60)

        # Create data module
        data_module = BridgeDataModule(
            train_dir=train_dir,
            val_dir=val_dir,
            voxel_size=args.voxel_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=args.augment,
            val_split=args.val_split,
        )

        # Create model
        model = BridgeLightningModule(
            input_channels=1,
            num_classes=4,
            base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            class_weights=class_weights_list,
            monitor=effective_monitor,
            monitor_mode=monitor_mode,
        )

        # Logger: TensorBoard
        tensorboard_logger = TensorBoardLogger(
            save_dir="./experiments",
            name=args.exp_name,
            default_hp_metric=False,
        )

        csv_logger = CSVLogger(
            save_dir="./experiments",
            name=args.exp_name,
            version=tensorboard_logger.version,
        )

        # Archive class weights used for this run (self-contained)
        log_dir = Path(tensorboard_logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        class_weights_dest = log_dir / "class_weights.json"
        if args.class_weights is not None:
            shutil.copy2(Path(args.class_weights).expanduser().resolve(), class_weights_dest)
        else:
            weights_used = model.class_weights.cpu().tolist()
            with open(class_weights_dest, "w") as f:
                json.dump({"weights": weights_used, "source": "built-in default"}, f, indent=2)

        # Setup checkpoint callback (use train_loss when no validation data)
        if has_validation:
            checkpoint_callback = ModelCheckpoint(
                filename=f'bridge-unet-{{epoch:02d}}-{{{effective_monitor}:.4f}}',
                monitor=effective_monitor,
                save_top_k=5,
                mode=monitor_mode,
                save_last=True
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                filename='bridge-unet-{epoch:02d}-{train_loss:.4f}',
                monitor='train_loss',
                save_top_k=5,
                mode='min',
                save_last=True
            )

        callbacks = [checkpoint_callback]
        if args.early_stopping and has_validation:
            callbacks.append(
                EarlyStopping(
                    monitor=effective_monitor,
                    mode=monitor_mode,
                    patience=args.early_stopping_patience,
                    verbose=True,
                )
            )
        elif args.early_stopping and not has_validation:
            print("Note: --early-stopping requires validation; early stopping not enabled.")

        # Create trainer
        # Determine accelerator and devices based on gpus argument
        if args.gpus is None:
            # Auto-detect: will use GPU if available, otherwise CPU
            accelerator = "auto"
            devices = "auto"
        elif args.gpus == 0:
            # Explicitly use CPU
            accelerator = "cpu"
            devices = 1
        else:
            # Use GPU with specified number of devices
            accelerator = "gpu"
            devices = args.gpus

        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator=accelerator,
            devices=devices,
            logger=[tensorboard_logger, csv_logger],
            callbacks=callbacks,
            log_every_n_steps=10,
        )

        if HAS_TORCHVIEW:
            save_network_graph(model, tensorboard_logger.log_dir)

        # Train model
        trainer.fit(model, data_module)

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Checkpoints saved to: ./experiments/{args.exp_name}")
        return

    # Directory for test loader and visualize: use --train-dir (default training data)
    effective_dir = args.train_dir

    # Visualization mode
    if args.visualize:
        visualize_voxelization(effective_dir, args.sample_idx, args.voxel_size)
        return

    # Test data loader
    print("=" * 60)
    print("Testing Bridge Data Loader")
    print("=" * 60)
    print(f"Data directory: {effective_dir}")
    print(f"Voxel size: {args.voxel_size*100:.1f} cm")
    print(f"Augmentation: {args.augment}")
    print("=" * 60)

    # Create dataset
    try:
        dataset = BridgeDataset(effective_dir, voxel_size=args.voxel_size, augment=args.augment)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=sparse_collate_fn,
        shuffle=True
    )

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print("\nTesting data loader...")
    print("-" * 60)

    # Test a few batches
    for batch_idx, batch in enumerate(loader):
        coords = batch['coordinates']
        feats = batch['features']
        labels = batch['labels']

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Coordinates shape: {coords.shape}  [N_total, 4] = [batch_id, x, y, z]")
        print(f"  Features shape: {feats.shape}     [N_total, 1] = [intensity]")
        print(f"  Labels shape: {labels.shape}       [N_total,]")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")
        print(f"  Label distribution:")
        for label in torch.unique(labels):
            count = torch.sum(labels == label).item()
            pct = 100 * count / len(labels)
            print(f"    {CLASS_NAMES.get(label.item(), f'Class {label.item()}')}: {count:,} ({pct:.1f}%)")

        if batch_idx >= 2:  # Show first 3 batches
            break

    print("\n" + "=" * 60)
    print("Data loader test complete!")
    print("=" * 60)
    print(f"\nTo visualize a sample, run:")
    print(f"  python src/train.py --visualize --sample-idx 0 --voxel-size {args.voxel_size}")


if __name__ == "__main__":
    main()
