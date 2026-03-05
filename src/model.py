"""
Sparse ResNet U-Net Model for Bridge Point Cloud Classification

This module implements a sparse 3D U-Net architecture using spconv for efficient
processing of sparse point cloud voxel grids. The model uses ResNet-style residual
blocks in both encoder and decoder paths with skip connections.

Classes:
    ResidualBlock: ResNet-style block with sparse convolutions
    SparseUNet: U-Net architecture with encoder-decoder structure
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import functools


class ResidualBlock(nn.Module):
    """
    ResNet-style residual block for sparse convolutions.

    Uses SubMConv3d to maintain sparsity pattern while applying convolutions.
    """

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            norm_fn: Normalization function (e.g., BatchNorm1d)
            indice_key: Key for spconv indice management
        """
        super().__init__()

        # SubMConv3d keeps the sparsity pattern unchanged (active points stay active)
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(out_channels)

        # Shortcut connection for ResNet capability
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False, indice_key=indice_key
            )

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: SparseConvTensor input

        Returns:
            SparseConvTensor output
        """
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SparseUNet(nn.Module):
    """
    Sparse 3D U-Net with ResNet-style blocks for bridge point cloud classification.

    Architecture:
    - Encoder: 4 levels with downsampling (x1, x2, x4, x8)
    - Decoder: 3 levels with upsampling and skip connections
    - Output: Per-voxel classification into num_classes
    """

    def __init__(self, input_channels=1, num_classes=4, base_channels=16):
        """
        Args:
            input_channels: Number of input features (default: 1 for intensity)
            num_classes: Number of output classes (default: 4)
                - 0: Background/Unclassified
                - 1: Ground/Water
                - 2: Bridge Deck
                - 3: Obstacles/High Noise
            base_channels: Base number of channels (default: 16)
        """
        super().__init__()
        self.sparse_shape = [4096, 4096, 1024]  # Max grid size (adjust based on max bridge size / 0.1)

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # --- ENCODER ---
        # Initial Block
        self.conv_input = spconv.SubMConv3d(input_channels, base_channels, 3, padding=1, bias=False, indice_key='subm0')
        self.bn_input = norm_fn(base_channels)
        self.relu = nn.ReLU()

        # Encoder Block 1 (x1)
        self.enc1 = ResidualBlock(base_channels, base_channels, norm_fn, indice_key='subm0')

        # Downsample 1 (x2) -> stride 2
        self.down1 = spconv.SparseConv3d(base_channels, base_channels*2, 3, 2, padding=1, bias=False, indice_key='down1')
        self.bn_down1 = norm_fn(base_channels*2)
        self.enc2 = ResidualBlock(base_channels*2, base_channels*2, norm_fn, indice_key='subm1')

        # Downsample 2 (x4)
        self.down2 = spconv.SparseConv3d(base_channels*2, base_channels*4, 3, 2, padding=1, bias=False, indice_key='down2')
        self.bn_down2 = norm_fn(base_channels*4)
        self.enc3 = ResidualBlock(base_channels*4, base_channels*4, norm_fn, indice_key='subm2')

        # Downsample 3 (x8) - Bottleneck
        self.down3 = spconv.SparseConv3d(base_channels*4, base_channels*8, 3, 2, padding=1, bias=False, indice_key='down3')
        self.bn_down3 = norm_fn(base_channels*8)
        self.bottleneck = ResidualBlock(base_channels*8, base_channels*8, norm_fn, indice_key='subm3')

        # --- DECODER ---
        # Upsample 3 (x4) - Inverse of Downsample 3
        self.up3 = spconv.SparseInverseConv3d(base_channels*8, base_channels*4, 3, indice_key='down3', bias=False)
        self.bn_up3 = norm_fn(base_channels*4)
        self.dec3 = ResidualBlock(base_channels*8, base_channels*4, norm_fn, indice_key='subm2')  # Concatenation doubles channels

        # Upsample 2 (x2)
        self.up2 = spconv.SparseInverseConv3d(base_channels*4, base_channels*2, 3, indice_key='down2', bias=False)
        self.bn_up2 = norm_fn(base_channels*2)
        self.dec2 = ResidualBlock(base_channels*4, base_channels*2, norm_fn, indice_key='subm1')

        # Upsample 1 (x1)
        self.up1 = spconv.SparseInverseConv3d(base_channels*2, base_channels, 3, indice_key='down1', bias=False)
        self.bn_up1 = norm_fn(base_channels)
        self.dec1 = ResidualBlock(base_channels*2, base_channels, norm_fn, indice_key='subm0')

        # --- HEAD ---
        # Map back to num_classes
        self.classifier = spconv.SubMConv3d(base_channels, num_classes, 3, padding=1, bias=True, indice_key='subm0')

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x: spconv.SparseConvTensor with features and indices

        Returns:
            torch.Tensor: (N, num_classes) logits for each voxel
        """
        # Encoder
        x = self.conv_input(x)
        x = x.replace_feature(self.bn_input(x.features))
        x = x.replace_feature(self.relu(x.features))

        e1 = self.enc1(x)

        x = self.down1(e1)
        x = x.replace_feature(self.bn_down1(x.features))
        x = x.replace_feature(self.relu(x.features))
        e2 = self.enc2(x)

        x = self.down2(e2)
        x = x.replace_feature(self.bn_down2(x.features))
        x = x.replace_feature(self.relu(x.features))
        e3 = self.enc3(x)

        x = self.down3(e3)
        x = x.replace_feature(self.bn_down3(x.features))
        x = x.replace_feature(self.relu(x.features))
        b = self.bottleneck(x)

        # Decoder (with Skip Connections)
        # Note: In spconv, inverse convs recover the spatial shape of the corresponding downsample
        u3 = self.up3(b)
        u3 = u3.replace_feature(self.bn_up3(u3.features))
        u3 = u3.replace_feature(self.relu(u3.features))
        # Skip connection: Concatenate features from encoder
        u3 = u3.replace_feature(torch.cat([u3.features, e3.features], dim=1))
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = u2.replace_feature(self.bn_up2(u2.features))
        u2 = u2.replace_feature(self.relu(u2.features))
        u2 = u2.replace_feature(torch.cat([u2.features, e2.features], dim=1))
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = u1.replace_feature(self.bn_up1(u1.features))
        u1 = u1.replace_feature(self.relu(u1.features))
        u1 = u1.replace_feature(torch.cat([u1.features, e1.features], dim=1))
        d1 = self.dec1(u1)

        # Classification
        out = self.classifier(d1)

        return out.features  # Returns (N, num_classes)
