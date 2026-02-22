# Bridge Classification

USGS LiDAR Bridge Point Cloud Classification Pipeline

---

## Overview

A comprehensive pipeline for classifying 3D LiDAR point clouds around bridges from
USGS 3DEP data into 4 semantic classes using a sparse 3D U-Net. The pipeline covers
data acquisition through inference, with automated weak supervision for scalable
training data generation (550K+ bridges).

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, classification schema, full algorithm details |
| [Data Pipeline](data-pipeline.md) | Step-by-step data flow with shapes at each stage |
| [AWS Batch Inference](aws-batch-inference.md) | Scaling inference with AWS Batch array jobs |
| [Module Reference](module-reference.md) | Every module's public API and CLI arguments |
| [Design Decisions](decisions.md) | Rationale for key architectural choices |

## Quick Reference

### Classification Schema

| Model Class | Name | ASPRS Input | ASPRS Output |
|-------------|------|-------------|--------------|
| 0 | Background/Unclassified | 1, 7, others | 1 |
| 1 | Ground/Water | 2 (Ground), 9 (Water) | 2 |
| 2 | Bridge Deck | 17 | 17 |
| 3 | Obstacles/High Noise | 18 | 18 |

### Pipeline Stages

```mermaid
flowchart LR
  A[OSM Bridges + USGS EPT] --> B[Download & Weak Supervise]
  B --> C[Preprocess & Normalize]
  C --> D[Split Train/Val/Test]
  D --> E[Train Sparse U-Net]
  D --> F[Compute Class Weights]
  F --> E
  E --> G[Inference]
```

### Getting Started

See the [README](https://github.com/erdc/bridge_classification#readme) for full installation and pipeline run commands.
