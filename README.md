# Prost_Graph

## Overview

**Prost_Graph** is a deep learning framework designed for the prediction of **Gleason Grade** from prostate biopsy Whole Slide Images (WSIs).

The pipeline consists of **WSI patch extraction** followed by training deep learning models on the extracted patches for histopathological classification. It supports **training, inference, evaluation (metrics), and heatmap visualization**, aiming to provide a scalable and accurate solution for automated Gleason grading within computational pathology workflows.

---

## Patch Extraction

Patch generation is performed using the **CPath (Computational Pathology) pipeline**. The CLAM framework [1] is used for tissue segmentation and tile extraction.

All extracted patches from a WSI are stored as a tensor of shape **(N, 3, S, S)**, where:

* **N**: number of patches
* **S**: patch size

Additionally, segmentation masks and stitched visualizations are generated for quality control.

### Setup

Clone the patch extraction repository:

```bash
git clone https://github.com/PabloMeseguerEsbri/CPathPipeline.git
cd /workspace/CPathPipeline
```

### Patch Extraction Example

Assuming an initial magnification of **20x**, the following command extracts **512×512 patches at 10x magnification without overlap**:

```bash
python patching/extract_patches.py \
  --source <path_tif> \
  --save_dir <path_save> \
  --seg \
  --patch \
  --stitch \
  --patch_size 512 \
  --step_size 512 \
  --custom_downsample 2
```

After execution, verify that patches, masks, and stitched outputs are correctly generated.

---

## Project Structure

The `models/` directory includes:

### Feature Extraction Models

* Convolutional Neural Networks (CNN) [2]
* Vision Transformers (ViT) [3]
* Foundation Models (KEEP) [4]
* Foundation Models with Parameter-Efficient Fine-Tuning (PEFT) [5]

### Aggregation Architectures

* Multiple Instance Learning (MIL) [6]
* Graph Attention Networks (GAT) [7]

Each model type is organized into dedicated subdirectories to ensure modularity and clarity.

---

## Training Strategy

The dataset is split as follows:

* **90% Training set**

  * From this, **20% is used for validation**
* **10% Test set**

This strategy ensures robust evaluation while minimizing overfitting.

---

## Installation

To install dependencies:

```bash
pip install -r requirements.txt
python -m pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu11
```

---

## Hyperparameter Optimization

The training pipeline supports an optional **hyperparameter tuning mode**.

Enable it using:

```bash
--hiperparam_tuning
```

When activated, the system performs automated hyperparameter exploration to identify optimal configurations and improve model performance.

---

## References

[1] Lu, M. Y., et al. (2021). *Data-efficient and weakly supervised computational pathology on whole-slide images*. Nature Biomedical Engineering, 5(6), 555–570.

[2] Lu, M. Y., et al. (2024). *A visual-language foundation model for computational pathology*. Nature Medicine, 30, 863–874.

[3] Darcet, T., et al. (2023). *Vision Transformers Need Registers*. ICLR 2024.

[4] Zhou, X., et al. (2024). *A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis*. arXiv:2412.13126.

[5] Liu, H., et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*. NeurIPS.

[6] Amores, J. (2013). *Multiple instance classification: Review, taxonomy and comparative study*. Artificial Intelligence, 201, 81–105.

[7] Veličković, P., et al. (2018). *Graph Attention Networks*. ICLR 2018.
