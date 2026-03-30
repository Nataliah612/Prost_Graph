# Prost_Graph

## Overview
**Prost_Graph** is a deep learning framework for predicting the **Gleason Grade** from prostate biopsy Whole Slide Images (WSI).

The pipeline is based on **WSI patch extraction (patching)** followed by training deep learning models on extracted image patches for histopathology classification, inference and evaluacion of the models in metrics and visualization of heatmaps. Developing a scalable and accurate system for automated Gleason Grade prediction, supporting computational pathology workflows through modern deep learning techniques.

## Patch Extraction
Patch generation is performed using the open-source pipeline from **CPath (Computational Pathology GitHub repository)**. CLAM software [1] is used to segment and extract the tiles. All patches within WSI are stacked into a matrix of shape (N,3,S,S), being N the number of patches and S their size. Masks and stitches are provided to visualize the segmentation.

Clone the CPath patch extraction repository:
<pre> git clone https://github.com/PabloMeseguerEsbri/CPathPipeline.git </pre>
<pre> cd /workspace/CPathPipeline </pre>
Considering initial 20x magnification, run the following line to extract 512x512 patches at 10x magnification without overlap.

<pre> python patching/extract_patches.py --source <path_tif> --save_dir <path_save> --seg --patch --stitch --patch_size 512 --step_size 512 --custom_downsample 2 </pre>

Verify that patches are correctly generated and organized by slide/class.

## Project Structure

The models/ directory contains:

Feature extraction models: 
- Convolutional Neural Networks (CNN) [2]
- Vision Transformers (ViT) [3]
- Foundation models (KEEP) [4]
- Foundation models (KEEP) with Parameter-Efficient Fine-Tuning (PEFT) methods [5]

Classification heads
Agregation architectures:
- MIL [6]
- GAT [7]

Each model type is organized in its corresponding subfolder for clarity and modularity.

## Training Strategy
The dataset is split as follows:
* 90% Training set, from this subset, 20% is used for validation
* 10% Test set

This split ensures robust model evaluation and prevents overfitting.

## Hyperparameter Research

The training pipeline supports an optional hyperparameter optimization mode.

To enable this research mode, use the following flag:

<pre> --hiperparam_tuning </pre>

When activated, the system performs automated exploration of hyperparameters to improve model performance and identify optimal configurations.

## Resources
[1] Lu, M. Y., Williamson, D. F., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6), 555-570.
[2] M. Y. Lu et al., “A visual-language foundation model for computational pathology,” Nature Me
dicine, vol. 30, págs. 863-874, 3 mar. de 2024, ISSN: 1546170X. DOI: 10.1038/S41591-024
02856-4;SUBJMETA.
[3] T. Darcet, M. Oquab, J. Mairal y P. Bojanowski, “Vision Transformers Need Registers,” 12th
International Conference on Learning Representations, ICLR 2024, sep. de 2023.
[4] Zhou, X., Sun, L., He, D., Guan, W., Wang, R., Wang, L., ... & Xie, W. (2024). A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis. arXiv preprint arXiv:2412.13126.
[5] H. Liu et al., “Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context
Learning,” Advances in Neural Information Processing Systems, vol. 35, págs. 1950-1965, dic. de
2022.
[6] J. Amores, “Multiple instance classification: Review, taxonomy and comparative study,” Artificial
Intelligence, vol. 201, págs. 81-105, ago. de 2013, ISSN: 0004-3702. DOI: 10.1016/J.ARTINT.
2013.06.003.
[7] P. Veliˇckovi´c, A. Casanova, P. Liò, G. Cucurull, A. Romero e Y. Bengio, “Graph Attention Net
works,” 6th International Conference on Learning Representations, ICLR 2018- Conference
Track Proceedings, oct. de 2017. DOI: 10.1007/978-3-031-01587-8_7.
