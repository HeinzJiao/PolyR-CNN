<div align="center">
  <h1>PolyR-CNN: R-CNN for End-to-End Polygonal Building Outline Extraction</h1>
  <p><strong>PolyR-CNN</strong> is an efficient end-to-end framework for directly predicting vectorized building polygons and bounding boxes from remote sensing imagery.</p>
  <p>
    <a href="https://doi.org/10.1016/j.isprsjprs.2024.10.006"><img src="https://img.shields.io/badge/ISPRS-2024-0A66C2" alt="ISPRS 2024"></a>
    <a href="https://github.com/HeinzJiao/PolyR-CNN"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub"></a>
  </p>
</div>

## Overview

PolyR-CNN predicts polygonal building outlines in a fully end-to-end manner, avoiding the inefficiencies of multi-stage pipelines. It leverages Region-of-Interest features for polygon prediction and introduces vertex proposal features to guide more regular and compact outlines.

## Data Preprocessing

To reduce training overhead, the CrowdAI annotations are preprocessed with localized polygon padding so that each polygon contains a fixed number of vertices. This design minimizes repeated on-the-fly computation during training.

Run preprocessing with:

```bash
python3 preprocess_annotation.py --json_path /path/to/annotations.json --save_path /path/to/save/annotation_preprocessed.json --is_training True --num_corners 96
```

## Environment Setup

### 1. Create a Conda Environment

```bash
conda create -n polyrcnn python=3.8
conda activate polyrcnn
```

### 2. Install Detectron2

Follow the official [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/tree/main).

### 3. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

## Training

Start training on the CrowdAI dataset with:

```bash
python3 train_net.py --num-gpus <number_of_gpus> --config-file configs/polyrcnn.res50.100pro.aicrowd.yaml
```

You can switch to alternative backbones or settings using the configuration files under [`configs/`](./configs/).

## Testing and Evaluation

For visualization on a single image or a folder of images, use `demo.py`. For evaluation on a complete COCO-format dataset, use `evaluate.py`, which saves predictions in COCO format for downstream analysis.

For detailed metric evaluation such as MS-COCO metrics and PoLiS, please refer to the evaluation tools released in [HiSup](https://github.com/SarahwXU/HiSup), especially `tools/evaluation.py`.

## Citation

If you find PolyR-CNN useful in your research, please consider citing:

```bibtex
@article{jiao2024polyrcnn,
  title={PolyR-CNN: R-CNN for end-to-end polygonal building outline extraction},
  author={Jiao, Weiqin and Persello, Claudio and Vosselman, George},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={218},
  pages={33--43},
  year={2024},
  publisher={Elsevier}
}
```

## Related Research

This repository belongs to a broader research line on polygonal vectorization from aerial imagery and large-scale topographic map generation. If PolyR-CNN is relevant to your work, you may also want to follow the companion papers below.

| Paper | Venue | Focus | Resources |
| --- | --- | --- | --- |
| **ACPV-Net** | CVPR 2026 | All-class polygonal vectorization and topology-aware vector basemap generation from a single aerial image. | [Paper](https://arxiv.org/abs/2603.16616) · [Code](https://github.com/HeinzJiao/ACPV-Net) · [Data](https://huggingface.co/datasets/HeinzJiao/Deventer-512) · [Weights](https://huggingface.co/HeinzJiao/deventer512_vmamba-s_m_vh-ldm_kl4_b8) |
| **LDPoly** | ISPRS 2025 | Latent diffusion for polygonal road outline extraction in topographic mapping. | [Paper](https://doi.org/10.1016/j.isprsjprs.2025.10.005) · [Code](https://github.com/HeinzJiao/LDPoly) · [Data and Weights](https://drive.google.com/drive/folders/1jsjuZxFdU9a8q-m0TNCj1MfX9rixTYJl?usp=sharing) · [Demo](https://colab.research.google.com/drive/1IW5AGfn3w3y9wSquYgXolGhcVwIWkoNd#scrollTo=eval_run) |
| **RoIPoly** | ISPRS 2025 | RoI query-based building polygon extraction with logit-guided vertex interaction. | [Paper](https://doi.org/10.1016/j.isprsjprs.2025.03.030) · [Code](https://github.com/HeinzJiao/RoIPoly) |

## Acknowledgements

This repository benefits from the excellent open-source contributions of [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN). We thank the authors for their great work.
