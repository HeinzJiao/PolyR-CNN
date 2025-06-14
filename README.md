# PolyR-CNN: R-CNN for End-to-End Polygonal Building Outline Extraction

This is the official repository for **PolyR-CNN**, a model designed for end-to-end polygonal building outline extraction.

## Data Preprocessing

To minimize training time, we have preprocessed the CrowdAI dataset by localizing the polygon padding process to ensure a fixed number of vertices per polygon, reducing the need for on-the-fly computation during training. For detailed information on the polygon padding process, please refer to the article.

To run the preprocessing yourself, you can execute the following command:

```bash
python3 preprocess_annotation.py --json_path /path/to/annotations.json --save_path /path/to/save/annotation_preprocessed.json --is_training True --num_corners 96
```
## Environment Setup



## Training

To start training the PolyR-CNN model on the CrowdAI dataset, you can use the following command:

```bash
python3 train_net.py --num-gpus <number_of_gpus> --config-file configs/polyrcnn.res50.100pro.aicrowd.yaml
```

You can use different configuration files from the [`configs/`](./configs/) folder depending on your requirements (e.g., switching to ResNet101 or Swin Transformer backbones).

## Testing

To test the PolyR-CNN model on a single image or all images in a folder and visualize the results, you can use the demo.py script. Refer to the arguments in the [`demo.py`](./demo.py/) script for detailed usage.

To evaluate the model on a complete COCO-format dataset, use the [`evaluate.py`](./evaluate.py/) script. The predicted results will be saved as a COCO-format prediction file.

For detailed metric evaluation, such as MS-COCO metrics, PoLiS, etc., refer to the evaluation tools provided in the [HiSup repository](https://github.com/SarahwXU/HiSup). Specifically, you can use the tools/evaluation.py script in HiSup to test COCO-format datasets.

## 🚀 New Paper Alert

This repository is part of a broader research effort on vectorized object outline extraction for large-scale topographic map generation. Check out our latest publications:

- **RoIPoly: Vectorized Building Outline Extraction Using Vertex and Logit Embeddings**  
  *ISPRS Journal of Photogrammetry and Remote Sensing*, 2025  
  📄 [Paper (DOI)](https://doi.org/10.1016/j.isprsjprs.2025.03.030) ｜ 💻 [Official Code](https://github.com/HeinzJiao/RoIPoly)  
  → Introduces a transformer-based RoI polygon decoder with efficient vertex-level interaction and logit-guided embedding, enabling direct and redundancy-free polygon generation that eliminates the need for post-processing — setting new SOTA on CrowdAI.

