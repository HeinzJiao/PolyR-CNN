# PolyR-CNN: R-CNN for End-to-End Polygonal Building Outline Extraction

This is the official repository for **PolyR-CNN**, a model designed for end-to-end polygonal building outline extraction.

## Data Preprocessing

To minimize training time, we have preprocessed the CrowdAI dataset by localizing the polygon padding process to ensure a fixed number of vertices per polygon, reducing the need for on-the-fly computation during training. For detailed information on the polygon padding process, please refer to the article.

The preprocessed CrowdAI dataset can be downloaded [here](#) (Google Drive link to be provided).

To run the preprocessing yourself, you can execute the following command:

```bash
python3 preprocess_annotation.py --json_path /path/to/annotations.json --save_path /path/to/save/annotation_preprocessed.json --is_training True --num_corners 96
```

## Training

To start training the PolyR-CNN model on the CrowdAI dataset, you can use the following command:

```bash
python3 train_net.py --num-gpus <number_of_gpus> --config-file configs/polyrcnn.res50.100pro.aicrowd.yaml
```

You can use different configuration files from the [`configs/`](./configs/) folder depending on your requirements (e.g., switching to ResNet101 or Swin Transformer backbones).

## Status

The remaining code is currently being organized and will be uploaded soon. Stay tuned for updates!

## The paper is accepted by the ISPRS Journal of Photogrammetry and Remote Sensing (IF 10.6)！

## Coming Soon

- Complete inference pipelines
