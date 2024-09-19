# PolyR-CNN: R-CNN for End-to-End Polygonal Building Outline Extraction

This is the official repository for **PolyR-CNN**, a model designed for end-to-end polygonal building outline extraction.

## Data Preprocessing

To minimize training time, we have preprocessed the CrowdAI dataset by localizing the polygon padding process to ensure a fixed number of vertices per polygon, reducing the need for on-the-fly computation during training. For detailed information on the polygon padding process, please refer to the article.

The preprocessed CrowdAI dataset can be downloaded [here](#) (Google Drive link to be provided).

To run the preprocessing yourself, you can execute the following command:

```bash
python3 preprocess_annotation.py --json_path /path/to/annotations.json --save_path /path/to/save/annotation_preprocessed.json --is_training True --num_corners 96
```

## Status

The remaining code is currently being organized and will be uploaded soon. Stay tuned for updates!

## Coming Soon

- Complete training and inference pipelines
- Example datasets and usage instructions
