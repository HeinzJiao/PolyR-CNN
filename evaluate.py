import json
import time
import numpy as np
from tqdm import tqdm
import os
import torch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from polyrcnn import PolyRCNNDatasetMapper
from demo import setup_cfg, get_parser, reduce_redundant_vertices, nms_bbox


def single_annotation(image_id, poly, bbox, score):
    # Convert bbox from [x_min, y_min, x_max, y_max] to [x, y, width, height]
    x_min, y_min, x_max, y_max = bbox
    bbox_xywh = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Create the annotation dictionary
    annotation = {
        "image_id": int(image_id),
        "category_id": 100,  # Fixed category ID for buildings
        "score": score,
        "segmentation": poly,
        "bbox": bbox_xywh
    }
    return annotation


def register_my_dataset(dataset_name, TEST_JSON, TEST_PATH):
    """Register your own COCO-format dataset.

       usage::
       from detectron2.data import DatasetCatalog, MetadataCatalog
       from detectron2.data.datasets.coco import load_coco_json

    :param TRAIN_JSON: the file path of the annotation
    :param TRAIN_PATH: the folder path of the images
    """
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(TEST_JSON, TEST_PATH, dataset_name))
    MetadataCatalog.get(dataset_name).set(json_file=TEST_JSON, image_root=TEST_PATH)


def main():
    """
    Perform predictions on the test dataset and save results in JSON format.
    """
    args = get_parser().parse_args()

    # Setup configuration and model
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Initiate the dataloader for the test dataset
    dataset_mapper = PolyRCNNDatasetMapper(cfg, is_train=False)
    dataloader = build_detection_test_loader(DatasetCatalog.get(cfg.DATASETS.TEST[0]), mapper=dataset_mapper)  # cfg.DATASETS.TEST = ('shanghai_test',)

    # Initialize progress bar for batch processing
    test_iterator = tqdm(dataloader, desc="Processing batches")

    speed = []
    predictions = []

    # Process each batch in the dataloader
    for i_batch, batched_inputs in enumerate(test_iterator):
        t0 = time.time()

        with torch.no_grad():
            predictions_per_batch = model(batched_inputs)

        polygons_per_batch = []  # Store predicted polygons
        bboxes_per_batch = []  # Store predicted bounding boxes
        scores_per_batch = []  # Store confidence scores

        # Process predictions for each image in the batch
        for i, predictions_per_image in enumerate(predictions_per_batch):
            image_id = batched_inputs[i]["image_id"]
            instances = predictions_per_image["instances"].to("cpu")
            instances = instances[instances.scores > args.confidence_threshold]
            num_instances = len(instances)
            if num_instances == 0:
                continue

            # Extract relevant predictions
            bbox_scores = instances.scores.numpy()
            bboxes = instances.pred_boxes.tensor.numpy()  # torch.Size([N, 4])
            corner_scores = instances.corner_scores.numpy()  # (N, M)
            N, M = corner_scores.shape
            contours = instances.pred_polygons.view(-1, M, 2)  # torch.Size([12, 96, 2])

            if args.nms_bbox:
                keep = nms_bbox(bboxes, bbox_scores, thres=0.5)
                corner_scores = corner_scores[keep]
                contours = contours[keep]
                bboxes = bboxes[keep]
                bbox_scores = bbox_scores[keep]

            # Post-process predictions to reduce redundant vertices
            pred_polygons_thres, pred_corners_scores, pred_bboxes, pred_bboxes_scores = reduce_redundant_vertices(
                corner_scores, contours, args.corner_threshold, bbox_scores, bboxes,
                nms=args.nms, merge=args.merge, nms_thres=2, merge_thres=10
            )

            # Append results for the current image
            bboxes_per_batch.append(pred_bboxes)
            scores_per_batch.append(pred_bboxes_scores)
            polygons_per_batch.append(pred_polygons_thres)

        # Measure processing time for the current batch
        speed.append(time.time() - t0)

        # Format predictions for saving
        for i, polygons_per_image in enumerate(polygons_per_batch):
            image_id = batched_inputs[i]["image_id"]
            bboxes_per_image = bboxes_per_batch[i]
            scores_per_image = scores_per_batch[i]

            for j, polygon in enumerate(polygons_per_image):
                bbox = bboxes_per_image[j].tolist()
                score = float(scores_per_image[j])
                polygon = list(np.array(polygon).flatten().astype(np.float64))

                # polygons with 2 vertices will be mistaken as bounding boxes and cause error in eval_coco.py
                if len(polygon) > 4:
                    predictions.append(single_annotation(image_id, [polygon], bbox, score))

    # Print average processing time per image
    avg_speed = np.mean(speed) / cfg.SOLVER.IMS_PER_BATCH
    print(f"Average model speed: {avg_speed:.4f} seconds per image")

    # Save predictions to JSON file
    output_path = os.path.join(args.output, "predictions.json")
    with open(output_path, "w") as fp:
        json.dump(predictions, fp)
    print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    register_my_dataset(dataset_name="aicrowd_test",
                        TEST_JSON="./data/aicrowd/val/annotations.json",
                        TEST_PATH="./data/aicrowd/val/images")
    main()
