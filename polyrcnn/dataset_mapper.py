import copy
import logging

import numpy as np
import torch
import pycocotools.mask as mask_util
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import cv2
from util.poly_ops import is_clockwise, resort_corners_and_labels

__all__ = ["PolyRCNNRCNNDatasetMapper"]


def build_transform_gen(is_train):
    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        # You can define your own data augmentation methods here.
        # Currently, only RandomFlip is applied.
        tfm_gens.append(T.RandomFlip())  # Flip the image horizontally or vertically with the given probability.
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class PolyRCNNDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by PolyRCNN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        self.crop_gen = None

        self.tfm_gens = build_transform_gen(is_train)  # RandomFlip()
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        self.num_corners = cfg.MODEL.PolyRCNN.NUM_CORNERS

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations']

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # obj is a dictionary with the following keys:
            # - 'iscrowd': <int>, indicates if the annotation contains more than one polygon
            # - 'bbox': <list>, [x0, y0, x1, y1] coordinates, values are float64
            # - 'category_id': <int>, the category label for the object
            # - 'segmentation': <list>, [[x0, y0, x1, y1, ..., x_n, y_n]] polygon points, values are int64
            # - 'cor_cls_poly': [array], array of shape (num_corners,), labels for each vertex of the polygon, values are int64
            # - 'bbox_mode': <BoxMode enum>, BoxMode.XYXY_ABS, indicates the mode for bounding box coordinates

            for anno in annos:
                anno['segmentation'][0], anno["cor_cls_poly"] = resort_corners_and_labels(anno['segmentation'][0],
                                                                                          anno["cor_cls_poly"])

            instances = annotations_to_instances(annos, image_shape, self.num_corners)
            # instances is an object with the following attributes:
            # - instances.gt_boxes: <Boxes>, torch.Tensor of shape (N, 4), dtype=torch.float32, contains the bounding box coordinates
            # - instances.gt_classes: <torch.Tensor>, torch.int64, shape (N), contains the class labels for each instance
            # - instances.gt_cor_cls_img: <torch.Tensor>, torch.int32, shape (N, num_corners), contains the labels for each polygon vertex
            # - instances.gt_masks: <torch.Tensor>, torch.int32, shape (N, num_corners*2), preprocessed and uniformly sampled polygon vertices
            dataset_dict["instances"] = filter_empty_instances(instances)

        return dataset_dict

# The following functions are a rewritten version of the original functions from detectron2.
# They have been modified to adapt to the custom dataset format with specific handling of polygon corners
# and other data structures required for this project.
def annotations_to_instances(annos, image_size, num_corners):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance: dict.keys = ['iscrowd', 'bbox', 'category_id', 'segmentation', 'bbox_mode',
                                                   'cor_cls_poly']
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_masks", "gt_corner_classes".
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    if len(annos):
        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes  # torch.Size([N]), int64
    else:
        target.gt_classes = torch.zeros(torch.Size([0]), dtype=torch.int32)

    if len(annos):
        cor_cls_img = [obj["cor_cls_poly"] for obj in annos]
        cor_cls_img = torch.tensor(cor_cls_img, dtype=torch.int32)
        target.gt_cor_cls_img = cor_cls_img  # torch.Size([N, num_corners]), torch.int32
    else:
        target.gt_cor_cls_img = torch.zeros((0, num_corners), dtype=torch.int32)

    if len(annos) and "segmentation" in annos[0]:
        segms = np.stack([obj["segmentation"][0] for obj in annos])
        masks = torch.tensor(segms, dtype=torch.int32)
        target.gt_masks = masks  # torch.Size([N, num_corners * 2]), torch.int32
    else:
        target.gt_masks = torch.zeros((0, num_corners * 2), dtype=torch.int32)

    return target


def filter_empty_instances(
    instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False
):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(nonempty(instances.gt_masks))

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m | x
    if return_mask:
        return instances[m], m
    return instances[m]


def nonempty(polygons):
    """
    :param polygons: torch.Size([N, num_corners * 2])
    """
    keep = [1 if polygon.shape[0] > 0 else 0 for polygon in polygons]
    return torch.from_numpy(np.asarray(keep, dtype=bool))
