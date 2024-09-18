from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

__all__ = ["PolyRCNN"]


@META_ARCH_REGISTRY.register()
class PolyRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  # ["p2", "p3", "p4", "p5"]
        self.num_classes = cfg.MODEL.PolyRCNN.NUM_CLASSES  # 1
        self.num_proposals = cfg.MODEL.PolyRCNN.NUM_PROPOSALS
        self.num_corners = cfg.MODEL.PolyRCNN.NUM_CORNERS
        self.hidden_dim = cfg.MODEL.PolyRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.PolyRCNN.NUM_HEADS  # 6
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        # self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)  # hidden_dim=256
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)  # initialize a (self.num_proposals, 4) tensor
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)  # x_c, y_c, w, h
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)  # [0.5, 0.5, 1, 1]

        # Build proposal polygons:
        self.init_proposal_polygons = nn.Embedding(self.num_proposals, self.num_corners * 2)
        polygon_reg = torch.tensor(cfg.INIT_POLYGON, dtype=torch.float64)
        for i in range(self.num_corners * 2):
            nn.init.constant_(self.init_proposal_polygons.weight[:, i], polygon_reg[i])
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.PolyRCNN.CLASS_WEIGHT  # 2
        giou_weight = cfg.MODEL.PolyRCNN.GIOU_WEIGHT  # 2
        l1_weight = cfg.MODEL.PolyRCNN.L1_WEIGHT  # 5
        no_object_weight = cfg.MODEL.PolyRCNN.NO_OBJECT_WEIGHT  # 0.1
        self.deep_supervision = cfg.MODEL.PolyRCNN.DEEP_SUPERVISION  # True
        self.use_focal = cfg.MODEL.PolyRCNN.USE_FOCAL  # True

        # Build HungarianMatcher.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)

        weight_dict = {"loss_ce": class_weight,  # 2
                       "loss_bbox": l1_weight,  # 5
                       "loss_giou": giou_weight,  # 2
                       "loss_cor": 1,
                       "loss_polygon": 5,
                       }
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "corners", "polygons"]

        # Build Criterion.
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)  # This class computes the loss.

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                * file_name: "./datasets/AIcrowd/train/images/....jpg"
        """

        images, images_whwh = self.preprocess_image(batched_inputs)  # images_whwh: torch.Tensor of shape [batch_size, 4], [[width, height, width, height], ...]
        if isinstance(images, (list, torch.Tensor)):  # False
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)  # images.tensor.shape: torch.Size([batch_size, channels, height, width])
        features = list()        
        for f in self.in_features:  # self.in_features: ["p2", "p3", "p4", "p5"]
            feature = src[f]
            features.append(feature)

        # Prepare Proposal Boxes.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        bs, num_proposals, _ = proposal_boxes.shape

        # Prepare Proposal Polygons.
        proposal_polygons = self.init_proposal_polygons.weight.clone()
        proposal_polygons = proposal_polygons[None] * images_whwh.repeat(1, int(self.num_corners / 2))[:, None, :]

        # Prediction.
        outputs_class, outputs_coord, outputs_corners, outputs_polygons = self.head(features,
                                                                                    proposal_boxes,
                                                                                    proposal_polygons,
                                                                                    None,  # pro_features
                                                                                    images_whwh)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                  'pred_corners': outputs_corners[-1], 'pred_polygons': outputs_polygons[-1]}
        # The output from the last iteration:
        # - outputs_class[-1]: torch.Size([batch_size, num_proposals, num_classes (focal) or num_classes + 1 (BCE)])
        # - outputs_coord[-1]: torch.Size([batch_size, num_proposals, 4]), representing [x1, y1, x2, y2] in absolute coordinates
        # - outputs_corners[-1]: torch.Size([batch_size, num_proposals, num_corners]), representing corner labels
        # - outputs_polygons[-1]: torch.Size([batch_size, num_proposals, num_corners*2), representing absolute corner coordinates

        if self.training:
            gt_instances = [x["instances"] for x in batched_inputs]
            gt_names = [x["file_name"] for x in batched_inputs]
            # Prepare Ground Truth.
            targets = self.prepare_targets(gt_instances, gt_names)
            # Predictions.
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'pred_polygons': d}
                                         for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1],
                                                               outputs_corners[:-1], outputs_polygons[:-1])]
            # Calculate Losses.
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]

            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            polygon_pred = output["pred_polygons"]
            corner_cls = output["pred_corners"]
            results = self.inference(box_cls, box_pred, images.image_sizes, polygon_pred, corner_cls)
            if do_postprocess:  # True
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

    def prepare_targets(self, targets, names):
        new_targets = []
        for i in range(len(targets)):
            targets_per_image = targets[i]
            image_name = names[i]
            target = {}

            boxes = targets_per_image.gt_boxes.tensor  # torch.Size([bs, 4]), xyxy, absolute bounding box coordinates
            gt_classes = targets_per_image.gt_classes  # torch.Size([bs])
            encoded_polygons = targets_per_image.gt_masks  # torch.Size([bs, num_corners*2]), preprocessed and uniformly sampled polygon vertices
            corners = targets_per_image.gt_cor_cls_img  # torch.Size([bs, num_corners]), the labels for each polygon vertex

            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            gt_boxes = boxes / image_size_xyxy  # xyxy, normalized
            target["image_name"] = image_name
            target["height"] = h
            target["weight"] = w
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = box_xyxy_to_cxcywh(gt_boxes).to(self.device)  # torch.Size([bs, 4]), cxcywh, normalized
            target["boxes_xyxy"] = boxes.to(self.device)  # torch.Size([bs, 4]), xyxy, absolute coordinates
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target['polygons_xyxy'] = encoded_polygons.to(self.device)  # torch.Size([bs, num_corners*2])
            image_size_xyxy_tgt_96 = image_size_xyxy_tgt.repeat(1, int(self.num_corners / 2))
            target["image_size_xyxy_tgt_96"] = image_size_xyxy_tgt_96.to(self.device)  # torch.Size([bs, num_corners*2])
            target['corners'] = corners.to(self.device)  # torch.Size([bs, num_corners])
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes, polygon_pred, corner_cls):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, num_classes or num_classes+1).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x1,y1,x2,y2) box regression values for every proposal.
            image_sizes (List[torch.Size]): the input image sizes
            polygon_pred (Tenser): (batch_size, num_proposals, num_corners*2).
                The tensor predicts the polygon vertex coordinates.
            corner_cls (Tensor): (batch_size, num_proposals, 96).
                The tensor predicts the probability of each vertex being a corner.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            # The tensor predicts the classification probability for each proposal.
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
            # labels: torch.Tensor of shape (num_classes * num_proposals)
            # Contains repeated class indices in the form tensor([0, ..., num_classes-1, 0, ..., num_classes-1, ...])

            bs, _, _ = corner_cls.shape
            corner_cls = corner_cls.view(bs, self.num_proposals, self.num_corners)
            corner_scores = F.sigmoid(corner_cls)
            # Focal loss --> sigmoid
            # binary cross entropy --> sigmoid
            # multi-class cross entropy --> softmax

            for i, (scores_per_image, box_pred_per_image, image_size, polygon_pred_per_image,
                    corner_scores_per_image) in enumerate(zip(
                    scores, box_pred, image_sizes, polygon_pred, corner_scores
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                # topk selects the top num_proposals scores and records their indices in the flattened tensor
                # scores_per_image: torch.Size([num_proposals]), topk_indices: torch.Size([num_proposals])
                labels_per_image = labels[topk_indices]
                # torch.Size([num_proposals]), record the class labels of the top num_proposals scores
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]
                polygon_pred_per_image = polygon_pred_per_image.view(-1, 1, self.num_corners*2).repeat(1, self.num_classes, 1).view(-1, self.num_corners*2)
                polygon_pred_per_image = polygon_pred_per_image[topk_indices]
                corner_scores_per_image = corner_scores_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.pred_polygons = polygon_pred_per_image  # torch.Size([num_proposals, 192])
                result.corner_scores = corner_scores_per_image

                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)  # box_cls: (bs, num_proposals, num_classes+1)
            # scores: (bs, num_proposals) labels: (bs, num_proposals)
            # multi-class cross entropy: softmax
            # binary cross entropy: sigmoid
            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)  # (num_proposals, 4) [x1, y1, x2, y2] absolute coordinates
                result.scores = scores_per_image  # (num_proposals)
                result.pred_classes = labels_per_image  # (num_proposals)
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]  # (bs, c, h, w)
        images = ImageList.from_tensors(images, self.size_divisibility)  # self.size_divisibility=32
        # size_divisibility: add padding to ensure the common height and width is divisible by 'size_divisibility'.
        # This depends on the model and many models need a divisibility of 32.

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)  # (bs, 4)

        return images, images_whwh