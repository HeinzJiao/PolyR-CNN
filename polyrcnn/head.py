import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.PolyRCNN.NUM_CLASSES  # 1
        num_corners = cfg.MODEL.PolyRCNN.NUM_CORNERS  #
        d_model = cfg.MODEL.PolyRCNN.HIDDEN_DIM  # 256
        dim_feedforward = cfg.MODEL.PolyRCNN.DIM_FEEDFORWARD  # 2048
        nhead = cfg.MODEL.PolyRCNN.NHEADS  # 8 multi-head self-attention
        dropout = cfg.MODEL.PolyRCNN.DROPOUT  # 0.0
        activation = cfg.MODEL.PolyRCNN.ACTIVATION
        num_heads = cfg.MODEL.PolyRCNN.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, num_corners, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_corners = num_corners
        self.return_intermediate = cfg.MODEL.PolyRCNN.DEEP_SUPERVISION  # True
        
        # Init parameters.
        self.use_focal = cfg.MODEL.PolyRCNN.USE_FOCAL
        self.num_classes = num_classes  # 1
        if self.use_focal:
            prior_prob = cfg.MODEL.PolyRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  # .in_features ["p2", "p3", "p4", "p5"]
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION  # 7
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_polygons, init_features, images_whwh):
        """
        Args:
            init_bboxes (torch.Tensor): Tensor of shape (batch_size, num_proposals, 4),
                containing the bounding boxes in [x1, y1, x2, y2] format with absolute coordinates.
            init_features (None): Currently not used, set to None.
            init_polygons (torch.Tensor): Tensor of shape (batch_size, num_proposals, num_corners * 2),
                containing the absolute coordinates of polygon vertices.
        """
        inter_class_logits = []
        inter_pred_bboxes = []
        inter_corners_logits = []
        inter_pred_polygons = []

        # Prepare Proposal Boxes, Proposal Features and Proposal Polygons.
        bs = len(features[0])
        bboxes = init_bboxes
        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)  # (1, num_proposals*bs, hidden_dim=256)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        polygons = init_polygons
        
        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, corner_logits, pred_polygons = rcnn_head(features,
                                                                                bboxes,
                                                                                polygons,
                                                                                proposal_features,
                                                                                self.box_pooler,
                                                                                images_whwh)

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                inter_corners_logits.append(corner_logits)
                inter_pred_polygons.append(pred_polygons)

            # Update Proposal Boxes and Proposal Polygons for the Next Head.
            bboxes = pred_bboxes.detach()
            polygons = pred_polygons.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), torch.stack(inter_corners_logits), \
                   torch.stack(inter_pred_polygons)

        return class_logits[None], pred_bboxes[None], corner_logits[None], pred_polygons[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, num_corners, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model  # 256
        self.num_corners = num_corners
        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 256, 2048
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 2048, 256

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # This MLP is used to generate the vertex proposal feature for each polygon from its vertex coordinates.
        self.corner_mlp = nn.Sequential(nn.Linear(self.num_corners * 2, self.d_model),
                                        nn.GELU(),
                                        nn.Linear(self.d_model, self.d_model))

        # box cls. (box classfication head: a linear projection)
        num_cls = cfg.MODEL.PolyRCNN.NUM_CLS  # 1
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # box reg. (box regression head: a 3-layer perceptron)
        num_reg = cfg.MODEL.PolyRCNN.NUM_REG  # 3
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # corner cls. (corner classification head: a 3-layer perceptron)
        num_cor = cfg.MODEL.PolyRCNN.NUM_COR  # 3
        cor_module = list()
        for _ in range(num_cor):
            cor_module.append(nn.Linear(d_model, d_model, False))
            cor_module.append(nn.LayerNorm(d_model))
            cor_module.append(nn.ReLU(inplace=True))
        self.cor_module = nn.ModuleList(cor_module)

        # polygon reg. (polygon regression head: a 3-layer perceptron)
        num_pol = cfg.MODEL.PolyRCNN.NUM_POL  # 3
        pol_module = list()
        for _ in range(num_pol):
            pol_module.append(nn.Linear(d_model, d_model, False))
            pol_module.append(nn.LayerNorm(d_model))
            pol_module.append(nn.ReLU(inplace=True))
        self.pol_module = nn.ModuleList(pol_module)
        
        # pred.
        self.use_focal = cfg.MODEL.PolyRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.corner_logits = nn.Linear(d_model, self.num_corners)  # same for either focal or binary cross entropy loss
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.polygons_delta = nn.Linear(d_model, self.num_corners * 2)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights  # (2.0, 2.0, 1.0, 1.0)

    def forward(self, features, bboxes, polygons, pro_features, pooler, images_whwh):
        """
        Args:
            features (list): List of feature maps from different levels of the backbone.
            bboxes (torch.Tensor): Tensor of shape (batch_size, num_proposals, 4) containing absolute bounding box coordinates
                in the format xyxy.
            polygons (torch.Tensor): Tensor of shape (batch_size, num_proposals, num_corners * 2) containing absolute
                polygon vertex coordinates.
            pro_features (torch.Tensor): Tensor of shape (num_proposals, batch_size, hidden_dim=256) representing the proposal features.
        """

        bs, num_proposals = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(bs):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(bs * num_proposals, self.d_model, -1).permute(2, 0, 1)  # torch.Size([7*7, bs*num_proposals, 256])

        if pro_features is None:
            pro_features = polygons / images_whwh.repeat(1, int(self.num_corners / 2))[:, None, :]
            pro_features = (pro_features * 2.) - 1.  # range from -1 to 1
            pro_features = pro_features.to(torch.float32)
            pro_features = self.corner_mlp(pro_features.view(bs * num_proposals, -1)).view(bs, num_proposals, -1)  # torch.Size([bs, num_proposals, 256])

        # self_att.
        pro_features = pro_features.view(bs, num_proposals, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)  # torch.Size([num_proposals, bs, 256])

        # inst_interact.
        pro_features = pro_features.view(num_proposals, bs, self.d_model).permute(1, 0, 2).reshape(1, bs * num_proposals, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)  # (1, bs*num_proposals, 256)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)  # (1, bs*num_proposals, 256)
        
        fc_feature = obj_features.transpose(0, 1).reshape(bs, num_proposals, -1)

        # self_attn.
        fc_feature = fc_feature.permute(1, 0, 2)
        fc_feature2 = self.self_attn1(fc_feature, fc_feature, value=fc_feature)[0]
        fc_feature = fc_feature + self.dropout4(fc_feature2)
        fc_feature = self.norm4(fc_feature)  # (num_proposals, bs, 256)

        fc_feature = fc_feature.transpose(0, 1).reshape(bs * num_proposals, -1)

        # prediction heads
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        cor_feature = fc_feature.clone()
        pol_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        for cor_layer in self.cor_module:
            cor_feature = cor_layer(cor_feature)
        for pol_layer in self.pol_module:
            pol_feature = pol_layer(pol_feature)

        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        corner_logits = self.corner_logits(cor_feature)
        polygons_deltas = self.polygons_delta(pol_feature)
        pred_polygons = self.apply_polydeltas(polygons_deltas, bboxes.view(-1, 4), polygons.view(-1, self.num_corners*2))

        return class_logits.view(bs, num_proposals, -1), pred_bboxes.view(bs, num_proposals, -1), \
            corner_logits.view(bs, num_proposals, -1), pred_polygons.view(bs, num_proposals, -1)

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (bs, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (bs, 4) xyxy
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths  # (N*nr_boxes,)
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights  # [2.0, 2.0, 1.0, 1.0]
        dx = deltas[:, 0::4] / wx  # (N*nr_boxes, k=1)
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

    def apply_polydeltas(self, deltas, boxes, polygons):
        """
        Apply the transformation `deltas` (dx1, dy1, dx2, dy2, ...) to the given `polygons`.

        Args:
            deltas (torch.Tensor): Transformation deltas of shape (bs * num_proposals, num_corners * 2),
                where each element represents the change in x and y for each vertex of the polygon.
            polygons (torch.Tensor): Original polygon coordinates of shape (bs * num_proposals, num_corners * 2),
                in the format (x1, y1, x2, y2, ...).
        """
        polygons = polygons.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]  # (bs*num_proposals, )
        heights = boxes[:, 3] - boxes[:, 1]  # (bs*num_proposals, )

        x = polygons[:, 0::2]  # (bs*num_proposals, num_corners)
        y = polygons[:, 1::2]  # (bs*num_proposals, num_corners)

        dx = deltas[:, 0::2]  # (bs*num_proposals, num_corners)
        dy = deltas[:, 1::2]  # (bs*num_proposals, num_corners)

        pred_x = dx * widths[:, None] + x
        pred_y = dy * heights[:, None] + y

        pred_polygons = torch.zeros_like(deltas)
        pred_polygons[:, 0::2] = pred_x
        pred_polygons[:, 1::2] = pred_y

        return pred_polygons


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.PolyRCNN.HIDDEN_DIM  # 256
        self.dim_dynamic = cfg.MODEL.PolyRCNN.DIM_DYNAMIC  # 64
        self.num_dynamic = cfg.MODEL.PolyRCNN.NUM_DYNAMIC  # 2
        self.num_params = self.hidden_dim * self.dim_dynamic  # 256*64
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)  # 2*256*64

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2  # 256*49
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)  # [bs*num_proposals, 1, 2*256*64]

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)  # [bs*num_proposals, 256, 64]
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)  # [bs*num_proposals, 64, 256]

        features = torch.bmm(features, param1)  # [bs*num_proposals, 49, 256], [bs*num_proposals, 256, 64] -> [bs*num_proposals, 49, 64]
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)  # [bs*num_proposals, 49, 64], [bs*num_proposals, 64, 256] -> [bs*num_proposals, 49, 256]
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)  # [bs*num_proposals, 49*256]
        features = self.out_layer(features)  # [bs*num_proposals, 49*256] -> [bs*num_proposals, 256]
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
