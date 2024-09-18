import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
#from .losses import MaskRasterizationLoss
from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from scipy.optimize import linear_sum_assignment


class SetCriterion(nn.Module):
    """ This class computes the loss for PolyRCNN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth buildings and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise box and polygon)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            use_focal:
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.focal_loss_alpha = cfg.MODEL.PolyRCNN.ALPHA
        self.focal_loss_gamma = cfg.MODEL.PolyRCNN.GAMMA
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        #self.raster_loss = MaskRasterizationLoss(None)

    def loss_labels(self, outputs, targets, indices):
        """
        Compute box classification loss.

        Args:
            outputs (dict): Contains:
                - 'pred_logits' (torch.Tensor): (batch_size, num_proposals, num_classes) if using focal loss,
                    or (batch_size, num_proposals, num_classes + 1) for BCE
                - 'pred_boxes' (torch.Tensor): (bs, num_proposals, 4), absolute [x1, y1, x2, y2]
                - 'pred_corners' (torch.Tensor): (bs, num_proposals, num_corners)
                - 'pred_polygons' (torch.Tensor): (bs, num_proposals, num_corners * 2), absolute coordinates

            targets (list[dict]): One dict per image, each with:
                - 'labels' (torch.Tensor): (num_gt_boxes_i,)
                    ground truth box classes where reference boxes are class 0, others are class 1.
                - 'corners' (torch.Tensor): (num_gt_boxes_i, num_corners)
                    ground truth corner classes for each polygon where reference corners are class 0, others are class 1.
                - ...

            indices (list): Matched indices between predictions and targets.
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)  # (num_gt_boxes_batch,)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # (num_gt_boxes_batch)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # Labels of proposals matched to ground truth boxes are set to 0, the rest are set to 1.
        target_classes[idx] = target_classes_o  # (bs, num_proposals)

        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.
            class_loss = sigmoid_focal_loss_jit(
                src_logits,
                labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / idx[0].shape[0]
            losses = {'loss_ce': class_loss}
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {'loss_ce': loss_ce}

        return losses

    def loss_corners(self, outputs, targets, indices):
        """
        Compute corner classification loss.
        """
        use_focal_corner = True

        idx = self._get_src_permutation_idx(indices)  # batch_idx, src_idx
        src_corners = outputs['pred_corners'][idx]
        tgt_corners = torch.cat([t["corners"][J] for t, (_, J) in zip(targets, indices)])
        # src_corners, tgt_corners: (bs, num_gt_boxes_batch, num_corners)
        # The reference corners are marked as class 0, the others are marked as class 1.

        if use_focal_corner:  # True
            src_corners = torch.unsqueeze(src_corners, -1)
            src_corners = src_corners.flatten(0, 1)
            tgt_corners = tgt_corners.flatten(0, 1)
            pos_inds = torch.nonzero(tgt_corners != 1, as_tuple=True)[0]  # the indices of reference corners
            labels = torch.zeros_like(src_corners)
            labels[pos_inds.long(), tgt_corners[pos_inds.long()].long()] = 1  # change the labels of reference corners to 1
            # comp focal loss.
            loss_cor = sigmoid_focal_loss_jit(
                src_corners,
                labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / idx[0].shape[0]
        else:
            mask = torch.nonzero(tgt_corners.flatten() == 1)
            # In the ground truth (tgt_corners), reference corners are marked as class 0 and others as class 1.
            # For BCE loss, we need to switch the classes: set non-reference corners (class 1) to 0 and vice versa.
            tgt_corners_ = torch.ones(tgt_corners.flatten().shape, device=src_corners.device)
            tgt_corners_[mask] = 0
            tgt_corners_ = tgt_corners_.view(tgt_corners.shape)
            loss_cor = F.binary_cross_entropy_with_logits(src_corners, tgt_corners_.type(torch.float32), torch.tensor(10))

        losses = {'loss_cor': loss_cor}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """
        Compute the L1 regression loss and GIoU loss for bounding boxes.

        Args:
            indices (list of tuples): Contains Hungarian matching information between predictions and targets.
                Each element is a tuple (row_ind, col_ind) for an image in the batch:
                - row_ind (torch.Tensor): Indices of the matched proposals, shape (num_gt_boxes_i,)
                - col_ind (torch.Tensor): Indices of the matched ground truth boxes, shape (num_gt_boxes_i,)
                  For the i-th image in the batch, num_gt_boxes_i is the number of ground truth boxes in the image.
                  For example, the gt_box at col_ind is matched with the proposal at row_ind.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)  # batch_idx, src_idx
        src_boxes = outputs['pred_boxes'][idx]
        # src_boxes: (num_gt_boxes_batch, 4),
        # the predicted proposal boxes matched to the ground truth boxes in the batch.
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_boxes: (num_gt_boxes_batch, 4),
        # the ground truth boxes corresponding to the matched src_boxes in the batch.
        # src_boxes and target_boxes are both absolute coordinates.
        image_names = [t['image_name'] for t in targets]

        losses = {}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / idx[0].shape[0]

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size
        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / idx[0].shape[0]

        return losses

    def loss_polygons(self, outputs, targets, indices):
        """Compute the L1 regression loss for polygons.
        """
        idx = self._get_src_permutation_idx(indices)
        src_polygons = outputs['pred_polygons'][idx]  # (num_gt_boxes_batch, 192)
        target_polygons = torch.cat([t['polygons_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        image_size = torch.cat([v["image_size_xyxy_tgt_96"] for v in targets])
        src_polygons_ = src_polygons / image_size
        target_polygons_ = target_polygons / image_size
        loss_polygon = F.l1_loss(src_polygons_, target_polygons_, reduction='none')
        losses['loss_polygon'] = loss_polygon.sum() / idx[0].shape[0]

        # To be implemented ...
        # loss_raster_mask = self.raster_loss(src_polygons, target_polygons)
        # src_polygons: torch.Size([num_gt_boxes_batch, num_corners*2=192])
        # target_polygons: torch.Size([num_gt_boxes_batch, num_corners*2=192])
        # losses['loss_raster'] = loss_raster_mask

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # torch.full_like(src, i): (num_gt_boxes_i,) each element=i
        # batch_idx: (num_gt_boxes_batch,) index the image id of each gt_box in the batch e.g. [0,0,...,1,1,...,2,2,...]
        src_idx = torch.cat([src for (src, _) in indices])
        # src=row_ind: (num_gt_boxes_i,) e.g [0, 1, 4] the indices of proposals matched to corresponding gt_boxes
        # src_idx: (num_gt_boxes_batch,) the indices of proposals in all the images of the batch in sequence
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'corners': self.loss_corners,
            'boxes': self.loss_boxes,
            'polygons': self.loss_polygons
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Args
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)  # the total number of target boxes in one batch
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class  # class_weight=2.0
        self.cost_bbox = cost_bbox  # l1_weight=5.0
        self.cost_giou = cost_giou  # giou_weight=2.0
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.PolyRCNN.ALPHA  # 0.25
            self.focal_loss_gamma = cfg.MODEL.PolyRCNN.GAMMA  # 2.0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Args:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]  # (bs, num_proposals, num_classes)

        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])  # (num_gt_boxes_batch)
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])  # (num_gt_boxes_batch, 4)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha  # 0.25
            gamma = self.focal_loss_gamma  # 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])  # (bs, 4)
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)  # (bs*num_proposals, 4)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])  # (num_gt_boxes_batch, 4)

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)  # (bs*num_proposals, num_gt_boxes_batch)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()   # (bs, num_proposals, num_gt_boxes_batch)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # i: the image id in the batch
        # c: (bs, num_proposals, num_gt_boxes_i) num_gt_boxes_i: the num of gt boxes of the ith img
        # c[i]: (num_proposals, num_gt_boxes_i) the cost matrix of the ith image in the batch
        # linear_sum_assignment(c[i]): row_ind, col_ind
        # row_ind: (num_gt_boxes_i,) e.g [0, 1, 4] the indices of proposals
        # col_ind: (num_gt_boxes_i,) e.g. [1, 0, 2] the indices of gt_boxes
        # the gt_box with id 1/0/2 is assigned to the proposal with id 0/1/4
        # indices = [(row_ind, col_ind), ... ] len=batch_size
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

