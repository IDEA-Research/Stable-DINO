# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

import torch.nn.functional as F

from .two_stage_criterion import TwoStageCriterion

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, prob = None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    if prob is None:
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class StableDINOCriterion(TwoStageCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        ta_alpha: float = 0.0,
        ta_beta: float = 2.0,
        two_stage_binary_cls=False,
        use_ce_loss_type: str = "stable-dino",
        stg1_assigner=None,
        enc_kd_loss_weight: float = -1.0,
        enc_kd_loss_gamma: float = 2.0,
        target_post_process: str = "none"
    ):
        super(TwoStageCriterion, self).__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.two_stage_binary_cls = two_stage_binary_cls
        if self.two_stage_binary_cls:
            raise NotImplementedError

        # refer to task-aligned loss
        self.ta_alpha = ta_alpha
        self.ta_beta = ta_beta

        self.use_ce_loss_type = use_ce_loss_type

        self.stg1_assigner = stg1_assigner
        if stg1_assigner == "deta":
            raise NotImplementedError
        else:
            self.stg1_assigner_func = None


        self.enc_kd_loss_weight = enc_kd_loss_weight
        self.enc_kd_loss_gamma = enc_kd_loss_gamma

        self.target_post_process = target_post_process

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )

        losses = {"loss_class": loss_class}

        return losses

    def loss_labels_stabledino(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] # bs, nq, 80
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1] # bs, nq, 80

        focal_alpha = self.alpha
        focal_gamma = self.gamma

        out_prob = src_logits.sigmoid()
        bs, nq = src_logits.shape[:2]

        # get new target
        if self.use_ce_loss_type in ['stable-dino']:
            src_boxes = outputs["pred_boxes"][idx]  # (nbox, 4)
            tgt_bbox = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            tgt_labels = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_bbox))[0].diag() # (b*num_queries, ngt)
            _s = out_prob
            _u = torch.zeros_like(_s)
            _u[idx[0], idx[1], tgt_labels] = iou
            _t = _s.pow(self.ta_alpha) * _u.pow(self.ta_beta) 
            # (b, num_queries, num_classes) 
            # p**alpha * u**beta if pos, 0 if neg

            if self.target_post_process == "exp":
                _t = (_t.exp() - 1) / (torch.e - 1)
            elif self.target_post_process == "sin":
                _t = (_t * torch.pi / 2).sin()
            else:
                assert self.target_post_process == "none", self.target_post_process

        # get loss
        if self.use_ce_loss_type in ['stable-dino']:
            # refer to: Tal loss
            # we first shift the quality _t to larger than prob
            # follow the paper: TOOD: Task-aligned One-stage Object Detection.
            # link: https://readpaper.com/paper/3201828441 
            ngt_in_batch = [len(t["boxes"]) for t in targets]
            norm_t = torch.zeros_like(_t)

            all_out_bbox = outputs["pred_boxes"]
            all_tgt_bbox = torch.cat([v["boxes"] for v in targets]) # nbox
            all_iou = box_iou(box_cxcywh_to_xyxy(all_out_bbox.flatten(0, 1)), box_cxcywh_to_xyxy(all_tgt_bbox))[0].view(bs, nq, -1) # (b*num_queries, ngt)

            cum = 0
            for i in range(bs):
                if self.use_ce_loss_type == 'stable-dino':
                    max_iou = 1.0
                    # import ipdb; ipdb.set_trace()
                else:
                    if ngt_in_batch[i] == 0:
                        max_iou = 1.0
                    else:
                        max_iou = all_iou[i, :, cum:cum+ngt_in_batch[i]].max()
                # normalizer each item with the max iou in each batch
                normalizer = max((max_iou / (_t[i].max() + 1e-8)).detach(), 1)
                norm_t[i] = _t[i] * normalizer

                cum += ngt_in_batch[i]

            # must detach !
            norm_t = norm_t.detach()


            neg_loss = (1 - focal_alpha) * (out_prob**focal_gamma) * (1 - target_classes_onehot) * (-(1 - out_prob + 1e-8).log())
            pos_loss = target_classes_onehot * (
                focal_alpha * ((norm_t - out_prob)**focal_gamma) *  
                (-norm_t * out_prob.log() - (1 - norm_t) * (1 - out_prob + 1e-8).log())
            )
            loss_class = (pos_loss + neg_loss).sum() / num_boxes
        elif self.use_ce_loss_type == "none":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        else:
            assert self.use_ce_loss_type in ['none', 'stable-dino'], "use_ce_loss_type should be none or stable-dino"

        losses = {"loss_class": loss_class}


        return losses        


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes


        return losses


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        class_loss = self.loss_labels_stabledino
        is_dn = kwargs.get('isdn', False)
        if is_dn:
            class_loss = self.loss_labels
        loss_map = {
            "class": class_loss,
            "boxes": self.loss_boxes,
        }
        
        new_kwargs = {}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **new_kwargs)


    def forward(self, outputs, targets, dn_metas=None, return_matching_results=False, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(StableDINOCriterion, self).forward(outputs, targets)

        # import pdb;pdb.set_trace()
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs
                    )
                )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses
