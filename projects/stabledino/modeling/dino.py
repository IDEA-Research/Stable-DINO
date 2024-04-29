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


import copy
import math
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.layers.nms import batched_nms

from ..utils.box_ops import box_iou_pairwise, generalized_box_iou_pairwise, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        gdn_k: int = 2,
        neg_step_type: str = "none",
        no_img_padding: bool = False,
        dn_to_matching_block: bool = False,
        dn_select_min_noise_as_pos: bool = False,
        dn_select_1st_noise_as_pos: bool = False,
        nms_thresh: float = -1.0,
        dn_detach_dn2matching: bool = False,
        watch_center_in_box: bool = False,
        decoder_input_loss: bool = False,
        encoder_ckpt_path: str = 'none',
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.no_img_padding = no_img_padding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.watch_center_in_box = watch_center_in_box
        self.decoder_input_loss = decoder_input_loss

        # denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.gdn_k = gdn_k
        self.neg_step_type = neg_step_type
        assert self.neg_step_type in ["none", "multipy"] or self.neg_step_type.startswith("add"), \
            "neg_step_type should be in ['none', 'multipy', 'add:k', 'addonce:k', 'addoncexy:k', 'addxy:k']"
        self.dn_to_matching_block = dn_to_matching_block
        self.dn_select_min_noise_as_pos = dn_select_min_noise_as_pos
        self.dn_select_1st_noise_as_pos = dn_select_1st_noise_as_pos
        if self.dn_select_1st_noise_as_pos:
            assert self.dn_select_min_noise_as_pos, "dn_select_1st_noise_as_pos should be used with dn_select_min_noise_as_pos"
        self.dn_detach_dn2matching = dn_detach_dn2matching

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        num_pred_class_embed = num_pred
        if self.decoder_input_loss:
            num_pred_class_embed += 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred_class_embed)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # nms_thresh
        self.nms_thresh = nms_thresh


        # aug weight dict
        base_weight_dict = copy.deepcopy(self.criterion.weight_dict)
        if self.aux_loss:
            weight_dict = self.criterion.weight_dict
            aux_weight_dict = {}
            aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
            aux_length = self.transformer.decoder.num_layers - 1
            if self.decoder_input_loss:
                aux_length += 1
            for i in range(aux_length):
                aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            self.criterion.weight_dict = weight_dict     
        if hasattr(self.criterion, "enc_kd_loss_weight"):
            self.criterion.weight_dict['loss_kd'] = self.criterion.enc_kd_loss_weight

        # load pre-trained encoder
        self.encoder_ckpt_path = encoder_ckpt_path
        self.load_encoder()

    def load_encoder(self):
        if self.encoder_ckpt_path == "none":
            return
        if not os.path.exists(self.encoder_ckpt_path):
            raise FileNotFoundError(f"encoder ckpt {self.encoder_ckpt_path} not found")
        
        print(f"Loading encoder from {self.encoder_ckpt_path} ...")
        encoder_ckpt = torch.load(self.encoder_ckpt_path, map_location="cpu")
        _load_res = self.load_state_dict(encoder_ckpt, strict=False)
        print(f"Loading encoder from {self.encoder_ckpt_path} done")
        missing_keys = _load_res.missing_keys
        missing_keys = [k for k in missing_keys if "encoder" in k]
        unexpected_keys = _load_res.unexpected_keys
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        # import ipdb; ipdb.set_trace()


    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)


        if self.training:
            batch_size, _, H, W = images.tensor.shape
            if self.no_img_padding:
                img_masks = images.tensor.new_zeros(batch_size, H, W)
            else:
                img_masks = images.tensor.new_ones(batch_size, H, W)
                for img_id in range(batch_size):
                    img_h, img_w = batched_inputs[img_id]["instances"].image_size
                    img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (   
            init_state,
            inter_states,
            init_reference,  # [0..1]
            inter_references,  # [0..1]
            enc_state,
            enc_reference,  # [0..1]
            dn_content_query, 
            dn_anchor_boxes, # [0..1]
            anchors,
            enc_outputs_class,
            enc_outputs_coord,
            enc_memory,
            backbone_memory,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            dn_meta=dn_meta,
        )
        # import ipdb; ipdb.set_trace()

        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        if self.transformer.encoder_roi_pooling_layer is not None:
            inter_states[0] += self.transformer.encoder_roi_pooling_layer.sum_parameter() * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.watch_center_in_box:
            init_boxes_list = []
            init_boxes_list.append(init_reference)
            init_boxes_list.extend(outputs_coord[:-1])
            assert len(init_boxes_list) == len(outputs_coord), f"{len(init_boxes_list)} vs {len(outputs_coord)}"
            output["init_boxes"] = init_boxes_list[-1]
            if self.aux_loss:
                for i in range(len(init_boxes_list) - 1):
                    output["aux_outputs"][i]["init_boxes"] = init_boxes_list[i]

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[self.transformer.decoder.num_layers](enc_state)
        output["enc_outputs"] = {
            "pred_logits": interm_class, 
            "pred_boxes": interm_coord,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord": enc_outputs_coord,
            "enc_memory": enc_memory,
            "backbone_memory": backbone_memory,
        }
        if self.watch_center_in_box:
            output["enc_outputs"]["init_boxes"] = init_reference

        # encoder denoising
        if dn_meta and dn_meta["single_padding"] > 0 and dn_anchor_boxes is not None:
            interm_class_dn = self.transformer.decoder.class_embed[self.transformer.decoder.num_layers](dn_content_query)
            interm_coord_dn = dn_anchor_boxes
            dn_meta["output_known_lbs_bboxes"]["enc_outputs"] = {"pred_logits": interm_class_dn, "pred_boxes": interm_coord_dn}

        # decoder input loss
        if self.decoder_input_loss:
            # split dn and non-dn
            if dn_meta is not None and dn_meta["single_padding"] > 0:
                padding_size = dn_meta["single_padding"] * dn_meta["dn_num"]
                decoder_content_input_dn = init_state[:, :padding_size]
                decoder_content_input = init_state[:, padding_size:]
            else:
                decoder_content_input_dn = None
                decoder_content_input = init_state

            pred_class_dec_in = self.transformer.decoder.class_embed[self.transformer.decoder.num_layers + 1](decoder_content_input)

            output["aux_outputs"].insert(0, {
                "pred_logits": pred_class_dec_in,
                "pred_boxes": interm_coord,
            })

            if dn_meta is not None and dn_meta["single_padding"] > 0 and dn_anchor_boxes is not None:
                pred_class_dec_in_dn = self.transformer.decoder.class_embed[self.transformer.decoder.num_layers + 1](decoder_content_input_dn)
                dn_meta["output_known_lbs_bboxes"]["aux_outputs"].insert(0, {
                    "pred_logits": pred_class_dec_in_dn,
                    "pred_boxes": interm_coord_dn,
                })
            else:
                if dn_meta is not None and  "output_known_lbs_bboxes" in dn_meta and "aux_outputs" in dn_meta["output_known_lbs_bboxes"]:
                    dn_meta["output_known_lbs_bboxes"]["aux_outputs"].insert(0, None)

        if os.getenv("UNSTABLE_MATCHING", "0") == "1":
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            matching_results = self.criterion(output, targets, return_matching_results=True)
            # import ipdb; ipdb.set_trace()

            res = {}
            for layerid, (mat1, mat2) in enumerate(zip(matching_results[:-1], matching_results[1:])):
                mat1, mat2 = mat1[0], mat2[0]

                # sort mat1
                matarg1 = mat1[1].sort()[1]
                mat1 = [mat1[0][matarg1], mat1[1][matarg1]]

                # sort mat2
                matarg2 = mat2[1].sort()[1]
                mat2 = [mat2[0][matarg2], mat2[1][matarg2]]

                # count!
                num_gt = mat1[0].shape[0]
                num_match = (mat1[0] == mat2[0]).sum().item()
                res[f"{layerid}_gt"] = num_gt
                res[f"{layerid}_match"] = num_match
            return res
                


        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.


        for example, a batch input with 2 images, one have 1 gt [0], the other have 2 gt [0, 1]. gdn_k is 2. num_queries is 100.
        the output will be:
            input_query_label/box of img1: [0, -, 0, -, ...]. total: 2 * num_queries
            input_query_label/box of img2: [0, 1, 0, 1, ...]. total: 2 * num_queries
            output dn_num: 50. (100 (num_queries) / 2 (max(num_gt)))
                note that the output dn_number is not the same as the input dn_number, which is 100.
            output single_padding: 4. (2 (max(num_gt)) * 2 (gdn_k))

        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :pargit: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        gdn_k = self.gdn_k
        neg_step_type: str = self.neg_step_type
        if neg_step_type.startswith("add"):
            neg_step: float = float(neg_step_type.split(":")[-1])

        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * gdn_k
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * gdn_k))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(gdn_k * dn_number, 1).view(-1)
        known_labels = labels.repeat(gdn_k * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(gdn_k * dn_number, 1).view(-1) # batch id
        known_bboxs = boxes.repeat(gdn_k * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        # add noise to boxes
        pad_size = int(single_padding * gdn_k * dn_number)
        total_gt = sum(known_num).item()
        item_idx_of_k = torch.arange(gdn_k).repeat(dn_number).repeat_interleave(total_gt).cuda()
        # for example: single_padding is 4, gdn_k is 2, dn_number is 33
        # bs = 2, total_gt = 5
        # item_idx_of_k: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] x25
        if neg_step_type.startswith("addonce"):
            item_idx_of_k = (item_idx_of_k > 0).float()
            # item_idx_of_k: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] x25
        """
        # # code in CDN
        # positive_idx = (
        #     torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        # )
        # positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        # positive_idx = positive_idx.flatten()
        # negative_idx = positive_idx + len(boxes)
        """
        if self.dn_select_min_noise_as_pos:
            is_pos = torch.zeros_like(item_idx_of_k).bool()
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # convert to [x1, y1, x2, y2]
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            # get diff of bbox (w/2 h/2)
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            # generate random noise
            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            ) # [num_dn * gdn_k, 4]. -1 or 1
            # import ipdb; ipdb.set_trace()
            rand_part = torch.rand_like(known_bboxs)

            # process noise scale
            # # old code in CDN:
            # # rand_part[negative_idx] += 1.0
            if neg_step_type.startswith("add"):
                if 'xy' in neg_step_type:
                    add_noise = torch.zeros_like(rand_part)
                    add_noise[:, :2] = item_idx_of_k[:, None].repeat(1, 2)
                    rand_part = rand_part + add_noise
                else:
                    rand_part = rand_part + neg_step * item_idx_of_k[:, None]
            elif neg_step_type == 'multipy':
                rand_part = rand_part * (1 + item_idx_of_k[:, None])

            # apply sign
            rand_part *= rand_sign

            # add noise
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            # convert back to [x, y, w, h]
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            # select min noise as pos
            if self.dn_select_min_noise_as_pos:
                if self.dn_select_1st_noise_as_pos:
                    is_pos = (item_idx_of_k == 0) # for equal to CDN
                else:
                    gious = generalized_box_iou_pairwise(box_cxcywh_to_xyxy(known_bbox_expand), box_cxcywh_to_xyxy(known_bboxs)) # [dn_number * gdn_k * total_gt]
                    k_indice = gious.view(dn_number, gdn_k, total_gt).max(1)[1].flatten() # [dn_number * total_gt]
                    tgtid_indice = torch.arange(total_gt).repeat(dn_number).cuda()
                    group_indices = torch.arange(dn_number).repeat_interleave(total_gt).cuda() * gdn_k * total_gt
                    final_indice = k_indice * total_gt + tgtid_indice + group_indices
                    is_pos[final_indice] = True

        # transform noised items to desired inputs
        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        # init input_query_label and input_query_bbox
        input_query_label = torch.zeros(pad_size, hidden_dim).cuda().repeat(batch_size, 1, 1)
        input_query_bbox = torch.zeros(pad_size, 4).cuda().repeat(batch_size, 1, 1)
        if self.dn_select_min_noise_as_pos:
            dn_is_pos = torch.zeros(pad_size).bool().cuda().repeat(batch_size, 1)

        # build map from indices of catted gts to indices of input_query_label (each item in bs.)
        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )           # [0, 1, 0]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(gdn_k * dn_number)]
            ).long()    # [0, 1, 0, 2, 3, 2, ..., 198, 199, 198] 
            # import ipdb; ipdb.set_trace()

        # map noised inputs to input_query_label and input_query_bbox
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            # known_bid:            [0, 0, 1, 0, 0, 1, 0, 0, 1, ...]
            # map_known_indice:     [0, 1, 0, 2, 3, 2, 4, 5, 4, ...]
            if self.dn_select_min_noise_as_pos:
                dn_is_pos[(known_bid.long(), map_known_indice)] = is_pos
            
        # build attention masks
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        if self.dn_to_matching_block:
            # dn query cannot see the maching query
            attn_mask[:pad_size, pad_size:] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * gdn_k * i : single_padding * gdn_k * (i + 1),
                    single_padding * gdn_k * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * gdn_k * i : single_padding * gdn_k * (i + 1), : single_padding * i * gdn_k
                ] = True
            else:
                attn_mask[
                    single_padding * gdn_k * i : single_padding * gdn_k * (i + 1),
                    single_padding * gdn_k * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * gdn_k * i : single_padding * gdn_k * (i + 1), : single_padding * gdn_k * i
                ] = True

        # build gradient detach mask
        if self.dn_detach_dn2matching:
            # detach dn to matching
            dn2matching_detach_mask = torch.zeros(tgt_size, tgt_size).to("cuda").bool()
            dn2matching_detach_mask[:pad_size, pad_size:] = True

        # build dn meta
        dn_meta = {
            "single_padding": single_padding * gdn_k, 
                # size of each group. for example, 
                # [(0, 1, 2, 0, 1, 2), (0, 1, 2, 0, 1, 2)]
                # num_gt = 3. 
                # single_padding = group size = 6. 
                # gdn_k = 2. (each gt repeat 2 times in each group).
            "dn_num": dn_number,
                # dn_number is total number of groups.
                # in this example, dn_number = 2. (2 groups)
            "gdn_k": gdn_k,
        }
        if self.dn_detach_dn2matching:
            dn_meta["dn2matching_detach_mask"] = dn2matching_detach_mask

        if self.dn_select_min_noise_as_pos:
            dn_meta['dn_is_pos'] = dn_is_pos

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        # # nms for the results
        if self.nms_thresh > 0:
            results = self.nms(results)
            
        return results

    def nms(self, results):
        new_results = []
        for i, result in enumerate(results):
            keep = batched_nms(result.pred_boxes.tensor, result.scores, result.pred_classes, self.nms_thresh)
            new_result = Instances(result.image_size)
            new_result.pred_boxes = Boxes(result.pred_boxes.tensor[keep])
            new_result.scores = result.scores[keep]
            new_result.pred_classes = result.pred_classes[keep]
            new_results.append(new_result)
        return new_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets
