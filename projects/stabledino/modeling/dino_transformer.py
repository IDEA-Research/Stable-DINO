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

import os
import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid


class StableDINOTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
        multi_level_fusion: str = "none",
    ):
        super(StableDINOTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = embed_dim
        self.pre_norm = not post_norm

        if num_layers > 0 and post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        # dense-fusion
        self.multi_level_fusion = multi_level_fusion
        if self.multi_level_fusion == "none":
            pass       
        elif self.multi_level_fusion == "dense-fusion":
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * (num_layers + 1), embed_dim),
                nn.LayerNorm(embed_dim),
            )
            nn.init.constant_(self.fusion_layer[0].bias, 0)
        else:
            raise NotImplementedError("Not implemented fusion method: {}".format(self.multi_level_fusion))

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        query_list = [query]
        for idx, layer in enumerate(self.layers):
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            if self.multi_level_fusion in ['dense-fusion']:
                query_list.append(query)
            else:
                assert self.multi_level_fusion == 'none'

        if self.multi_level_fusion == 'dense-fusion':
            query = self.fusion_layer(torch.cat(query_list, dim=-1))

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DINOTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        look_forward_twice=True,
        extra_self_attn=False,
    ):
        self.extra_self_attn = extra_self_attn
        if self.extra_self_attn:
            operation_order = ("self_attn", "norm", "cross_attn", "norm", "self_attn", "norm", "ffn", "norm")
            super(DINOTransformerDecoder, self).__init__(
                transformer_layers=BaseTransformerLayer(
                    attn=[
                        MultiheadAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            attn_drop=attn_dropout,
                            batch_first=True,
                        ),
                        MultiScaleDeformableAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=attn_dropout,
                            batch_first=True,
                            num_levels=num_feature_levels,
                        ),
                        MultiheadAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            attn_drop=attn_dropout,
                            batch_first=True,
                        ),
                    ],
                    ffn=FFN(
                        embed_dim=embed_dim,
                        feedforward_dim=feedforward_dim,
                        output_dim=embed_dim,
                        ffn_drop=ffn_dropout,
                    ),
                    norm=nn.LayerNorm(embed_dim),
                    operation_order=operation_order,
                ),
                num_layers=num_layers,
            )
        else:
            operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
            super(DINOTransformerDecoder, self).__init__(
                transformer_layers=BaseTransformerLayer(
                    attn=[
                        MultiheadAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            attn_drop=attn_dropout,
                            batch_first=True,
                        ),
                        MultiScaleDeformableAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=attn_dropout,
                            batch_first=True,
                            num_levels=num_feature_levels,
                        ),
                    ],
                    ffn=FFN(
                        embed_dim=embed_dim,
                        feedforward_dim=feedforward_dim,
                        output_dim=embed_dim,
                        ffn_drop=ffn_dropout,
                    ),
                    norm=nn.LayerNorm(embed_dim),
                    operation_order=operation_order,
                ),
                num_layers=num_layers,
            )
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4
        
        if self.extra_self_attn:
            attn_masks.append(attn_masks[0])

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DINOTransformer(nn.Module):
    """Transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        two_stage_num_proposals=900,
        learnt_init_query=True,
        content_query_init_type="none",
        extra_learnt_init_query=False,
        encoder_denoising=False,
        encoder_roi_pooling_layer=None,
        memory_fusion_type="none",
        memory_kd_loss="none",
    ):
        super(DINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.encoder_denoising = encoder_denoising

        self.embed_dim = self.encoder.embed_dim

        if decoder.num_layers > 0:
            self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        else:
            self.level_embeds = None
        
        # encoder denosing
        self.encoder_roi_pooling_layer = encoder_roi_pooling_layer
        if self.encoder_denoising:
            assert encoder_roi_pooling_layer is not None, "encoder_roi_pooling_layer should not be None"
        

        # init decoder queries
        self.learnt_init_query = learnt_init_query
        self.extra_learnt_init_query = extra_learnt_init_query
        self.content_query_init_type = content_query_init_type
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        if self.extra_learnt_init_query:
            assert not self.learnt_init_query
            assert self.content_query_init_type != 'none'
            self.extra_tgt_embed = nn.Embedding(self.two_stage_num_proposals + 1, self.embed_dim)
        if self.learnt_init_query:
            assert self.content_query_init_type == "none", "content_query_init_type should be none"
        else:
            assert self.content_query_init_type in ["mlp:feat", "mlp:box", "mlp:both"]
            if self.content_query_init_type in ["mlp:feat"]:
                self.init_content_query_proj = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 3)
            elif self.content_query_init_type in ["mlp:box"]:
                self.init_content_query_proj = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 3)
            elif self.content_query_init_type in ["mlp:both"]:
                self.init_content_query_proj = MLP(3 * self.embed_dim, self.embed_dim, self.embed_dim, 3)
            self.content_query_norm = nn.LayerNorm(self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        # memory fusion
        self.memory_fusion = memory_fusion_type
        if self.memory_fusion == "linear":
            self.fusion_run = nn.Linear(self.embed_dim * 2, self.embed_dim)
            nn.init.zeros_(self.fusion_run.bias)
        elif self.memory_fusion == "linearnorm":
            self.fusion_run = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
            nn.init.zeros_(self.fusion_run[0].bias)
        else:
            assert self.memory_fusion == "none", "memory_fusion_type {} not implemented".format(self.memory_fusion)

        self.memory_kd_loss = memory_kd_loss
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.level_embeds is not None:
            nn.init.normal_(self.level_embeds)

    def unflatten_features(self, memory, spatial_shapes):
        N, S, C = memory.shape
        unflattened = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            unflattened.append(memory[:, _cur : (_cur + H * W)].view(N, H, W, C))
            _cur += H * W
        return unflattened

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_projection(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        attn_masks,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1) if self.level_embeds is not None else pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        return feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios

    def query_selection(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        query_embed,
        level_start_index,
        valid_ratios,
    ):
        # transform outputs
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        bs = output_memory.shape[0]
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        # obtain dn queries
        dn_content_query: torch.Tensor = query_embed[0]
        dn_anchor_boxes: torch.Tensor = query_embed[1] # unsigmoided.

        # build for encoder dn
        if self.encoder_denoising and dn_content_query is not None:
            # flatten memory to multi level features
            reference_points = dn_anchor_boxes[:, :, None].sigmoid() * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            dn_content_query = self.encoder_roi_pooling_layer(
                query=dn_content_query,
                value=memory,
                key_padding_mask=mask_flatten,
                reference_points=reference_points, # sigmoided.
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                identity=0,
            )
            dn_content_query = self.enc_output_norm(self.enc_output(dn_content_query))
            dn_anchor_boxes = dn_anchor_boxes + self.decoder.bbox_embed[self.decoder.num_layers](dn_content_query)

        # project to class and bbox
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if dn_anchor_boxes is not None:
            # cat with dn reference points
            reference_points = torch.cat([dn_anchor_boxes.sigmoid(), reference_points], 1)

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        if self.learnt_init_query:
            # the default pipeline in DINO
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
            if dn_content_query is not None:
                # cat with dn features
                target = torch.cat([dn_content_query, target], 1)
        else:
            if self.content_query_init_type == "none":
                target = target_unact.detach()
                if dn_content_query is not None:
                    # cat with dn features
                    target = torch.cat([dn_content_query, target], 1)
            else:
                _target_unact = self.content_query_norm(target_unact.detach())
                if dn_content_query is not None:
                    # cat with dn features
                    _target_unact = torch.cat([dn_content_query.detach(), _target_unact], 1)

                if self.content_query_init_type == "mlp:feat":
                    target = self.init_content_query_proj(_target_unact)
                elif self.content_query_init_type == "mlp:box":
                    target = self.init_content_query_proj( 
                        get_sine_pos_embed(reference_points)
                     )
                elif self.content_query_init_type == "mlp:both":
                    input_tensor = torch.cat((
                        get_sine_pos_embed(reference_points),
                        _target_unact
                    ), dim=-1)
                    target = self.init_content_query_proj(input_tensor)

                
                if self.extra_learnt_init_query:
                    target[:, -self.two_stage_num_proposals:] += self.extra_tgt_embed.weight[:self.two_stage_num_proposals]
                    target[:, :-self.two_stage_num_proposals] += self.extra_tgt_embed.weight[self.two_stage_num_proposals]



        if not self.encoder_denoising:
            dn_content_query = None
            dn_anchor_boxes = None

        return target, reference_points, target_unact, topk_coords_unact, dn_content_query, dn_anchor_boxes, output_proposals, enc_outputs_class, enc_outputs_coord_unact, output_memory
            

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        attn_masks,
        **kwargs,
    ):  
        """
        Args:
            - multi_level_feats: list of tensor, each tensor has shape (bs, c, h, w)
            - multi_level_masks: list of tensor, each tensor has shape (bs, h, w)
            - multi_level_pos_embeds: list of tensor, each tensor has shape (bs, c, h, w)
            - query_embed: a tuple. size 2. denosing queries.
                - query_embed[0]: tensor, has shape (bs, num_query, embed_dims)
                - query_embed[1]: tensor, has shape (bs, num_query, 4). unsigmoided. xywh.
        """

        # flatten feats, masks and pos_embeds
        feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios = self.forward_projection(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
        )

        # feed into transformer encoder
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat_flatten.device
        )
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        if self.memory_kd_loss != "none":
            enc_memory = memory.clone()
            backbone_memory = feat_flatten.clone()
        else:
            enc_memory = None
            backbone_memory = None

        if self.memory_fusion != "none":
            # import ipdb; ipdb.set_trace()
            memory = self.fusion_run(torch.cat([memory, feat_flatten], dim=-1))


        # query selction
        target, reference_points, target_unact, topk_coords_unact, dn_content_query, dn_anchor_boxes, enc_output_proposals, enc_outputs_class, enc_outputs_coord_unact, enc_output_memory \
            = self.query_selection(
            memory, mask_flatten, spatial_shapes, query_embed, level_start_index, valid_ratios)
        # target: bs, num_dn+num_queries, embed_dims
        # reference_points: bs, num_dn+num_queries, 4. xywh. sigmoided.
        # target_unact: bs, num_queries, embed_dims
        # topk_coords_unact: bs, num_queries, 4. xywh. unsigmoided.


        # decoder
        inter_states, inter_references = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )

        init_state = self.decoder.norm(target)
        init_reference_out = reference_points
        inter_references_out = inter_references
        if dn_anchor_boxes is not None:
            dn_anchor_boxes = dn_anchor_boxes.sigmoid()
        return (
            init_state,
            inter_states,
            init_reference_out, # [0..1]
            inter_references_out, # [0..1]
            target_unact,
            topk_coords_unact.sigmoid(), # [0..1]
            dn_content_query, 
            dn_anchor_boxes, # [0..1]
            enc_output_proposals.sigmoid(), # [0..1]
            enc_outputs_class,
            enc_outputs_coord_unact.sigmoid(), # [0..1]
            enc_memory,
            backbone_memory,
        )
