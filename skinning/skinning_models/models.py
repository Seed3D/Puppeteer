#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
from torch import nn, einsum
import torch.nn.functional as F

from skinning_models.networks import PreNorm, FeedForward, Attention, PointEmbed
from third_partys.Michelangelo.encode import load_model
from third_partys.PartField.encode import partfield

class SkinningBlock(nn.Module):
    def __init__(self, dim, args, heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        # -- Self-Attn (joint)
        self.self_attn_j   = PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim = dim)
        self.ff_j          = PreNorm(dim, FeedForward(dim, mult=ff_mult))

        # -- Cross-Attn (point -> shape)
        self.cross_attn_ps = PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim = dim)
        self.ff_ps         = PreNorm(dim, FeedForward(dim, mult=ff_mult))
        
        # -- Cross-Attn (joint -> shape)
        self.cross_attn_js = PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim = dim)
        self.ff_js         = PreNorm(dim, FeedForward(dim, mult=ff_mult))
        
        # -- Cross-Attn (joint -> point)
        self.cross_attn_jp = PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim = dim)
        self.ff_jp         = PreNorm(dim, FeedForward(dim, mult=ff_mult))

        # -- Cross-Attn (point -> joint)
        self.cross_attn_pj = PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim = dim)
        self.ff_pj         = PreNorm(dim, FeedForward(dim, mult=ff_mult))

        self.use_TAJA = args.use_TAJA
        if self.use_TAJA:
            self.rel_pos_embedding = nn.Embedding(10, dim // 4)  
            self.rel_pos_proj = nn.Linear(dim // 4, heads)
            self.rel_pos_scale = nn.Parameter(torch.ones(1) * 0.1)  # Initial value 0.1
 
    def forward(self, point, joint, shape, valid_mask=None, graph_dist=None):
        """
        point: (B, Np, dim)
        joint: (B, Nj, dim)
        shape: (B, Ns, dim)
        valid_mask: (B, Nj) or None
        return:
            updated_point, updated_joint
        """
        
        # 1) joint self-attention with TAJA
        if self.use_TAJA:
            batch_size, n_joints = joint.shape[0], joint.shape[1]
            
            dist_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2) 
            safe_dist = torch.where(dist_mask, graph_dist, torch.zeros_like(graph_dist))
            distances_clamped = torch.clamp(safe_dist, 0, 9).long()  # (B, Nj, Nj)
     
            rel_pos_embeddings = self.rel_pos_embedding(distances_clamped)  # (B, Nj, Nj, dim//4)
            rel_pos_encoding = self.rel_pos_proj(rel_pos_embeddings) * self.rel_pos_scale  # (B, Nj, Nj, heads)
            
            rel_pos_encoding = torch.where(
                dist_mask.unsqueeze(-1).expand_as(rel_pos_encoding),
                rel_pos_encoding,
                torch.zeros_like(rel_pos_encoding)
            ) 
        else:
            rel_pos_encoding = None
        
        joint_enhance = self.self_attn_j(joint, context=joint, context_mask=valid_mask, rel_pos=rel_pos_encoding) + joint
        joint_enhance = self.ff_j(joint_enhance) + joint_enhance

        # 2) point->shape
        point_context = self.cross_attn_ps(point, context=shape) + point
        point_context = self.ff_ps(point_context) + point_context
        
        # 2) joint->shape
        joint_context = self.cross_attn_js(joint_enhance, context=shape, query_mask=valid_mask) + joint_enhance
        joint_context = self.ff_js(joint_context) + joint_context
        
        # 3) joint->point
        joint_refine = self.cross_attn_jp(joint_context, context=point_context, query_mask=valid_mask) + joint_context
        joint_refine = self.ff_jp(joint_refine) + joint_refine
        
        # 4) point->joint
        point_final = self.cross_attn_pj(point_context, context=joint_refine, context_mask=valid_mask) + point_context
        point_final = self.ff_pj(point_final) + point_final
        
        return point_final, joint_refine

class SkinningNetStacked(nn.Module):
    def __init__(self, args, dim=768, heads=8, dim_head=64, ff_mult=4, scale_init=1.):
        super().__init__()
        self.args = args
        self.max_joints = args.max_joints
        
        self.skeleton_condition = PointEmbed(dim=dim)
        self.scale = nn.Parameter(torch.tensor(scale_init), requires_grad=True)

        self.point_encoder = load_model()
        self.point_encoder.eval()
        for param in self.point_encoder.parameters():
            param.requires_grad = False
    
        self.point_embed_pe = PointEmbed(dim=dim)
        self.point_embed = partfield()
        self.proj = nn.Sequential(
                nn.Linear(448, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
    
        # multiple blocks
        self.blocks = nn.ModuleList([
            SkinningBlock(dim, args, heads=heads, dim_head=dim_head, ff_mult=ff_mult)
            for _ in range(args.depth)
        ])

    def process_point_feature(self, point_feature):
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        point_feature_first_column = point_feature[:, 0:1]
        encode_feature = torch.cat([point_feature_first_column, shape_latents], dim=1)
        return encode_feature
    
    def forward(self,
                sample_points, skeleton, pc_w_norm, dist_graph, 
                valid_mask=None,  # (B, Nj)
                target=None):         # (B, Np, Nj)
        """
        cosine similarity + softmax + loss
        """
       
        point_out1 = self.point_embed(sample_points) 
        point_out1 = self.proj(point_out1)
        point_out2 = self.point_embed_pe(sample_points)
        point_out = point_out1 + point_out2 # (bs, 8192, 768)

        joint_out = self.skeleton_condition(skeleton) # (bs, args.max_joints, 768)

        point_feature = self.point_encoder.encode_latents(pc_w_norm)
        shape_feature = self.process_point_feature(point_feature=point_feature) # （bs, 257, 768）

        for block in self.blocks:
            point_out, joint_out = block(point_out, joint_out, shape_feature, valid_mask=valid_mask, graph_dist=dist_graph)

        point_norm = F.normalize(point_out, p=2, dim=-1)  # (B, Np, D)
        joint_norm = F.normalize(joint_out, p=2, dim=-1)  # (B, Nj, D)

        score_cos = einsum('b i d, b j d -> b i j', point_norm, joint_norm)
        score = (self.scale.abs()+1e-9) * score_cos
        
        skinning_weight = F.softmax(score, dim=-1) 
    
        if target is None:
            return skinning_weight
        