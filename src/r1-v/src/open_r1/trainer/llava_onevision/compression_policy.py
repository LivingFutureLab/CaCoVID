import os
import cv2
import copy
import math
import time
import torch
import numpy as np
import deepspeed
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .attention_utils import Qwen2DecoderLayer, Qwen2RotaryEmbedding, Qwen2RMSNorm
from transformers.modeling_utils import PreTrainedModel
from .sample_utils import *
from .compression_policy_base import *
from glob import glob

class CompressionPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.text_config.hidden_size
        self.max_sample_rate = config.vision_config.max_sample_rate
        self.imp_ratio = getattr(config.vision_config, "imp_ratio", None)
        self.frame_ratio = getattr(config.vision_config, "frame_ratio", None)
        self.sample_mode_num = getattr(config.vision_config, "sample_mode_num", {})
        self.group_size = getattr(config.vision_config, "group_size", 2)
        self.max_group_prob = getattr(config.vision_config, "max_group_prob", 2)
        self.max_sample_ratio_for_each_frame = getattr(config.vision_config, "max_sample_ratio_for_each_frame", 0.5)

        config._attn_implementation = config._attn_implementation
        self.rotary_emb = Qwen2RotaryEmbedding(config=config.text_config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config.text_config, layer_idx) for layer_idx in range(1)]
        )
        self.layernorm = Qwen2RMSNorm(embed_dim, eps=config.text_config.rms_norm_eps)
        self.token_compress = nn.Sequential(nn.Linear(embed_dim, embed_dim//2, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//2, embed_dim//4, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//4, embed_dim//8, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//8, embed_dim//16, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//16, 2, bias=True))
        self.frame_compress = nn.Sequential(nn.Linear(embed_dim, embed_dim//2, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//2, embed_dim//4, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//4, embed_dim//8, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//8, embed_dim//16, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//16, 2, bias=True))
        self.mode_selection = nn.Sequential(nn.Linear(embed_dim, embed_dim//2, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//2, embed_dim//4, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//4, embed_dim//8, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//8, embed_dim//16, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//16, 3, bias=True))

    def forward(self, video_embeds, tau=1, return_logits=False, topk=0.1, token_select_strategy=False, 
                      old_token_policy_logits=None, old_frame_policy_logits=None, **kwargs):
        inputs_embeds = kwargs['inputs_embeds']
        attention_mask = kwargs['attention_mask']
        position_ids = kwargs['position_ids']
        video_mask = kwargs['video_mask']
        num_generation = kwargs['num_generation']
        video_grid_thw = kwargs['video_grid_thw']
        max_sample_rate = kwargs.get('max_sample_rate', None)
        t, h, w = video_grid_thw
        device = inputs_embeds.device
        self.doc_id = kwargs.get("doc_id", None)

        # 建立视频与指令的全局交互
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                is_causal=False,
            )
            hidden_states = layer_outputs[0]

        video_embeds = hidden_states[video_mask, :]
        video_embeds_ori = inputs_embeds[video_mask, :]
        text_embeds = hidden_states[~video_mask, :].mean(dim=0, keepdim=True)

        # 模式选择
        mode_selection_logits = self.mode_selection(text_embeds)*100

        # 视频特征评估
        token_compress_logits = self.token_compress(self.layernorm(video_embeds))
        frame_embeds = video_embeds.reshape(t, h, w, video_embeds.shape[-1]).mean(dim=(1, 2))

        frame_compress_logits = self.frame_compress(self.layernorm(frame_embeds))
        
        if return_logits: # 如果要输出embedding这里也要输出
            return token_compress_logits, {"frame_compress_logits": frame_compress_logits,
                                           "mode_selection_logits": mode_selection_logits}

        if self.training:
            token_compress_sample, extra_info = self.dynamic_sample_for_training(frame_compress_logits if old_frame_policy_logits is None else old_frame_policy_logits,
                                                                                token_compress_logits if old_token_policy_logits is None else old_token_policy_logits, 
                                                                                max_sample_rate,
                                                                                video_grid_thw,
                                                                                num_generation,
                                                                                video_embeds_ori)
        else:
            dynamic_sample_for_evaluation = getattr(self, f"dynamic_sample_for_evaluation_{token_select_strategy}")
            token_compress_sample, extra_info = dynamic_sample_for_evaluation(frame_compress_logits,
                                                                                    token_compress_logits, 
                                                                                    mode_selection_logits,
                                                                                    topk,
                                                                                    video_grid_thw,
                                                                                    video_embeds_ori)
        return token_compress_sample, extra_info

    def dynamic_sample_for_evaluation_token(self, frame_compress_logits, token_compress_logits, mode_selection_logits,
                                    topk, video_grid_thw, video_embeds_ori):
        token_compress_sample, extra_info = self.sample_token_eval(frame_compress_logits, token_compress_logits, video_grid_thw, topk, video_embeds_ori)
        return token_compress_sample, extra_info

    def sample_token_eval(self, frame_compress_logits, token_compress_logits, video_grid_thw, topk, video_embeds_ori):
        t, h, w = video_grid_thw
        device = token_compress_logits.device
        imp_ratio = self.imp_ratio
        max_sample_ratio_for_each_frame = self.max_sample_ratio_for_each_frame
        
        frame_compress_logits_ = frame_compress_logits.float().diff(dim=-1).reshape(t)
        frame_compress_logits_ = frame_compress_logits_ / frame_compress_logits_.abs().max() * 3
        frame_compress_softmax = frame_compress_logits_.softmax(dim=-1).clip(min=1/(2*t), max=1/(t*topk)*max_sample_ratio_for_each_frame)
        frame_compress_softmax = frame_compress_softmax / frame_compress_softmax.sum()
        
        token_compress_logits = token_compress_logits.float().diff(dim=-1).reshape(1, -1)*0.001
        token_compress_softmax = token_compress_logits.reshape(t, h*w).softmax(dim=-1)
        topk_indices_list = []
        topk_for_each_frame = (frame_compress_softmax*round(imp_ratio*topk*h*w*t)).round().clip(min=1, max=int(imp_ratio*h*w*max_sample_ratio_for_each_frame)).long()
        res_num = math.floor(imp_ratio*topk*h*w*t) - topk_for_each_frame.sum()
        if res_num < 0:
            while res_num < 0:
                topk_for_each_frame[topk_for_each_frame.argmax()] = topk_for_each_frame[topk_for_each_frame.argmax()] - 1
                res_num += 1
        else:
            while res_num > 0:
                topk_for_each_frame[topk_for_each_frame.argmin()] = topk_for_each_frame[topk_for_each_frame.argmin()] + 1
                res_num -= 1
        topk_mask = (torch.arange(h*w, device=device).unsqueeze(0).repeat(t, 1) < topk_for_each_frame.unsqueeze(-1)).reshape(-1)
        sorted_index = token_compress_softmax.sort(dim=-1, descending=True).indices + torch.arange(t, device=device).unsqueeze(-1)*h*w
        topk_indices = sorted_index.reshape(-1)[topk_mask]
        token_sample = torch.zeros(t*h*w, device=device).long()
        token_sample[topk_indices] = 1
        if imp_ratio < 1.0:
            token_sample = self.div_sample_eval(video_embeds_ori, token_sample, math.floor(topk*t*h*w)-token_sample.sum(), inv=True, video_grid_thw=video_grid_thw)

        if os.environ.get("PRINT_LOG", "0") == "1":
            print(f"Sample {topk}/{h*w} token in each frame for token sample, token sample rate %.4f"%(token_sample.float().mean()))
        return token_sample.unsqueeze(0), {"token_compress_softmax": token_compress_softmax,
                                           "token_sample": token_sample}

    def dynamic_sample_for_training(self, frame_compress_logits, token_compress_logits, 
                                    max_sample_rate, video_grid_thw, num_generation, video_embeds_ori):
        token_sample_all = []
        extra_info = {}
        for mode in ['content', 'token', 'frame']:
            token_sample, extra_info_item = getattr(self, f"sample_{mode}")(frame_compress_logits, token_compress_logits, 
                                                           max_sample_rate, video_grid_thw, num_generation,
                                                           video_embeds_ori)
            token_sample_all.append(token_sample)
            extra_info.update(extra_info_item)
        token_sample_all = torch.cat(token_sample_all)
        return token_sample_all, extra_info

    def sample_content(self, frame_compress_logits, token_compress_logits, 
                        max_sample_rate, video_grid_thw, num_generation, video_embeds_ori):
        device = frame_compress_logits.device
        t, h, w = video_grid_thw
        content_sample = torch.zeros(t*h*w, device=device)
        video_embeds_ori = video_embeds_ori.reshape(t, h*w, -1)
        for i in range(t):
            topk = round(max_sample_rate*h*w)
            content_sample[i*h*w:(i+1)*h*w] = self.div_sample(video_embeds_ori[i], 
                                                            content_sample[i*h*w:(i+1)*h*w], 
                                                            topk, inv=True, video_grid_thw=video_grid_thw)
        if os.environ.get("PRINT_LOG", "0") == "1":
            print(f"Sample {topk}/{h*w} tokens in each frame for content sample, total 1 sample")
        return content_sample.unsqueeze(0), {"content_sample": content_sample}

    def sample_token(self, frame_compress_logits, token_compress_logits, 
                        max_sample_rate, video_grid_thw, num_generation, video_embeds_ori):
        t, h, w = video_grid_thw
        device = frame_compress_logits.device
        token_compress_logits = token_compress_logits.float().diff(dim=-1).reshape(1, -1)
        token_compress_softmax = token_compress_logits.reshape(t, h*w).softmax(dim=-1)
        num_group = max(int(1 / max_sample_rate / self.group_size), 1) # 每一组里有多倍的token可以选
        max_prob_for_each_group = max(min(self.max_group_prob / num_group, 0.9), 0.1) if num_group > 1 else 1.0 # 采样概率最大不超过两倍的均值
        token_sample = []
        k = round(max_sample_rate*h*w)
        for token_compress_softmax_item in token_compress_softmax:
            token_sample_item = sample_without_replacement_h(token_compress_softmax_item.unsqueeze(0), 
                                                            k=k, num_generation=self.sample_mode_num['token'],
                                                            num_group=num_group, max_prob_for_each_group=max_prob_for_each_group)
            token_sample.append(token_sample_item)
        token_sample = torch.stack(token_sample, dim=1).reshape(self.sample_mode_num['token'], t*h*w)
        # 将第一个赋值为TOPK的index，不进行采样与content采样对比学习
        top_index = token_compress_softmax.topk(k=k, dim=-1).indices + torch.arange(t, device=device).unsqueeze(-1)*h*w
        token_sample[0] = token_sample[0]*0
        token_sample[0, top_index] = 1
        if os.environ.get("PRINT_LOG", "0") == "1":
            print(f"Sample {k}/{h*w} token in each frame for token sample, total {self.sample_mode_num['token']} sample, \
token maximum score {token_compress_softmax.max().item()}, minimum score {token_compress_softmax.min().item()}")
        return token_sample, {"token_compress_softmax": token_compress_softmax,
                              "token_sample": token_sample}

    def sample_frame(self, frame_compress_logits, token_compress_logits, 
                        max_sample_rate, video_grid_thw, num_generation, video_embeds_ori):
        t, h, w = video_grid_thw
        device = frame_compress_logits.device
        frame_compress_logits = frame_compress_logits.float().diff(dim=-1).reshape(1, -1)
        frame_compress_softmax_ = frame_compress_logits.reshape(1, -1).softmax(dim=-1)
        frame_ratio = self.frame_ratio
        num_frame_sample = max(math.ceil(frame_compress_softmax_.shape[-1]*frame_ratio), 1)
        frame_compress_softmax = frame_compress_softmax_.repeat(self.sample_mode_num['frame'], 1)
        frame_compress_sample_index = torch.multinomial(clip_max_probability(frame_compress_softmax, 3/t), 
                                                          num_samples=num_frame_sample, replacement=False) # 有放回采样
        frame_compress_sample_index[0] = frame_compress_softmax[0].topk(k=num_frame_sample).indices
        frame_sample = torch.zeros([self.sample_mode_num['frame'], t], device=device)
        frame_sample_ = frame_sample.scatter(dim=1, index=frame_compress_sample_index, src=torch.ones_like(frame_compress_sample_index).to(frame_sample.dtype))
        frame_sample = frame_sample_.repeat_interleave(h*w, dim=-1).long()
        content_sample_all = []
        for frame_sample_item in frame_sample:
            topk = round(max_sample_rate*h*w)*t
            content_sample = self.div_sample(video_embeds_ori, 
                                            frame_sample_item, 
                                            topk, inv=False, 
                                            video_grid_thw=video_grid_thw,
                                            with_coord_t=True)
            content_sample_all.append(content_sample)
        frame_sample = torch.stack(content_sample_all, dim=0)
        if os.environ.get("PRINT_LOG", "0") == "1":
            print(f"Sample {num_frame_sample}/{t} frames in each video for frame sample, total {self.sample_mode_num['frame']} sample, \
frame maximum score {frame_compress_softmax.max().item()}, minimum score {frame_compress_softmax.min().item()}")
        return frame_sample, {"frame_sample": frame_sample,
                              "frame_compress_softmax": frame_compress_softmax_,
                              "frame_compress_sample": frame_sample_,}

    def div_sample(self, hidden_states, token_compress_sample, topk, inv=False, **kwargs):
        device_type = hidden_states.device
        if inv:
            selected_hidden_states = hidden_states[~token_compress_sample.bool(), :]
        else:
            selected_hidden_states = hidden_states[token_compress_sample.bool(), :]
        dist_matrix = torch.cdist(selected_hidden_states.float(), selected_hidden_states.float()) / (selected_hidden_states.shape[-1] ** 0.5)

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=4, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(
            density.shape, device=device_type, dtype=density.dtype) * 1e-6

        density_mask = density[None, :] > density[:, None]
        density_mask = density_mask.type(selected_hidden_states.dtype)
        dist_max = dist_matrix.max()
        dist_0, index_parent = (dist_matrix * density_mask + dist_max * (1 - density_mask)).min(dim=-1)
        density_score = dist_0 * density

        sampled_indexs = torch.topk(density_score, k=topk, dim=-1).indices
        selected_mask = torch.zeros(selected_hidden_states.shape[0], device=device_type, dtype=token_compress_sample.dtype)
        selected_mask = selected_mask.scatter(0, sampled_indexs, 1)
        if inv:
            token_compress_sample = token_compress_sample.masked_scatter(~token_compress_sample.bool(), selected_mask)
        else:
            token_compress_sample = token_compress_sample.masked_scatter(token_compress_sample.bool(), selected_mask)
        return token_compress_sample

    def div_sample_eval(self, hidden_states, token_compress_sample, topk, inv=False, **kwargs):
        device_type = hidden_states.device
        if inv:
            selected_hidden_states = hidden_states[~token_compress_sample.bool(), :]
        else:
            selected_hidden_states = hidden_states[token_compress_sample.bool(), :]
        sampled_indexs = torch.linspace(0, selected_hidden_states.shape[0]-1, topk).round().long().to(device_type)
        selected_mask = torch.zeros(selected_hidden_states.shape[0], device=device_type, dtype=token_compress_sample.dtype)
        selected_mask = selected_mask.scatter(0, sampled_indexs, 1)
        if inv:
            token_compress_sample = token_compress_sample.masked_scatter(~token_compress_sample.bool(), selected_mask)
        else:
            token_compress_sample = token_compress_sample.masked_scatter(token_compress_sample.bool(), selected_mask)
        return token_compress_sample

