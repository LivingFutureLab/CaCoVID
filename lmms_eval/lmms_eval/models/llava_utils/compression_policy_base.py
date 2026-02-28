import os
import cv2
import math
import torch
import numpy as np
import deepspeed
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from easydict import EasyDict as edict
from .attention_utils import Qwen2DecoderLayer, Qwen2RotaryEmbedding, Qwen2RMSNorm
from transformers.modeling_utils import PreTrainedModel
from .sample_utils import *
from glob import glob

def grayscale_to_heatmap(img):
    norm_img = (img - img.min()) / (img.max() - img.min()) * 255
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    return heat_img

def add_heat_to_image(score_maps, base_path, temporal_patch=2, sample=None):
    score_maps = score_maps
    image_heap_2 = []
    image_heap_2_ori = []
    image_heap_2_norm = []
    sample_2_list = []
    for i in range(temporal_patch):
        if temporal_patch == 2:
            image_paths = sorted(glob(base_path+"/debug/*"))[i::2]
        else:
            image_paths = sorted(glob(base_path+"/debug/*"))
        if score_maps is not None:
            images = [plt.imread(image_path)[..., :3]*255 for image_path in image_paths[:score_maps.shape[0]]]
        else:
            images = [plt.imread(image_path)[..., :3]*255 for image_path in image_paths[:sample.shape[0]]]
        image_shape = images[0].shape[:2]
        target_shape = (image_shape / np.array(image_shape).max() * 128).astype(np.int32).tolist()[::-1]
        images = [cv2.resize(image, target_shape) for image in images]
        score_map_list = []
        image_list = []
        sample_list = []
        if sample is not None and score_maps is not None:
            for image, score_map, sample_map in zip(images, score_maps, sample):
                score_map = cv2.resize(score_map, target_shape)
                sample_map = cv2.resize(sample_map, target_shape, interpolation=cv2.INTER_NEAREST)
                sample_list.append(sample_map)
                score_map_list.append(score_map)
                image_list.append(image)
        elif score_maps is not None:
            for image, score_map in zip(images, score_maps):
                score_map = cv2.resize(score_map, target_shape)
                score_map_list.append(score_map)
                image_list.append(image)
        elif sample is not None:
            for image, sample_map in zip(images, sample):
                sample_map = cv2.resize(sample_map, target_shape, interpolation=cv2.INTER_NEAREST)
                sample_list.append(sample_map)
                image_list.append(image)
        image_cat = np.concatenate(image_list, axis=0)
        if len(score_map_list) > 0:
            score_map_cat = np.concatenate(score_map_list, axis=0)
            heatmap_cat = grayscale_to_heatmap(score_map_cat)
            image_heap_2.append(image_cat*0.5 + heatmap_cat*0.5)
        # score_map_heatmap_list = [grayscale_to_heatmap(score_map) for score_map in score_map_list]
        # score_map_heatmap_cat = np.concatenate(score_map_heatmap_list, axis=0)
        if len(sample_list) > 0:
            sample_map_cat = grayscale_to_heatmap(np.concatenate(sample_list, axis=0))
            sample_2_list.append(image_cat*0.2 + sample_map_cat*0.8)
        image_heap_2_ori.append(image_cat)
        # image_heap_2_norm.append(image_cat*0.5 + score_map_heatmap_cat*0.5)
    image_heat = np.concatenate(image_heap_2+sample_2_list+image_heap_2_ori, axis=1).astype(np.uint8)
    return image_heat

def sample_without_replacement(probabilities: torch.Tensor, max_sample_rate: float = 1.0, visible_feature_num=None, k=None, num_generation=8) -> torch.Tensor:
    # ratio = torch.rand(1)[0] * max_sample_rate
    if k is None:
        ratio = max_sample_rate
        if visible_feature_num is not None:
            k = torch.round(ratio * visible_feature_num).clamp(min=1, max=probabilities.shape[-1]).long().item()
        else:
            k = torch.round(torch.tensor(ratio * probabilities.shape[-1])).clamp(min=1, max=probabilities.shape[-1]).long().item()
    probabilities_norm = probabilities / probabilities.sum(dim=-1, keepdim=True)
    probabilities_norm = probabilities_norm.repeat(num_generation, 1)
    probabilities_norm = clip_max_probability(probabilities_norm, max_prob=0.1)
    # print(probabilities_norm.max(), probabilities_norm.min())
    indices = torch.multinomial(probabilities_norm, num_samples=k, replacement=False)
    mask = torch.zeros_like(probabilities_norm).long()
    src = torch.ones_like(indices)
    mask.scatter_(-1, indices, src)
    return mask

# 1/(num_group*max_prob_for_each_group)收敛后的探索空间
def sample_without_replacement_h(probabilities: torch.Tensor, max_sample_rate: float = 1.0, 
                                 visible_feature_num=None, num_group=8, max_prob_for_each_group=0.5,
                                 num_generation=8, k=None) -> torch.Tensor:
    if k is None:
        ratio = max_sample_rate
        assert max_sample_rate * num_group < 1.0, f"ensure group have enough sample for {max_sample_rate} rate"
        if visible_feature_num is not None:
            k = torch.round(ratio * visible_feature_num).clamp(min=1, max=probabilities.shape[-1]).long().item()
        else:
            k = torch.round(torch.tensor(ratio * probabilities.shape[-1])).clamp(min=1, max=probabilities.shape[-1]).long().item()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    group_sample_mask = grouped_sample(probabilities, num_group=num_group, 
                                        max_prob_for_each_group=max_prob_for_each_group, 
                                        num_generation=num_generation)
    num_categories = probabilities.shape[-1]
    ele_in_each_group = num_categories // num_group
    flat_probs = probabilities.view(-1, num_categories)
    flat_probs = flat_probs * group_sample_mask
    flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True) + 1e-8)
    # clamp避免group内有太大的采样偏见
    # print("max_selected_values_prev", (flat_probs*group_sample_mask).max(dim=-1).values)
    # flat_probs = flat_probs.clamp(max=3/ele_in_each_group, min=1/ele_in_each_group/3)
    flat_probs = flat_probs.clamp(max=2/ele_in_each_group, min=1/ele_in_each_group/2)
    flat_probs = flat_probs * group_sample_mask
    flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True) + 1e-8)
    # print("max_selected_values_clip", (flat_probs*group_sample_mask).max(dim=-1).values)
    indices = torch.multinomial(flat_probs, num_samples=k, replacement=False)
    indices = indices.view(num_generation, k)
    mask = torch.zeros_like(probabilities).long().repeat(num_generation, 1)
    src = torch.ones_like(indices)
    mask.scatter_(-1, indices, src)
    return mask

# 1/(num_group*max_prob_for_each_group)收敛后的探索空间
def sample_without_replacement_h_inter(probabilities: torch.Tensor, max_sample_rate: float = 1.0, 
                                 visible_feature_num=None, num_group=8, max_prob_for_each_group=0.5,
                                 num_generation=8, k=None) -> torch.Tensor:
    if k is None:
        ratio = max_sample_rate
        assert max_sample_rate * num_group < 1.0, f"ensure group have enough sample for {max_sample_rate} rate"
        if visible_feature_num is not None:
            k = torch.round(ratio * visible_feature_num).clamp(min=1, max=probabilities.shape[-1]).long().item()
        else:
            k = torch.round(torch.tensor(ratio * probabilities.shape[-1])).clamp(min=1, max=probabilities.shape[-1]).long().item()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    group_sample_mask = grouped_sample_inter(probabilities, num_group=num_group, 
                                        max_prob_for_each_group=max_prob_for_each_group, 
                                        num_generation=num_generation)
    num_categories = probabilities.shape[-1]
    ele_in_each_group = num_categories // num_group
    flat_probs = probabilities.view(-1, num_categories)
    flat_probs = flat_probs * group_sample_mask
    flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True) + 1e-8)
    # clamp避免group内有太大的采样偏见
    # print("max_selected_values_prev", (flat_probs*group_sample_mask).max(dim=-1).values)
    # flat_probs = flat_probs.clamp(max=3/ele_in_each_group, min=1/ele_in_each_group/3)
    flat_probs = flat_probs.clamp(max=2/ele_in_each_group, min=1/ele_in_each_group/2)
    flat_probs = flat_probs * group_sample_mask
    flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True) + 1e-8)
    # print("max_selected_values_clip", (flat_probs*group_sample_mask).max(dim=-1).values)
    indices = torch.multinomial(flat_probs, num_samples=k, replacement=False)
    indices = indices.view(num_generation, k)
    mask = torch.zeros_like(probabilities).long().repeat(num_generation, 1)
    src = torch.ones_like(indices)
    mask.scatter_(-1, indices, src)
    return mask

def chunk_sample(probabilities, num_generation=32, k=8):
    total_frame = probabilities.shape[-1]
    chunk_size = math.ceil(total_frame / k)
    align_size = chunk_size * k
    probabilities_pad = F.pad(probabilities, (0, align_size-total_frame), value=0)
    probabilities_chunk = probabilities_pad.reshape(k, chunk_size)
    shift = torch.arange(k).to(probabilities.device)*chunk_size
    indices = torch.multinomial(probabilities_chunk, num_samples=num_generation, replacement=True).T+shift

    mask = torch.zeros_like(probabilities).long().repeat(num_generation, 1)
    src = torch.ones_like(indices)
    mask.scatter_(-1, indices, src)
    return mask


class CompressionPolicyBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.text_config.hidden_size
        self.max_sample_rate = config.vision_config.max_sample_rate
        config._attn_implementation = config._attn_implementation
        self.rotary_emb = Qwen2RotaryEmbedding(config=config.text_config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config.text_config, layer_idx) for layer_idx in range(1)]
        )
        self.layernorm = Qwen2RMSNorm(embed_dim, eps=config.text_config.rms_norm_eps)
        self.frame_compress = nn.Sequential(nn.Linear(embed_dim, embed_dim//2, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//2, embed_dim//4, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//4, embed_dim//8, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//8, embed_dim//16, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//16, 2, bias=True))
        self.token_compress = nn.Sequential(nn.Linear(embed_dim, embed_dim//2, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//2, embed_dim//4, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//4, embed_dim//8, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//8, embed_dim//16, bias=True), nn.SiLU(),
                                            nn.Linear(embed_dim//16, 2, bias=True))

    def forward(self, video_embeds, tau=1, return_logits=False, topk=0.1, num_frames=8, frame_select_strategy=False, **kwargs):
        inputs_embeds = kwargs['inputs_embeds']
        attention_mask = kwargs['attention_mask']
        position_ids = kwargs['position_ids']
        video_mask = kwargs['video_mask']
        num_generation = kwargs['num_generation']
        video_grid_thw = kwargs['video_grid_thw']
        max_sample_rate = kwargs.get('max_sample_rate', None)

        # video_seq_len = video_seq_len[0].item()

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
        
        # 取出视频特征，去掉newline
        video_embeds = hidden_states[video_mask, :]
        video_embeds_ori = inputs_embeds[video_mask, :]

        # 视频特征评估
        token_compress_logits = self.token_compress(self.layernorm(video_embeds))

        t, h, w = video_grid_thw
        frame_embeds = video_embeds.reshape(t, h, w, video_embeds.shape[-1]).mean(dim=(1, 2))
        if frame_embeds.shape[0] != t:
            t = frame_embeds.shape[0]
        frame_compress_logits = self.frame_compress(self.layernorm(frame_embeds))
        
        if return_logits: # 如果要输出embedding这里也要输出
            frame_compress_logits = frame_compress_logits.repeat_interleave(h*w, dim=0)
            total_compress_logits = frame_compress_logits + token_compress_logits
            return total_compress_logits, {}
    
    def dynamic_sample_for_training(self, frame_compress_logits, token_compress_logits, max_sample_rate, video_grid_thw, num_generation):
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = (frame_compress_logits_diff).softmax(dim=-1)
        frame_compress_logits_diff = frame_compress_logits_diff.repeat_interleave(h*w, dim=-1)
        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(1, -1)
        token_compress_softmax_reshape = (token_compress_logits_diff+frame_compress_logits_diff).softmax(dim=-1)

        if max_sample_rate is None:
            max_sample_rate = self.max_sample_rate
        num_group = int(1 / max_sample_rate / 3) # 每一组里有三倍的token可以选
        max_prob_for_each_group = max(min(3 / num_group, 0.9), 0.1) # 采样概率最大不超过三倍的均值
        print("MAX_PROB_FOR_EACH_GROUP", max_prob_for_each_group)
        token_compress_sample = sample_without_replacement_h(token_compress_softmax_reshape, 
                                                            max_sample_rate=max_sample_rate, num_generation=num_generation,
                                                            num_group=num_group, max_prob_for_each_group=max_prob_for_each_group).reshape(-1)
        print("PROB - ", 
            f"frame_prob_max: {frame_compress_softmax_reshape.max().item()}",
            f"frame_prob_min: {frame_compress_softmax_reshape.min().item()}",
            f"token_prob_max: {token_compress_softmax_reshape.max().item()}",
            f"token_prob_min: {token_compress_softmax_reshape.min().item()}")
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v10(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧窗口采样+特征TopK
        t, h, w = video_grid_thw
        logit_scale = 1.0
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_logits_diff = frame_compress_logits_diff.repeat_interleave(h*w, dim=-1)
        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        token_compress_softmax_reshape = ((token_compress_logits_diff+frame_compress_logits_diff) * logit_scale).softmax(dim=-1)[0]
        token_sorted_cumsum = token_compress_softmax_reshape.cumsum(dim=0)
        num_token = math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))
        threshold = (torch.arange(num_token).to(frame_compress_logits_diff.device) + 0.5) / num_token
        topk_index = (token_sorted_cumsum * (token_sorted_cumsum < threshold.unsqueeze(-1))).argmax(dim=-1) + 1
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v9(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧窗口采样+特征TopK
        t, h, w = video_grid_thw
        logit_scale = 1.0
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_logits_diff = frame_compress_logits_diff.repeat_interleave(h*w, dim=-1)
        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        similarity = video_embeds_ori @ video_embeds_ori.mean(dim=0)
        sorted_similarity = similarity.sort(descending=True).indices
        token_compress_softmax_reshape = ((token_compress_logits_diff+frame_compress_logits_diff) * logit_scale).softmax(dim=-1)[0]
        token_sorted_cumsum = token_compress_softmax_reshape[sorted_similarity].cumsum(dim=0)
        num_token = math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))
        threshold = (torch.arange(num_token).to(frame_compress_logits_diff.device) + 0.5) / num_token
        topk_index = (token_sorted_cumsum * (token_sorted_cumsum < threshold.unsqueeze(-1))).argmax(dim=-1) + 1
        topk_index = sorted_similarity[topk_index]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v8(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧窗口采样+特征TopK
        t, h, w = video_grid_thw
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        frame_compress_sample = torch.zeros_like(frame_compress_softmax_reshape).long()
        k = min(frame_compress_sample.shape[-1], num_frames)
        k = max(k, 1)
        chunk_size = max(frame_compress_sample.shape[-1] // k, 1)
        topk_index = []
        for i in range(k):
            if i == k-1:
                topk_index.append(frame_compress_softmax_reshape[i*chunk_size:].argmax() + i*chunk_size)
            else:
                topk_index.append(frame_compress_softmax_reshape[i*chunk_size:(i+1)*chunk_size].argmax() + i*chunk_size)
        topk_index = torch.tensor(topk_index)
        if topk_index.dtype != torch.int64:
            print(topk_index.dtype)
            print(topk_index)
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)

        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        similarity = video_embeds_ori @ video_embeds_ori.mean(dim=0)
        sorted_similarity = similarity.sort(descending=True).indices
        token_compress_softmax_reshape = token_compress_logits_diff.softmax(dim=-1)[0]
        token_compress_softmax_reshape = token_compress_softmax_reshape * frame_compress_sample
        token_compress_softmax_reshape = token_compress_softmax_reshape / token_compress_softmax_reshape.sum(dim=-1, keepdim=True)
        token_sorted_cumsum = token_compress_softmax_reshape[sorted_similarity].cumsum(dim=0)
        num_token = math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))
        threshold = (torch.arange(num_token).to(frame_compress_logits_diff.device) + 0.5) / num_token
        topk_index = (token_sorted_cumsum * (token_sorted_cumsum < threshold.unsqueeze(-1))).argmax(dim=-1) + 1
        topk_index = sorted_similarity[topk_index]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v7(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧ITS采样
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        frame_compress_sample = torch.zeros_like(frame_compress_logits[..., 0]).long()
        k = min(num_frames, frame_compress_sample.shape[-1])
        k = max(k, 1)
        frame_compress_softmax_cumsum = frame_compress_softmax_reshape.cumsum(dim=-1)
        threshold = (torch.arange(k).to(frame_compress_logits_diff.device) + 0.5) / k
        topk_index = (frame_compress_softmax_cumsum * (frame_compress_softmax_cumsum < threshold.unsqueeze(-1))).argmax(dim=-1) + 1
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)
        token_compress_sample = frame_compress_sample
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v6(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧窗口采样
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        frame_compress_sample = torch.zeros_like(frame_compress_softmax_reshape).long()
        k = min(frame_compress_sample.shape[-1], num_frames)
        k = max(k, 1)
        chunk_size = max(frame_compress_sample.shape[-1] // k, 1)
        topk_index = []
        for i in range(k):
            if i == k-1:
                topk_index.append(frame_compress_softmax_reshape[i*chunk_size:].argmax() + i*chunk_size)
            else:
                topk_index.append(frame_compress_softmax_reshape[i*chunk_size:(i+1)*chunk_size].argmax() + i*chunk_size)
        topk_index = torch.tensor(topk_index)
        if topk_index.dtype != torch.int64:
            print(topk_index.dtype)
            print(topk_index)
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)
        token_compress_sample = frame_compress_sample
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v5(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧TopK-ITS采样+特征TopK
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        filtered_index = frame_compress_softmax_reshape.sort(descending=True).indices[frame_compress_softmax_reshape.shape[-1]//4:]
        frame_compress_softmax_reshape[filtered_index] = 0
        frame_compress_softmax_reshape = frame_compress_softmax_reshape / frame_compress_softmax_reshape.sum()
        frame_compress_sample = torch.zeros_like(frame_compress_logits[..., 0]).long()
        k = min(num_frames, frame_compress_sample.shape[-1])
        k = max(k, 1)
        frame_compress_softmax_cumsum = frame_compress_softmax_reshape.cumsum(dim=-1)
        threshold = (torch.arange(k).to(frame_compress_logits_diff.device) + 0.5) / k
        topk_index = (frame_compress_softmax_cumsum * (frame_compress_softmax_cumsum < threshold.unsqueeze(-1))).argmax(dim=-1) + 1
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)

        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        token_compress_softmax_reshape = token_compress_logits_diff.softmax(dim=-1)[0]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        topk = topk
        topk_index = (token_compress_softmax_reshape*frame_compress_sample).topk(math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))).indices
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v4(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧ITS采样+特征TopK
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        frame_compress_sample = torch.zeros_like(frame_compress_logits[..., 0]).long()
        k = min(num_frames, frame_compress_sample.shape[-1])
        k = max(k, 1)
        frame_compress_softmax_cumsum = frame_compress_softmax_reshape.cumsum(dim=-1)
        threshold = (torch.arange(k).to(frame_compress_logits_diff.device) + 0.5) / k
        topk_index = (frame_compress_softmax_cumsum * (frame_compress_softmax_cumsum < threshold.unsqueeze(-1))).argmax(dim=-1) + 1
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)

        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        token_compress_softmax_reshape = token_compress_logits_diff.softmax(dim=-1)[0]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        topk = topk
        topk_index = (token_compress_softmax_reshape*frame_compress_sample).topk(math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))).indices
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v3(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧TopK采样+特征TopK
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        frame_compress_sample = torch.zeros_like(frame_compress_logits[..., 0]).long()
        k = min(num_frames, frame_compress_sample.shape[-1])
        k = max(k, 1)
        topk_index = frame_compress_softmax_reshape.topk(k=k).indices
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)

        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        token_compress_softmax_reshape = token_compress_logits_diff.softmax(dim=-1)[0]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        topk = topk
        topk_index = (token_compress_softmax_reshape*frame_compress_sample).topk(math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))).indices
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v2(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧均匀采样+特征TopK
        t, h, w = video_grid_thw
        frame_compress_sample = torch.zeros_like(frame_compress_logits[..., 0]).long()
        k = min(num_frames, frame_compress_sample.shape[-1])
        topk_index = torch.linspace(0, frame_compress_sample.shape[-1] - 1, k).round().long().to(frame_compress_sample.device)
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)

        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        token_compress_softmax_reshape = token_compress_logits_diff.softmax(dim=-1)[0]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        topk = topk
        topk_index = (token_compress_softmax_reshape*frame_compress_sample).topk(math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))).indices
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v1(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 帧窗口采样+特征TopK
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = frame_compress_logits_diff.softmax(dim=-1)[0]
        frame_compress_sample = torch.zeros_like(frame_compress_softmax_reshape).long()
        k = min(frame_compress_sample.shape[-1], num_frames)
        k = max(k, 1)
        chunk_size = max(frame_compress_sample.shape[-1] // k, 1)
        topk_index = []
        for i in range(k):
            if i == k-1:
                topk_index.append(frame_compress_softmax_reshape[i*chunk_size:].argmax() + i*chunk_size)
            else:
                topk_index.append(frame_compress_softmax_reshape[i*chunk_size:(i+1)*chunk_size].argmax() + i*chunk_size)
        topk_index = torch.tensor(topk_index)
        if topk_index.dtype != torch.int64:
            print(topk_index.dtype)
            print(topk_index)
        frame_compress_sample[topk_index] = 1
        frame_compress_sample = frame_compress_sample.repeat_interleave(h*w, dim=-1).reshape(-1)

        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(num_generation, -1)
        token_compress_softmax_reshape = token_compress_logits_diff.softmax(dim=-1)[0]
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        topk = topk
        topk_index = (token_compress_softmax_reshape*frame_compress_sample).topk(math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))).indices
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape

    def dynamic_sample_for_evalation_v0(self, frame_compress_logits, token_compress_logits, topk, num_frames, video_grid_thw, num_generation, video_embeds_ori):
        # 直接作为权重
        t, h, w = video_grid_thw
        frame_compress_logits_diff = (frame_compress_logits.float()[..., 1] - frame_compress_logits.float()[..., 0]).reshape(1, -1)
        frame_compress_softmax_reshape = (frame_compress_logits_diff).softmax(dim=-1)
        frame_compress_logits_diff = frame_compress_logits_diff.repeat_interleave(h*w, dim=-1)
        token_compress_logits_diff = (token_compress_logits.float()[..., 1] - token_compress_logits.float()[..., 0]).reshape(1, -1)
        token_compress_softmax_reshape = (token_compress_logits_diff+frame_compress_logits_diff).softmax(dim=-1)
        token_compress_sample = torch.zeros_like(token_compress_softmax_reshape).long()
        topk = topk
        topk_index = token_compress_softmax_reshape.topk(math.ceil(topk * (token_compress_softmax_reshape.shape[0] - 1))).indices
        token_compress_sample[topk_index] = 1
        return token_compress_sample, token_compress_softmax_reshape, token_compress_softmax_reshape

    def visualization(self, token_compress_softmax_reshape, video_grid_thw):
        if os.environ.get('DEBUG', "0") == "1":
            name = "base"
            debug_path = "/home/mayinchao.myc/projects/LMMS_EVAL/workspace"
            t, h, w = video_grid_thw
            score_map = token_compress_softmax_reshape.reshape(t, h, w)
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min()+1e-8)
            image_heat = add_heat_to_image(score_map.detach().cpu().numpy(), debug_path, temporal_patch=1)
            plt.imsave(f"{debug_path}/image_heat.png", image_heat)
        if os.environ.get('DEBUG', "0") == "2":
            name = "base"
            debug_path = "/home/mayinchao.myc/projects/Video-R1/workspace"
            t, h, w = video_grid_thw
            score_map = token_compress_softmax_reshape.reshape(t, h, w)
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min()+1e-8)
            image_heat = add_heat_to_image(score_map.detach().cpu().numpy(), debug_path, temporal_patch=1)
            plt.imsave(f"{debug_path}/image_heat.png", image_heat)