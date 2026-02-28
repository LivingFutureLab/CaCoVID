import os
import math
import torch
import numpy as np
import deepspeed
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def generate_coordinates(t, h, w):
    # 生成各个维度的索引
    t_coords = torch.arange(t)
    h_coords = torch.arange(h)
    w_coords = torch.arange(w)
    # 使用 meshgrid 生成三维坐标网格（indexing='ij' 保持与 NumPy 一致的索引方式）
    t_grid, h_grid, w_grid = torch.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
    # 将三个坐标张量在最后一个维度拼接，得到形状为 (t, h, w, 3) 的张量
    coords = torch.stack([t_grid, h_grid, w_grid], dim=-1)
    return coords

def masked_norm(p, mask, masked_value=0):
    p[~mask.bool()] = 0
    return p / (p*mask).sum()

def clip_min_probability(p, min_prob=0.3, mask=None) -> torch.Tensor:
    p = p.clone()
    # 避免非零元素无法支持最大概率限制，所有p=0元素的概率之和不要超过1/num_p
    # mask归一化
    if mask is not None:
        p = masked_norm(p, mask)
    else:
        p = p / p.sum()
        mask = torch.ones_like(p)
    # 判断min_prob能否支持目前的非零元素个数（按理来说clamp之后就没这个问题了）
    if (mask == 1).sum() == 1:
        return p
    # min_prob = min(min_prob, 0.999)
    assert min_prob * mask.sum() < 1.0, f"{min_prob}, {mask.sum()}: min_prob is too small!"
    p_ = p.clone()
    p_[p_ == 0] = 1
    p_min_value, p_min_index = p_.min(), p_.argmin()
    # 如果最大p满足条件，直接返回
    if p_min_value > min_prob or (mask == 1).sum() == 1:
        return p
    # 不满足条件继续递归，递归深度不超过1/min_prob
    mask[p_min_index] = 0
    rest_target = 1 - min_prob
    p_sub = clip_min_probability(p, min_prob / rest_target, mask) * rest_target
    p_sub[p_min_index] = min_prob
    return p_sub

def clip_max_probability(p, max_prob=0.3, mask=None) -> torch.Tensor:
    p = p.clone()
    # 避免非零元素无法支持最大概率限制，所有p=0元素的概率之和不要超过1/num_p
    p = p.clamp(min=1/p.shape[0]**2)
    # mask归一化
    if mask is not None:
        p = masked_norm(p, mask)
    else:
        p = p / p.sum()
        mask = torch.ones_like(p)
    # 判断max_prob能否支持目前的非零元素个数（按理来说clamp之后就没这个问题了）
    if (mask == 1).sum() == 1:
        return p
    max_prob = min(max_prob, 0.999)
    assert max_prob * mask.sum() > 1.0, f"{max_prob}, {mask.sum()}: max_prob is too small!"
    p_max_value, p_max_index = p.max(), p.argmax()
    # 如果最大p满足条件，直接返回
    if p_max_value <= max_prob or (mask == 1).sum() == 1:
        return p
    # 不满足条件继续递归，递归深度不超过1/max_prob
    mask[p_max_index] = 0
    rest_target = 1 - max(min(max_prob, 0.999), 0.001)
    p_sub = clip_max_probability(p, max_prob / rest_target, mask) * rest_target
    p_sub[p_max_index] = max_prob
    return p_sub

def grouped_sample(p_tensor: torch.Tensor, num_group: int, max_prob_for_each_group: float, num_generation=8) -> torch.Tensor:
    # 所有generate的p都一样
    p = p_tensor[0]
    p_value, p_index = p.sort(dim=-1, descending=True)
    p_len = len(p_value)
    assert num_group <= p_len, "group must be less than or equal to the length of the p_tensor"
    # 处理无法整除num_group的情况
    base_size = p_len // num_group
    remainder = p_len % num_group
    lengths = []
    for i in range(num_group):
        if i < remainder:
            lengths.append(base_size + 1)
        else:
            lengths.append(base_size)
    # 计算每个group的采样概率
    chunks = torch.split(p_value, lengths)
    index_chunks = torch.split(p_index, lengths)
    p_group = torch.stack([chunk.sum() for chunk in chunks]).to(p)
    # clamp保证所有group都有概率被采样到，避免某个group采样概率太小或太大
    p_group = clip_max_probability(p_group, max_prob=max_prob_for_each_group)
    p_group = p_group / p_group.sum()
    # 计算每个generate的采样group
    indices = torch.multinomial(p_group, num_samples=num_generation, replacement=True) # 有放回采样
    sample_mask = torch.zeros_like(p_tensor).repeat(num_generation, 1)
    for i, j in enumerate(indices):
        sample_mask[i, index_chunks[j]] = 1
    # 返回对应group的mask
    return sample_mask

def clip_max_probability(p, max_prob=0.3, mask=None) -> torch.Tensor:
    p = p.clone()
    # 避免非零元素无法支持最大概率限制，所有p=0元素的概率之和不要超过1/num_p
    p = p.clamp(min=1/p.shape[0]**2)
    # mask归一化
    if mask is not None:
        p = masked_norm(p, mask)
    else:
        p = p / p.sum()
        mask = torch.ones_like(p)
    # 判断max_prob能否支持目前的非零元素个数（按理来说clamp之后就没这个问题了）
    if (mask == 1).sum() == 1:
        return p
    max_prob = min(max_prob, 0.999)
    assert max_prob * mask.sum() > 1.0, f"{max_prob}, {mask.sum()}: max_prob is too small!"
    p_max_value, p_max_index = p.max(), p.argmax()
    # 如果最大p满足条件，直接返回
    if p_max_value <= max_prob or (mask == 1).sum() == 1:
        return p
    # 不满足条件继续递归，递归深度不超过1/max_prob
    mask[p_max_index] = 0
    rest_target = 1 - max(min(max_prob, 0.999), 0.001)
    p_sub = clip_max_probability(p, max_prob / rest_target, mask) * rest_target
    p_sub[p_max_index] = max_prob
    return p_sub

def grouped_sample_inter(p_tensor: torch.Tensor, num_group: int, max_prob_for_each_group: float, num_generation=8) -> torch.Tensor:
    # 所有generate的p都一样
    p = p_tensor[0]
    p_value, p_index = p.sort(dim=-1, descending=True)
    p_len = len(p_value)
    assert num_group <= p_len, "group must be less than or equal to the length of the p_tensor"
    # 处理无法整除num_group的情况
    num_group_inter = num_group // 2 + 1
    base_size = p_len // num_group_inter
    remainder = p_len % num_group_inter
    lengths = []
    for i in range(num_group_inter):
        if i < remainder:
            lengths.append(base_size + 1)
        else:
            lengths.append(base_size)
    # 计算每个group的采样概率
    chunks = torch.split(p_value, lengths)
    index_chunks = torch.split(p_index, lengths)
    chunks_inter = torch.split(p_value[base_size//2:base_size//2+sum(lengths[:num_group-num_group_inter])], lengths[:num_group-num_group_inter])
    index_chunks_inter = torch.split(p_index[base_size//2:base_size//2+sum(lengths[:num_group-num_group_inter])], lengths[:num_group-num_group_inter])
    chunks = chunks + chunks_inter
    index_chunks = index_chunks + index_chunks_inter
    p_group = torch.stack([chunk.sum() for chunk in chunks]).to(p)
    p_group = p_group / p_group.sum()
    # clamp保证所有group都有概率被采样到，避免某个group采样概率太小或太大
    p_group = clip_max_probability(p_group, max_prob=max_prob_for_each_group)
    p_group = p_group / p_group.sum()
    # 计算每个generate的采样group
    indices = torch.multinomial(p_group, num_samples=num_generation, replacement=True) # 有放回采样
    sample_mask = torch.zeros_like(p_tensor).repeat(num_generation, 1)
    for i, j in enumerate(indices):
        sample_mask[i, index_chunks[j]] = 1
    # 返回对应group的mask
    return sample_mask