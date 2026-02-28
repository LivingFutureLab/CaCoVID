# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import re
import sys
import yaml
import math
import json
import shutil
import random
import inspect
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
from rouge_score import rouge_scorer
from PIL import Image

from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Union

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, set_seed
from math_verify import parse, verify

from trainer import CPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers.training_args import OptimizerNames

from torch.utils.data import Dataset
# from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from decord import VideoReader, cpu


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )

    max_num_images: int = 256
    video_root: str = ""


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    freeze_language_modules: bool = False
    freeze_policy_layers_modules: bool = False
    apply_compression: bool = False
    compression_policy: str = None
    max_sample_rate: float = 1.0
    video_selection_kl_weight: float = 0.01
    grad_scale: float = 1.0
    frame_ratio: float = 0.5
    lr_drop_for_attn: float = 0.1
    group_size: float = 2.0
    max_group_prob: float = 2.0

@dataclass
class CustomerGRPOConfig(GRPOConfig):
    learning_rate_for_compression_policy: float = field(
        default=1e-4,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    # adamw_torch
    optim: Union[OptimizerNames, str] = field(
        default='adamw_torch',
        metadata={"help": "The optimizer to use."},
    )


def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if (output_ans.strip().strip(".")[:1] == gt_ans.strip().strip(".")[:1] or content.strip().strip(".")[:1] == gt_ans.strip().strip(".")[:1]) else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward
}

def read_video_decode(video_path, num_frames, decode_fps=1.0):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps
    total_frames_with_decode_fps = int(duration * decode_fps)

    num_frames = min(num_frames, total_frames)
    num_frames = min(num_frames, total_frames_with_decode_fps)
    
    select_idx = [round(_) for _ in np.linspace(0, total_frames-1, num_frames)]

    images = []
    for i in select_idx:
        frame = vr[i]
        frame_rgb = frame.asnumpy() #[:, :, ::-1]  # 转换为 RGB 格式
        images.append(Image.fromarray(frame_rgb))
    del vr 
    return images, duration

class CustomDataset(Dataset):
    def __init__(self, table_name: str, script_args: GRPOScriptArguments):
        super(CustomDataset, self).__init__()
        self.list_data_dict = []

        table_name_list = [e.strip() for e in table_name.split(",")]

        for table_name in table_name_list:
            if table_name.endswith(".json"):
                with open(table_name, "r") as f:
                    self.list_data_dict += json.load(f)

        print("\nTotal {} samples.".format(len(self.list_data_dict)))

        ## data checking
        for d in self.list_data_dict:
            for k in ["problem", "solution", "modality_type", "problem_type"]:
                assert k in d, "field {} not in data".format(k)
        
        self.script_args = script_args
        
        self.QUESTION_TEMPLATE = (
            "{Question}\n"
            # "Please think about this question as if you were a human pondering deeply. "
            # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            # "It's encouraged to include self-reflection or verification in the reasoning process. "
            # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
        )

        self.TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.).",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14).",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer.",
            "free-form": " Please provide your text answer.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14)."
        }
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        while True:
            # try:
            example = self.list_data_dict[i]

            problem = example["problem"]
            solution = example["solution"]
            query = self.QUESTION_TEMPLATE.format(Question=problem) + self.TYPE_TEMPLATE[example['problem_type']]
            
            modality_type = example["modality_type"]
            assert modality_type in ["image", "video"], "modality_type {} not support".format(modality_type)

            if modality_type == "video":
                images, duration = read_video_decode(os.path.join(self.script_args.video_root, example["video"]), self.script_args.max_num_images)
                fps = len(images) / duration
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                            },
                            {
                                "type": "text", 
                                "text": query
                            },
                        ],
                    },
                ]
            else:
                raise ValueError(f"Invalid modality_type: {modality_type}")

            out = {
                'problem': problem,
                'solution': solution,
                "problem_type": example["problem_type"],
                'prompt': prompt,
                'modality': modality_type,
            }
            out['video'] = images
            out["fps"] = fps
            assert images[0].height > 28 and images[0].width > 28, "low resoluton video"
            return out
            # except:
            #     print("resample")
            #     i = random.randint(0, len(self.list_data_dict)-1)

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = CustomDataset(script_args.dataset_name, script_args)
    
    trainer_cls = CPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        freeze_vision_modules=model_args.freeze_vision_modules,
        freeze_language_modules=model_args.freeze_language_modules,
        freeze_policy_layers_modules=model_args.freeze_policy_layers_modules,
        frame_ratio=model_args.frame_ratio,
        apply_compression=model_args.apply_compression,
        compression_policy=model_args.compression_policy,
        max_sample_rate=model_args.max_sample_rate,
        video_selection_kl_weight=model_args.video_selection_kl_weight,
        group_size=model_args.group_size,
        max_group_prob=model_args.max_group_prob
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, CustomerGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    main(script_args, training_args, model_args)
