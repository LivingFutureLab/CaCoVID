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

import re
import os
import oss2
import types
import textwrap
import contextlib
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
import random

import time
import shutil
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)



import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import deepspeed
import transformers
import matplotlib.pyplot as plt
from datasets import Dataset, IterableDataset
from packaging import version
# from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
# from .callbacks import SyncRefModelCallback
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_sagemaker_mp_enabled
from .llava_onevision import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from torch.utils.data import Sampler
from transformers.training_args import OptimizerNames

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from accelerate.utils import is_peft_model, set_seed, DistributedType
from accelerate.utils import is_deepspeed_available
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

class CPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        freeze_language_modules: Optional[bool] = False,
        freeze_policy_layers_modules: Optional[bool] = False,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        compression_policy: str = "CompressionPolicy",
        apply_compression: Optional[bool] = False,
        max_sample_rate: float = 1.0,
        video_selection_kl_weight: float = 0.01,
        num_iter_for_item: int = 5,
        frame_ratio: float = 0.5,
        group_size: float = 2.0,
        max_group_prob: float = 2.0,
    ):
        self.apply_compression = apply_compression
        self.freeze_language_modules = freeze_language_modules
        self.video_selection_kl_weight = video_selection_kl_weight
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.epsilon_low = 0.2 # 防止对某一个样本过度优化
        self.epsilon_high = 0.28 # 防止对某一个样本过度优化
        self.num_iter_for_item = num_iter_for_item # 对单一样本多次迭代提高样本利用率
        self.max_sample_rate = max_sample_rate # 采样比例
        self.max_sample_rate_for_repeat_sample = self.max_sample_rate # 自适应采样率：困难样本多采样、简单样本少采样
        self.max_num_images = script_args.max_num_images
        self.group_size = group_size
        self.max_group_prob = max_group_prob
        sample_mode_num = {
            "content": 1,
            "token": args.num_generations//4*3-1,
            "frame": args.num_generations//4
        }
        self.sample_mode_num = sample_mode_num
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        self.memory = {}
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            if args.fp16:
                model_init_kwargs["torch_dtype"] = torch.float16
            if args.bf16:
                model_init_kwargs["torch_dtype"] = torch.bfloat16
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs["compression_policy"] = compression_policy
            model_init_kwargs["apply_compression"] = apply_compression
            model_init_kwargs["max_sample_rate"] = max_sample_rate
            model_init_kwargs["frame_ratio"] = frame_ratio
            model_init_kwargs["sample_mode_num"] = self.sample_mode_num
            model_init_kwargs["group_size"] = self.group_size
            model_init_kwargs["max_group_prob"] = self.max_group_prob

            model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            if self.apply_compression and getattr(model.compression_policy, "layers", None) is not None:
                with deepspeed.zero.GatheredParameters(
                    list(model.compression_policy.layers.parameters()) + list(model.language_model.model.layers[:len(model.compression_policy.layers)].parameters()), modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        for i in range(len(model.compression_policy.layers)):
                            load_info = model.compression_policy.layers[i].load_state_dict(model.language_model.model.layers[i].state_dict())
                            shape_for_check = model.compression_policy.layers[0].self_attn.q_proj.weight.shape
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # 冻结模型参数，默认冻结所有大模型参数
        self.vision_modules_keywords = ["visual", "multi_modal_projector"]
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
        self.language_modules_keywords = ["model"]

        if freeze_language_modules:
            print("Freezing language modules...")
            for n, p in model.named_parameters():
                if any((keyword in n and "compression_policy" not in n) for keyword in self.language_modules_keywords):
                    p.requires_grad = False
        self.policy_layers_modules_keywords = ["compression_policy.layers"]
        if freeze_policy_layers_modules:
            print("Freezing policy modules...")
            for n, p in model.named_parameters():
                if any((keyword in n) for keyword in self.policy_layers_modules_keywords):
                    p.requires_grad = False
        for n, p in model.named_parameters():
            if p.requires_grad == True:
                print(f"    unfreeze {n}")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Processing class
        if processing_class is None:
            processing_class = LlavaOnevisionProcessor.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            processing_class.image_processor.max_pixels = max_pixels
            processing_class.image_processor.min_pixels = min_pixels

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        assert sum(sample_mode_num.values()) == args.num_generations

        # 输出确定的结果
        self.generation_config = GenerationConfig( 
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.01,  
            temperature=0.01, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        result = model(input_ids, freeze_language_modules=self.freeze_language_modules, **kwargs)
        if self.apply_compression:
            video_selection_logits = result.video_selection_logits
            video_selection_logps = video_selection_logits.log_softmax(dim=-1)
            # 多次迭代的情况下保留第一次迭代时的logp用于约束更新的梯度
            if self.valid_iter == 0:
                self.old_token_selection_logits = video_selection_logits.detach()
                self.old_token_selection_logps = video_selection_logps.detach()
            
            # 拼接之前的采样mask，这些mask已经有了对应的奖励可以直接复用
            video_selection_mask = self.memory['token_mask'].long()
            video_selection_logps = video_selection_logps.unsqueeze(0).repeat(video_selection_mask.shape[0], 1, 1)
            video_selection_logps = video_selection_logps.gather(-1, video_selection_mask.unsqueeze(-1)).squeeze(-1)
            video_selection_logps = video_selection_logps.reshape(video_selection_mask.shape[0], -1)

            frame_selection_logits = result.frame_selection_logits
            frame_selection_logps = frame_selection_logits.log_softmax(dim=-1)
            if self.valid_iter == 0:
                self.old_frame_selection_logits = frame_selection_logits.detach()
                self.old_frame_selection_logps = frame_selection_logps.detach()
            frame_selection_mask = self.memory['frame_mask'].long()
            frame_selection_logps = frame_selection_logps.unsqueeze(0).repeat(frame_selection_mask.shape[0], 1, 1)
            frame_selection_logps = frame_selection_logps.gather(-1, frame_selection_mask.unsqueeze(-1)).squeeze(-1)
            frame_selection_logps = frame_selection_logps.reshape(frame_selection_mask.shape[0], -1)
            
            mode_selection_logps = result.mode_selection_logits.log_softmax(dim=-1).reshape(-1)
            if self.valid_iter == 0:
                self.old_mode_selection_logps = mode_selection_logps.detach()
            all_token_selection_mask = result.all_token_selection_mask
            if all_token_selection_mask is not None:
                assert all_token_selection_mask.sum(-1).unique().shape[0] == 1
        else:
            video_selection_logps = None
            all_token_selection_mask = None

        logits = result.logits
        if logits is not None:
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            
            # 这里的logits筛选之后的，所以需要将input_id与logits用all_token_selection_mask对齐
            if all_token_selection_mask is not None:
                input_ids = input_ids[all_token_selection_mask].reshape(input_ids.shape[0], -1)
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps), video_selection_logps, frame_selection_logps, mode_selection_logps
        else:
            return None, video_selection_logps, frame_selection_logps, mode_selection_logps
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # 现在self.args.learning_rate_for_compression_policy与self.args.learning_rate是一样的
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'compression_policy' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'compression_policy' not in n)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'compression_policy' in n and 'layers' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate_for_compression_policy,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'compression_policy' in n and 'layers' not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate_for_compression_policy,
                },
                {
                    # 由于注意力层使用的预训练过的，所以学习率降低到10%
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'compression_policy' in n and 'layers' in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate_for_compression_policy*0.1,
                },
                {
                    # 由于注意力层使用的预训练过的，所以学习率降低到10%
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'compression_policy' in n and 'layers' in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate_for_compression_policy*0.1,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        # logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                # logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            # elif is_torch_hpu_available():
            #     logger.warning(
            #         "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
            #     )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        modality_types = [x["modality"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        if modality_types[0] == "image":
            image_inputs = [[image] for image in inputs[0]["image"]]

            prompt_inputs = self.processing_class(
                    text=copy.deepcopy(prompts_text),
                    images=image_inputs,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
            video_inputs = None
        elif modality_types[0] == "video":
            video_inputs = [example["video"] for example in inputs]
            fps = [example["fps"] for example in inputs]

            prompt_inputs = self.processing_class(
                    text=copy.deepcopy(prompts_text),
                    videos=video_inputs,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    fps=fps,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
            image_inputs = None
        
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # 输出的prompt_completion_ids是完整未被压缩的
            prompt_completion_ids, vision_info = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config, 
                                                                          return_vision_info=True, max_sample_rate=self.max_sample_rate_for_repeat_sample,
                                                                          old_frame_policy_logits=self.old_frame_selection_logits, 
                                                                          old_token_policy_logits=self.old_token_selection_logits)
            prompt_inputs['video_mask_for_refer_model'] = vision_info['video_mask']
            prompt_inputs['video_selection_mask_for_refer_model'] = vision_info['video_selection_mask']
            prompt_inputs['frame_selection_mask_for_refer_model'] = vision_info['frame_selection_mask']
            prompt_inputs['video_embeds_for_refer_model'] = vision_info['video_embeds']
            keep_token_num = vision_info['keep_token_num']
            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        prompt_inputs.pop("input_ids")
        # 要保持跟初始长度一致，因为要在再次推理时过滤掉complation的特征，保持跟初始的策略网络信息一致，控制输入用的
        prompt_inputs['prompt_length'] = prompt_length
        prompt_inputs['attention_mask'] = attention_mask
        
        if inputs[0]['modality'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()
        

        if inputs[0]['modality'] == 'video':
            if prompt_inputs["pixel_values_videos"].dim() == 5:
                prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1, 1, 1, 1)
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        # Sum the rewards from all reward functions
        if self.apply_compression:
            rewards_for_compression = rewards_per_func[:, [0]].sum(dim=1) # accuracy
        
        if os.environ.get("PRINT_LOG", "0") == "1":
            print("REWARD:", rewards_for_compression[[0, self.sample_mode_num['content'], self.sample_mode_num['content']+self.sample_mode_num['token']]].tolist())

        ####################### update memory
        selection_mask_for_memory = prompt_inputs['video_selection_mask_for_refer_model'].reshape(self.num_generations, -1)[:self.sample_mode_num['content']+self.sample_mode_num['token']]
        frame_selection_mask_for_memory = prompt_inputs['frame_selection_mask_for_refer_model'].reshape(self.sample_mode_num['frame'], -1)
        if self.memory.get("token_mask", None) is None:
            self.memory['token_mask'] = selection_mask_for_memory[self.sample_mode_num['content']:].long()
            self.memory['frame_mask'] = frame_selection_mask_for_memory.long()
            self.memory['token_reward'] = rewards_for_compression[self.sample_mode_num['content']:self.sample_mode_num['content']+self.sample_mode_num['token']]
            self.memory['frame_reward'] = rewards_for_compression[-self.sample_mode_num['frame']:]
            self.memory['mode_reward'] = rewards_for_compression[[0, self.sample_mode_num['content'], self.sample_mode_num['content']+self.sample_mode_num['token']]]
        else:
            self.memory['token_mask'] = torch.cat([self.memory['token_mask'], selection_mask_for_memory[self.sample_mode_num['content']:]], dim=0).long()
            self.memory['frame_mask'] = torch.cat([self.memory['frame_mask'], frame_selection_mask_for_memory], dim=0).long()
            self.memory['token_reward'] = torch.cat([self.memory['token_reward'], rewards_for_compression[self.sample_mode_num['content']:self.sample_mode_num['content']+self.sample_mode_num['token']]], dim=0)
            self.memory['frame_reward'] = torch.cat([self.memory['frame_reward'], rewards_for_compression[-self.sample_mode_num['frame']:]], dim=0)
            self.memory['mode_reward'] = torch.cat([self.memory['mode_reward'], rewards_for_compression[[0, self.sample_mode_num['content'], self.sample_mode_num['content']+self.sample_mode_num['token']]]], dim=0)
        ####################### 


        per_token_logps, video_selection_logps, frame_selection_logps, mode_selection_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
        if not self.freeze_language_modules:
            if keep_token_num is None:
                per_token_logps = per_token_logps[:, prompt_length - 1 :]
            else:
                per_token_logps = per_token_logps[:, keep_token_num - 1 :]

        # Compute grouped-wise rewards
        loss = 0

        # Compute grouped-wise rewards
        if self.apply_compression:
            mean_grouped_token_reward = self.memory['token_reward'].view(-1).mean(dim=0, keepdim=True)
            std_grouped_token_reward = self.memory['token_reward'].view(-1).std(dim=0, keepdim=True)
            advantages_for_token = (self.memory['token_reward'] - mean_grouped_token_reward) / (std_grouped_token_reward + 1e-4)

            # 取出old_logp
            video_selection_mask = self.memory['token_mask']
            old_token_selection_logps = self.old_token_selection_logps.unsqueeze(0).repeat(video_selection_mask.shape[0], 1, 1)
            old_token_selection_logps = old_token_selection_logps.gather(-1, video_selection_mask.unsqueeze(-1)).squeeze(-1)

            # 用于clip梯度避免过度优化
            coef_1 = torch.exp(video_selection_logps - old_token_selection_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            video_selection_loss1 = coef_1 * advantages_for_token.unsqueeze(1)
            video_selection_loss2 = coef_2 * advantages_for_token.unsqueeze(1)
            video_selection_loss = - torch.min(video_selection_loss1, video_selection_loss2)

            # 平衡正负样本在损失中的权重
            video_selection_mask = self.memory['token_mask'].reshape(*video_selection_loss.shape)
            video_selection_mask_mean = video_selection_mask.float().mean(dim=-1, keepdim=True)
            video_selection_mask_weight = (video_selection_mask * (1 - video_selection_mask_mean) + (1 - video_selection_mask) * video_selection_mask_mean).detach()
            video_selection_mask_weight = video_selection_mask_weight # * clip_mask
            video_selection_loss = ((video_selection_loss * video_selection_mask_weight).sum(dim=-1) / (video_selection_mask_weight.sum(dim=-1) + 1e-7)).mean()
            
            # 用于clip梯度避免过度优化
            mean_grouped_frame_reward = self.memory['frame_reward'].view(-1).mean(dim=0, keepdim=True)
            std_grouped_frame_reward = self.memory['frame_reward'].view(-1).std(dim=0, keepdim=True)
            advantages_for_frame = (self.memory['frame_reward'] - mean_grouped_frame_reward) / (std_grouped_frame_reward + 1e-4)
            frame_selection_mask = self.memory['frame_mask']
            old_frame_selection_logps = self.old_frame_selection_logps.unsqueeze(0).repeat(frame_selection_mask.shape[0], 1, 1)
            old_frame_selection_logps = old_frame_selection_logps.gather(-1, frame_selection_mask.unsqueeze(-1)).squeeze(-1)

            coef_1 = torch.exp(frame_selection_logps - old_frame_selection_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            frame_selection_loss1 = coef_1 * advantages_for_frame.unsqueeze(1)
            frame_selection_loss2 = coef_2 * advantages_for_frame.unsqueeze(1)
            frame_selection_loss = - torch.min(frame_selection_loss1, frame_selection_loss2)
            frame_selection_loss = frame_selection_loss.mean()

            mean_grouped_mode_reward = self.memory['mode_reward'].view(-1).mean(dim=0, keepdim=True)
            std_grouped_mode_reward = self.memory['mode_reward'].view(-1).std(dim=0, keepdim=True)
            advantages_for_mode = (self.memory['mode_reward'] - mean_grouped_mode_reward) / (std_grouped_mode_reward + 1e-4)
            mode_selection_logps = mode_selection_logps.repeat(self.memory['mode_reward'].shape[0] // self.old_mode_selection_logps.shape[0])
            old_mode_selection_logps = self.old_mode_selection_logps
            old_mode_selection_logps = old_mode_selection_logps.repeat(self.memory['mode_reward'].shape[0] // self.old_mode_selection_logps.shape[0])
            coef_1 = torch.exp(mode_selection_logps - old_mode_selection_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

            mode_selection_loss1 = coef_1 * advantages_for_mode
            mode_selection_loss2 = coef_2 * advantages_for_mode
            mode_selection_loss = - torch.min(mode_selection_loss1, mode_selection_loss2)
            mode_selection_loss = mode_selection_loss.mean()

            if self.video_selection_kl_weight > 0:
                ref_video_selection_logps = (torch.ones_like(video_selection_logps) * 0.5).log()
                video_selection_kl = torch.exp(ref_video_selection_logps - video_selection_logps) - (ref_video_selection_logps - video_selection_logps) - 1       # [B, 115], 对 KL 的泰勒展开
                video_selection_mean_kl = ((video_selection_kl * video_selection_mask_weight).sum(dim=-1) / (video_selection_mask_weight.sum(dim=-1) + 1e-7)).mean()
                video_selection_loss = video_selection_loss + self.video_selection_kl_weight * video_selection_mean_kl
                self._metrics["video_selection_kl"].append(self.accelerator.gather_for_metrics(video_selection_mean_kl).mean().item())

                if self.memory.get('frame_mask', None) is not None:
                    ref_frame_selection_logps = (torch.ones_like(frame_selection_logps) * 0.5).log()
                    frame_selection_kl = torch.exp(ref_frame_selection_logps - frame_selection_logps) - (ref_frame_selection_logps - frame_selection_logps) - 1       # [B, 115], 对 KL 的泰勒展开
                    frame_selection_mean_kl = frame_selection_kl.mean()
                    frame_selection_loss = frame_selection_loss + self.video_selection_kl_weight * frame_selection_mean_kl
                    self._metrics["frame_selection_kl"].append(self.accelerator.gather_for_metrics(frame_selection_mean_kl).mean().item())

            loss = loss + video_selection_loss + frame_selection_loss

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        self.accuracy = self.memory['token_reward'].mean()
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        if self.apply_compression:
            self._metrics["rewards_for_token"].append(self.accelerator.gather_for_metrics(self.memory['token_reward']).mean().item())
            self._metrics["rewards_for_token_std"].append(self.accelerator.gather_for_metrics(std_grouped_token_reward).mean().item())
            self._metrics["rewards_for_frame"].append(self.accelerator.gather_for_metrics(self.memory['frame_reward']).mean().item())
            self._metrics["rewards_for_frame_std"].append(self.accelerator.gather_for_metrics(std_grouped_frame_reward).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        # logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, args.gradient_accumulation_steps, args.device)
                # 重置参数
                self.max_sample_rate_for_repeat_sample = self.max_sample_rate
                self.accuracy = 0.5
                self.memory = {}
                self.valid_iter = 0
                self.repeat_num = 0
                self.old_frame_selection_logits = None
                self.old_token_selection_logits = None
                if self.max_num_images < 100:
                    low_thres = 0.01
                    high_thres = 0.12
                # 当有效探索达到规定数量时迭代下一个样本
                while self.valid_iter < self.num_iter_for_item:
                    if self.accuracy >= 0.875:
                        self.max_sample_rate_for_repeat_sample = max(self.max_sample_rate_for_repeat_sample / 2, min(low_thres, self.max_sample_rate))
                        print(f"EASY SAMPLE: {self.accuracy} {self.max_sample_rate_for_repeat_sample}")
                    elif self.accuracy <= 0.125:
                        self.max_sample_rate_for_repeat_sample = min(self.max_sample_rate_for_repeat_sample * 2, max(high_thres, self.max_sample_rate)) # 保证显存
                        print(f"HARD SAMPLE: {self.accuracy} {self.max_sample_rate_for_repeat_sample}")
                    for i, inputs in enumerate(batch_samples):
                        step += 1
                        do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                        # Since we perform prefetching, we need to manually set sync_gradients
                        self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                        if self.args.include_num_input_tokens_seen:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                input_tokens = inputs[main_input_name].numel()
                                input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                                self.state.num_input_tokens_seen += (
                                    self.accelerator.gather(input_tokens).sum().cpu().item()
                                )
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None

                        if step % args.gradient_accumulation_steps == 0:
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss = tr_loss + tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))

                        if do_sync_step:
                            # Since we perform prefetching, we need to manually set sync_gradients to True
                            self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            self.optimizer.step()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            if not self.accelerator.optimizer_step_was_skipped:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                            # 非有效探索不计入优化步骤
                            self.valid_iter = self.valid_iter + 1
                            # 当有效探索达到规定数量时迭代下一个样本
                            if self.valid_iter == self.num_iter_for_item:
                                self.state.global_step += 1
                                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                                self._maybe_log_save_evaluate(
                                    tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                                )
                        else:
                            self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                    # We also need to break out of the nested loop
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

