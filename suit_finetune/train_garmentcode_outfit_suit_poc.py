# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import shutil

import numpy as np
import torch
import random

import transformers
import tokenizers

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
import deepspeed
from functools import partial
from easydict import EasyDict as edict
from llava.garmentcode_utils import change_prompt, change_answer

from PIL import Image
import time
from llava.lisa_utils import AverageMeter, ProgressMeter, dict_to_cuda, Summary
from torch.utils.tensorboard import SummaryWriter
import tqdm
import shutil
from llava.json_fixer import repair_json
import bisect
import os
import glob

local_rank = None
os.environ["MASTER_PORT"] = "23480"

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_eval: str = field(default=None,
                                metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    suit_poc_mode: bool = False
    suit_init_checkpoint: Optional[str] = None
    suit_exp_name: str = "suit_poc_lora"
    suit_epochs: int = 1
    suit_steps_per_epoch: int = 220
    suit_batch_size: int = 1
    suit_grad_accumulation_steps: int = 8
    suit_learning_rate: float = 2e-5


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def find_all_linear_names(model, lora_target_modules=['q_proj', 'v_proj']):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        'mm_projector', 'vision_tower', 'vision_resampler', 'float_layer'
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)

    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def convert_rgba_to_rgb_with_white_bg(image_path):
    """Convert an RGBA image to an RGB image with a white background."""
    if not os.path.exists(image_path):
        image_path = image_path[:-4] + '.gif'

    # Open the image
    image = Image.open(image_path)

    # Check if the image is in RGBA mode
    if image.mode == 'RGBA':
        # Create a white background image of the same size as the original
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        
        # Paste the RGBA image onto the white background, using the alpha channel as the mask
        white_background.paste(image, mask=image.split()[3])  # Split to get the alpha channel
        image = white_background
    else:
        # If not RGBA, save it directly as an RGB image
        image = image.convert("RGB")
    
    return image



def autocrop(pil_img, matrix_HW_pct_range=[0.4, 0.95]):
    """
    random crop from an input image
    """
    img_focus = pil_img
    x_focus, y_focus = img_focus.size


    matrix_x_pct = random.uniform(matrix_HW_pct_range[0], matrix_HW_pct_range[1])
    matrix_x = round(matrix_x_pct*x_focus)
    matrix_y_pct = random.uniform(matrix_HW_pct_range[0], matrix_HW_pct_range[1])
    matrix_y = round(matrix_y_pct*y_focus)

    if matrix_x < 10 or matrix_y < 10:
        return pil_img

    x1 = random.randrange(0, x_focus - matrix_x)
    y1 = random.randrange(0, y_focus - matrix_y)
    cropped_img = img_focus.crop((x1, y1, x1 + matrix_x, y1 + matrix_y))
    
    return cropped_img


def generate_all_float_labels(all_floats, indices):
    if isinstance(all_floats[0], list) or len(indices) > 1:
        if not isinstance(all_floats[0], list):
            all_floats = [all_floats]
        if len(indices) == 2 and len(all_floats) == 1:
            # debug
            assert len(all_floats[0]) == len(indices[0]) + len(indices[1]), (len(all_floats[0]), len(indices[0]), len(indices[1]))
            all_floats_new = [
                [all_floats[0][i] for i in range(len(indices[0]))],
                [all_floats[0][i] for i in range(len(indices[0]), len(indices[0]) + len(indices[1]) )]
            ]
            all_floats = all_floats_new
        all_floats_padded = []
        all_floats_weight_padded = []
        for i, item in enumerate(all_floats):
            indices_i = indices[i]
            assert len(item) == len(indices_i), (len(item), len(indices_i), i, all_floats)
            all_floats_padded_tmp = np.zeros(76)
            all_floats_padded_tmp[indices_i] = np.array(item)
            all_floats_padded.append(all_floats_padded_tmp)

            all_floats_padded_flag = np.zeros(76)
            all_floats_padded_flag[indices_i] = 1
            all_floats_weight_padded.append(all_floats_padded_flag)
            
        all_floats = np.array(all_floats_padded).reshape(-1)
        all_floats_weight = np.array(all_floats_weight_padded).reshape(-1)
    
    else:
        all_floats_padded = np.zeros(76)
        indices_i = indices[0]
        assert len(indices_i) == len(all_floats), (len(all_floats), len(indices_i),  all_floats)
        all_floats_padded[indices_i] = np.array(all_floats)
        all_floats_padded_flag = np.zeros(76)
        all_floats_padded_flag[indices_i] = 1

        all_floats = np.array(all_floats_padded).reshape(-1)
        all_floats_weight = np.array(all_floats_padded_flag).reshape(-1)
    
    return all_floats, all_floats_weight


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 max_len=-1,
                 random_imgaug=False,
                 has_sewing_pattern=False):
        super(LazySupervisedDataset, self).__init__()
        # print('LazySupervisedDataset', LazySupervisedDataset, data_path)
        list_data_dict = json.load(open(data_path, "r"))
        self.has_sewing_pattern = has_sewing_pattern

        if max_len > 0:
            list_data_dict = list_data_dict[:max_len]
        # print('list_data_dict', list_data_dict)

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.random_imgaug = random_imgaug

        if not 'sample_prob' in list_data_dict[0]:
            self.probs = np.ones(len(list_data_dict))

        else:
            self.probs = [
                item['sample_prob'] for item in list_data_dict
            ]
            for i in range(len(self.probs)):
                if self.probs[i] > 0.15 and self.probs[i] < 0.35:
                    self.probs[i] = 0.05
                
                elif self.probs[i] < 0.15:
                    self.probs[i] = 0.02

            self.probs = np.array(self.probs)

        self.probs = self.probs / self.probs.sum()
        self.cumsum = np.cumsum(self.probs)

        with open('docs/all_float_paths.json', 'r') as f:
            all_float_paths = json.load(f)

        self.all_float_paths_dict = {
            item: idx for idx, item in enumerate(all_float_paths)
        }

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = bisect.bisect_left(self.cumsum, random.random())
        sources = copy.deepcopy(self.list_data_dict[i])

        ################ prompt augmentation ################
        prompt = sources["conversations"][0]["value"]
        prompt = change_prompt(prompt[:])
        sources["conversations"][0]["value"] = prompt

        if self.has_sewing_pattern:
            answer = sources["conversations"][1]["value"]
            answer, indices = change_answer(answer[:], self.all_float_paths_dict, True)
            sources["conversations"][1]["value"] = answer

        ################ add trivial image if no image ################
        rand_flag = True
        if '<image>' not in prompt:
            sources["conversations"][0]["value"] = '<image>\n' + prompt
            self.list_data_dict[i]['image'] = "docs/images/black_img.jpg"
            sources["image"] = "docs/images/black_img.jpg"
            rand_flag = False
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        has_floats = False
        if 'all_floats' in sources[0]:
            has_floats = True
            all_floats, all_floats_weight = generate_all_float_labels(sources[0]['all_floats'], indices)
            
        if 'image' in sources[0]:
            image_file = sources[0]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            image = convert_rgba_to_rgb_with_white_bg(os.path.join(image_folder, image_file))
            if self.random_imgaug and rand_flag:
                image = autocrop(image)
            elif rand_flag:
                image = autocrop(image, matrix_HW_pct_range=[0.8, 0.96])

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        # print('image', image.shape) # image torch.Size([3, 336, 336])
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_path'] = os.path.join(image_folder, image_file)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_path'] = ''
        
        if has_floats:
            data_dict['all_floats'] = torch.tensor(all_floats).float().reshape(-1)
            data_dict['float_weight'] = torch.tensor(all_floats_weight).float().reshape(-1)
        else:
            data_dict['all_floats'] = torch.zeros(0)
            data_dict['float_weight'] = torch.zeros(0)

        return data_dict




class LazySupervisedDatasetCmb(Dataset):
    def __init__(self, data_path_list: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 max_len=-1):
        super(LazySupervisedDatasetCmb, self).__init__()
        
        prob_dict = {
            "sewing_pattern_img": 0.1,
            "sewing_pattern_text": 0.1,
            "sewing_pattern_imgtext": 0.2,
            "wild_img": 0.1,
            "sewing_pattern_editing": 0.15,
            "ift": 0.35
        }

        self.datasets = []
        self.ratios = []
        for k, v in data_path_list.items():
            prob = prob_dict[k] / len(v)
            has_sewing_pattern = True if 'sewing_pattern' in k else False
            random_imgaug = True if k != 'ift' else False
            self.datasets.extend([
                LazySupervisedDataset(
                    data_path=v_i,
                    tokenizer=tokenizer, data_args=data_args, max_len=max_len,
                    random_imgaug=random_imgaug,
                    has_sewing_pattern=has_sewing_pattern) for v_i in v
            ])

            self.ratios.extend([prob] * len(v))

        self.dataset_num = len(self.datasets)
        assert len(self.ratios) == self.dataset_num
        assert np.abs(np.sum(self.ratios) - 1) < 1e-3

        self.ratio_cumsum = np.cumsum(self.ratios)
        self.lengths = [len(self.datasets[i]) for i in range(self.dataset_num)]
        print('self.lengths', self.lengths)

    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        myrandom = random.Random(i * 1000 + random.randint(0, 1000))
        p = myrandom.random()
        for j in range(self.dataset_num):
            if p < self.ratio_cumsum[j]:
                idx = myrandom.randint(0, self.lengths[j] - 1)
                return self.datasets[j][idx]




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        image_paths = [instance['image_path'] for instance in instances]
        all_floats = [instance['all_floats'] for instance in instances]
        float_weight = [instance['float_weight'] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            image_paths=image_paths,
            float_labels=all_floats,
            float_weight=float_weight
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, data_path_list) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDatasetCmb(tokenizer=tokenizer,
                                data_path_list=data_path_list,
                                data_args=data_args)
    
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path_eval,
                                data_args=data_args,
                                max_len=10)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return train_dataset, eval_dataset, data_collator



def validate_epoch(val_loader, model_engine, tokenizer, epoch, writer, args):
    return



def train_epoch(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    image_save_dir = os.path.join(args.log_dir, "training_images")
    os.makedirs(image_save_dir, exist_ok=True)
    
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    hmr_losses = AverageMeter("HMRLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            hmr_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    print('Start training ...')
    for global_step in range(args.steps_per_epoch):

        global_step_actual = epoch * args.steps_per_epoch + global_step
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            
            # continue

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            assert args.precision == "bf16"
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict.pop('image_paths')
            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            hmr_loss = output_dict["hmr_loss"]
            
            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            hmr_losses.update(hmr_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                hmr_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step_actual)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step_actual)
                writer.add_scalar(
                    "train/hmr_loss", hmr_losses.avg, global_step_actual
                )
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step_actual
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step_actual
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            hmr_losses.reset()
        
        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step_actual)
                
    return train_iter



def translate_args(model_args, data_args, training_args):
    args = edict(
        local_rank=local_rank,
        version=None,
        vis_save_path="./vis_output",
        precision="bf16",
        image_size=None,
        model_max_length=training_args.model_max_length,
        lora_r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        lora_target_modules=None,
        vision_tower=model_args.vision_tower,
        load_in_8bit=False,
        load_in_4bit=False,
        dataset=None,
        sample_rates=None,
        log_base_dir='./runs',
        exp_name=training_args.suit_exp_name if training_args.suit_poc_mode else "chatgarment_train_scratch",
        epochs=training_args.suit_epochs if training_args.suit_poc_mode else 40,
        steps_per_epoch=training_args.suit_steps_per_epoch if training_args.suit_poc_mode else 500,
        batch_size=training_args.suit_batch_size if training_args.suit_poc_mode else 4,
        grad_accumulation_steps=training_args.suit_grad_accumulation_steps if training_args.suit_poc_mode else 8,
        val_batch_size=1,
        workers=4 if training_args.suit_poc_mode else 8,
        lr=training_args.suit_learning_rate if training_args.suit_poc_mode else 1e-4,
        ce_loss_weight=1.0,
        no_eval=False,
        eval_only=False,
        vision_pretrained=None,
        resume="",
        start_epoch=0,
        print_freq=1,
        gradient_checkpointing=training_args.gradient_checkpointing,
        beta1=0.9,
        beta2=0.999,

        use_mm_start_end=False
    )   

    return args


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    args = translate_args(model_args, data_args, training_args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    bnb_model_from_pretrained_args = {}
    # writer = None
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    assert training_args.bits not in [4, 8]
    assert model_args.vision_tower is not None
    assert 'mpt' not in model_args.model_name_or_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")

    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    
    print('num_added_tokens', num_added_tokens, args.seg_token_idx)

    model = GarmentGPTFloat50ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        seg_token_idx=args.seg_token_idx,
        **bnb_model_from_pretrained_args
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False ####################### ?
    assert not model_args.freeze_backbone

    assert training_args.gradient_checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable() ###################### ？
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # tokenizer.add_tokens('<FLOAT>', special_tokens=True)
    assert model_args.version == "v1"
    if model_args.version in conversation_lib.conv_templates: # conv_vicuna_v1
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    lora_target_modules = find_all_linear_names(model)
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))
    model.print_trainable_parameters()

    if not training_args.suit_poc_mode:
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "float_layer"]
                ]
            ):
                rank0_print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    ######################### ？

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    assert not model_args.tune_mm_mlp_adapter
    
    assert not training_args.freeze_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer) ############### ?

    if training_args.suit_poc_mode:
        if not training_args.suit_init_checkpoint:
            raise ValueError("--suit_init_checkpoint is required in suit POC mode")
        rank0_print(
            "Loading official ChatGarment task checkpoint:",
            training_args.suit_init_checkpoint,
        )
        state_dict = torch.load(training_args.suit_init_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_" in name
        # resize_token_embeddings()/PEFT wrapping can replace the module that
        # received the earlier hook.  Re-register it after the final freeze so
        # re-entrant gradient checkpointing has a differentiable input.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    data_path_list = {   
        "sewing_pattern_img": [
            'checkpoints/training_data_zip/training/data_restpose_img_v1.json',
            'checkpoints/training_data_zip/training/data_img_v2.json',
            'checkpoints/training_data_zip/training/data_img_v4.json',
        ],
        "sewing_pattern_text": [
            'checkpoints/training_data_zip/training/data_detailtext_v2.json',
            'checkpoints/training_data_zip/training/data_detailtext_singlegarment_v2.json',
            'checkpoints/training_data_zip/training/data_detailtext_v4.json',
        ],
        "sewing_pattern_imgtext": [
            'checkpoints/training_data_zip/training/data_detailtextimg_v2.json',
            'checkpoints/training_data_zip/training/data_detailtextimg_v3.json',
            'checkpoints/training_data_zip/training/data_detailtextimg_v4.json',
            'checkpoints/training_data_zip/training/data_detailtextimg_singlegarment_v2.json'
        ],
        "wild_img": [
            'checkpoints/training_data_zip/shhq/sshq_llavaformat_list_detaildescp.json',
        ],
        "sewing_pattern_editing": [
            'checkpoints/training_data_zip/editing/random_newgarments_textsewing.json'
        ],
        "ift": [
            'playground/data/llava_v1_5_mix665k.json',
        ]
    }

    if training_args.suit_poc_mode:
        train_dataset = LazySupervisedDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            random_imgaug=True,
            has_sewing_pattern=False,
        )
        val_dataset = LazySupervisedDataset(
            data_path=data_args.data_path_eval,
            tokenizer=tokenizer,
            data_args=data_args,
            max_len=10,
        )
        collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        train_dataset, val_dataset, collate_fn = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            data_path_list=data_path_list,
        )
    
    rank0_print('train_dataset', len(train_dataset))
    # assert False

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": (
                    min(100, max(1, args.epochs * args.steps_per_epoch // 10))
                    if training_args.suit_poc_mode
                    else 100
                ),
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, _, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=collate_fn,
        config=ds_config,
    )

    resume = os.path.join(args.log_dir, "ckpt_model")
    if args.resume:
        args.resume = resume
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    
    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train_epoch(
            train_loader,
            model_engine, 
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if True:
            best_score = 0.0
            cur_ciou = 0.0
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            if training_args.suit_poc_mode:
                if args.local_rank == 0:
                    lora_state = {
                        name: parameter.detach().cpu()
                        for name, parameter in model_engine.module.state_dict().items()
                        if "lora_" in name
                    }
                    output_path = os.path.join(
                        args.log_dir, "suit_lora_state.bin"
                    )
                    torch.save(lora_state, output_path)
                    rank0_print("Saved suit LoRA state:", output_path)
                torch.distributed.barrier()
            else:
                model_engine.save_checkpoint(save_dir)


if __name__ == "__main__":
    train()
