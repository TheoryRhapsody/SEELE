import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple

import torch
from fastcore.all import patch_to
from transformers.file_utils import PaddingStrategy, _is_torch_device
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


ignore_list = ["offset_mapping", "text"]


@patch_to(BatchEncoding)
def to(self, device):
    if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
        data = {}
        for k, v in self.data.items():
            if k in ignore_list:
                data[k] = v
            else:
                if isinstance(v, (tuple, list)) and isinstance(v[0], dict):
                    data[k] = [
                        {subk: subv.to(device) for subk, subv in vv.items()} for vv in v
                    ]
                elif isinstance(v, (tuple, list)) and isinstance(v[0], torch.Tensor):
                    data[k] = [vv.to(device) for vv in v]
                else:
                    # print(v)
                    data[k] = v.to(device=device)
        self.data = data
    else:
        logger.warning(
            f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported."
        )
    return self


@dataclass
class DataCollatorForGPLinkerDuEE:
    tokenizer: PreTrainedTokenizerBase
    type_input_ids: torch.Tensor = None
    type_attention_mask: torch.Tensor = None
    type_token_type_ids: torch.Tensor = None
    role_index_labels: torch.Tensor = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    event_type_num: Optional[int] = None
    contrastive_method: Optional[str] = None
    

    def __post_init__(self):
        self.type_input_ids = torch.tensor(self.type_input_ids)
        self.type_attention_mask = torch.tensor(self.type_attention_mask)
        if self.type_token_type_ids is not None:
            self.type_token_type_ids = torch.tensor(self.type_token_type_ids)

    def __call__(
        self, features: List[Dict[str, Union[Tuple[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        new_features = [
            {k: v for k, v in f.items() if k not in ["labels"] + ignore_list}
            for f in features
        ]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is None: 
            # for test
            if self.contrastive_method == 'role_description':
                batch['type_input_ids'] = self.type_input_ids
                batch['type_attention_mask'] = self.type_attention_mask
                if self.type_token_type_ids is not None:
                    batch['type_token_type_ids'] = self.type_token_type_ids
            elif self.contrastive_method == 'event_description' or self.contrastive_method == "No":
                batch['type_input_ids'] = self.type_input_ids
                batch['type_attention_mask'] = self.type_attention_mask
                if self.type_token_type_ids is not None:
                    batch['type_token_type_ids'] = self.type_token_type_ids
                max_label_num = max([len(each_label_list) for each_label_list in self.role_index_labels])
                batched_role_index_labels = torch.zeros(self.event_type_num,max_label_num,2,dtype=torch.long)
                for type_id, index_list in enumerate(self.role_index_labels):
                    for role_id, (role_head, role_tail) in enumerate(index_list):
                        batched_role_index_labels[type_id, role_id, :] = torch.tensor([role_head, role_tail], dtype=torch.long)
                batch['role_index_labels'] = batched_role_index_labels
             
            if "text" in features[0].keys():
                batch["text"] = [feature["text"] for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [
                    feature["offset_mapping"] for feature in features
                ]
            return batch
        

        if self.contrastive_method == 'role_description':
            batch['type_input_ids'] = self.type_input_ids
            batch['type_attention_mask'] = self.type_attention_mask
            if self.type_token_type_ids is not None:
                batch['type_token_type_ids'] = self.type_token_type_ids
        elif self.contrastive_method == 'event_description' or self.contrastive_method == "No":
            batch['type_input_ids'] = self.type_input_ids
            batch['type_attention_mask'] = self.type_attention_mask
            if self.type_token_type_ids is not None:
                batch['type_token_type_ids'] = self.type_token_type_ids
            max_label_num = max([len(each_label_list) for each_label_list in self.role_index_labels])
            batched_role_index_labels = torch.zeros(self.event_type_num,max_label_num,2,dtype=torch.long)
            for type_id, index_list in enumerate(self.role_index_labels):
                for role_id, (role_head, role_tail) in enumerate(index_list):
                    batched_role_index_labels[type_id, role_id, :] = torch.tensor([role_head, role_tail], dtype=torch.long)
            batch['role_index_labels'] = batched_role_index_labels

            

        bs = batch["input_ids"].size(0)

        all_args_label_list = []
        all_head_label_list = []
        for each_label_ in labels:
           all_args_label_list.extend(each_label_["argu_labels"])
           all_head_label_list.extend(each_label_["head_labels"])

        arg_role_statistic_dict = {}
        event_type_statistic_dict = {}
        for each_arg_label in all_args_label_list:
            if each_arg_label [0] in arg_role_statistic_dict:
                arg_role_statistic_dict[each_arg_label [0]] += 1
            else:
                arg_role_statistic_dict[each_arg_label [0]] = 1

        for each_head_label in all_head_label_list:
            if each_head_label [0] in event_type_statistic_dict:
                event_type_statistic_dict[each_head_label [0]] += 1
            else:
                event_type_statistic_dict[each_head_label [0]] = 1

        max_argu_num = max([v for k, v in arg_role_statistic_dict.items()])
        max_head_tail_num = max([v for k, v in event_type_statistic_dict.items()])

        # max_head_num = max([len(lb["head_labels"]) for lb in labels])
        # max_tail_num = max([len(lb["tail_labels"]) for lb in labels])
        # max_argu_num = max(
        #     [(len(lb) - 1) // 2 for label in labels for lb in label["argu_labels"]]
        # )
        batch_argu_labels = torch.zeros(
            bs, self.num_labels, max_argu_num, 2, dtype=torch.long
        )
        batch_head_labels = torch.zeros(bs, self.event_type_num, max_head_tail_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, self.event_type_num, max_head_tail_num, 2, dtype=torch.long)


        for b, lb in enumerate(labels):
            
            cur_argu_label_tuple = [tuple(each_arg) for each_arg in lb["argu_labels"]]
            cur_head_label_tuple = [tuple(each_head) for each_head in lb["head_labels"]]
            cur_tail_label_tuple = [tuple(each_tail) for each_tail in lb["tail_labels"]]

            cur_argu_label_set = set(cur_argu_label_tuple)
            cur_head_label_set = set(cur_head_label_tuple)
            cur_tail_label_set = set(cur_tail_label_tuple)
            arg_label_index = [0] * self.num_labels
            head_label_index = [0] * self.event_type_num
            tail_label_index = [0] * self.event_type_num

            for [event_role_id, h, t] in cur_argu_label_set:
                if arg_label_index[event_role_id] < max_argu_num:
                    batch_argu_labels[b, event_role_id,  arg_label_index[event_role_id], :] = torch.tensor([h, t], dtype=torch.long)
                    arg_label_index[event_role_id] += 1

            for [event_type_id, h1, h2] in cur_head_label_set:
                if head_label_index[event_type_id] < max_head_tail_num:
                    batch_head_labels[b, event_type_id, head_label_index[event_type_id], :] = torch.tensor([h1, h2], dtype=torch.long)
                    head_label_index[event_type_id] += 1

            for [event_type_id, t1, t2] in cur_tail_label_set:
                if tail_label_index[event_type_id] < max_head_tail_num:
                    batch_tail_labels[b, event_type_id, tail_label_index[event_type_id], :] = torch.tensor([t1,t2], dtype=torch.long)
                    tail_label_index[event_type_id] += 1


        batch["labels"] = [
            batch_argu_labels,
            batch_head_labels,
            batch_tail_labels,
        ]
        return batch
