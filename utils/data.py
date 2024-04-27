import logging
import random
import os
import json
from datasets import Dataset
from torch.utils.data import DataLoader

from utils.collate import DataCollatorForGPLinkerDuEE

logger = logging.getLogger(__name__)


def duee_v1_process(example):
    events = []
    for e in example["event_list"]:
        # offset1 = len(e["trigger"]) - len(e["trigger"].lstrip())
        events.append(
            [
                [
                    e["event_type"],
                    "触发词",
                    e["trigger"]["text"],
                    str(e["trigger"]["offset"][0])
                    + ";"
                    + str(e["trigger"]["offset"][1]),
                ]
            ]
        )
        for a in e["arguments"]:
            # offset2 = len(a["argument"]) - len(a["argument"].lstrip())
            events[-1].append(
                [
                    e["event_type" ],
                    a["role"],
                    a["text"],
                    str(a["offset"][0])
                    + ";"
                    + str(a["offset"][1]),
                ]
            )
    del example["event_list"]
    return {"events": events}


def get_dataloader_and_dataset(
    args,
    tokenizer,
    labels2id,
    event_type_char2id,
    use_fp16=False,
    text_column_name="text",
    label_column_name="events",
):

    train_raw_dataset = Dataset.from_json(os.path.join(args.file_path, "FNDEE_train_12000_.json"))
    train_ds = train_raw_dataset.map(duee_v1_process)
    dev_raw_dataset = Dataset.from_json(os.path.join(args.file_path, "FNDEE_test_2500_.json"))
    dev_ds = dev_raw_dataset.map(duee_v1_process)
    if args.contrastive_method == 'role_description':
        role_label_description = Dataset.from_json(os.path.join(args.file_path, "FNDEE_role_description.json"))
    elif args.contrastive_method == 'event_description' or args.contrastive_method == "No":
        whole_event_description = Dataset.from_json(os.path.join(args.file_path, "FNDEE_event_description.json"))


    def tokenize_and_align_train_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        labels = []
        for b, events in enumerate(examples[label_column_name]):
            argu_labels = []
            head_labels = []
            tail_labels = []
            for event in events:
                for i1, (event_type1, rol1, word1, span1) in enumerate(event):
                    tp1 = labels2id.index((event_type1, rol1))
                    head1, tail1 = list(map(int, span1.split(";")))
                    tail1 = tail1 - 1
                    try:
                        h1 = tokenized_inputs.char_to_token(b, head1)
                        t1 = tokenized_inputs.char_to_token(b, tail1)
                    except Exception as e:
                        logger.info(f"{e} char_to_token error!")
                        continue
                    if h1 is None or t1 is None:
                        logger.info("find None!")
                        continue
                    # if tp1 not in argu_labels:
                    #     argu_labels[tp1] = [tp1]
                    argu_labels.append((int(tp1),h1, t1))

                    for i2, (event_type2, rol2, word2, span2) in enumerate(event):
                        if i2 > i1:
                            head2, tail2 = list(map(int, span2.split(";")))
                            tail2 = tail2 - 1
                            try:
                                h2 = tokenized_inputs.char_to_token(b, head2)
                                t2 = tokenized_inputs.char_to_token(b, tail2)
                            except Exception as e:
                                logger.info("char_to_token error!")
                                continue
                            if h2 is None or t2 is None:
                                logger.info("find None!")
                                continue
                            hl = (int(event_type_char2id.index(event_type2)),min(h1, h2), max(h1, h2))
                            tl = (int(event_type_char2id.index(event_type2)),min(t1, t2), max(t1, t2))
                            if hl not in head_labels:
                                head_labels.append(hl)
                            if tl not in tail_labels:
                                tail_labels.append(tl)

            # argu_labels = list(argu_labels.values())
            labels.append(
                {
                    "argu_labels": argu_labels if len(argu_labels)>0 else [(0,0,0)],
                    "head_labels": head_labels if len(head_labels)>0 else [(0,0,0)],
                    "tail_labels": tail_labels if len(tail_labels)>0 else [(0,0,0)]
                }
            )
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        return tokenized_inputs
    def prepare_role_description_features(examples):
        tokenized_examples = tokenizer(
            examples["description"],
            truncation=True,
            max_length=args.max_length,
            padding=True,
        )
        return tokenized_examples
    
    def prepare_event_description_features(examples):
        tokenized_inputs = tokenizer(
            examples["description"],
            max_length=args.max_length,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        all_role_label = []
        for b, role_label_list in enumerate(examples["role_offset"]):
            role_index_label = []
            for role_label in role_label_list:
                role_head,role_tail = role_label["offset"]
                role_tail = role_tail - 1
                try:
                    r_h1 = tokenized_inputs.char_to_token(b,role_head)
                    r_t1 = tokenized_inputs.char_to_token(b,role_tail)
                except Exception as e:
                    print(f"{e} char_to_token error!")
                    continue
                if r_h1 is None or r_t1 is None:
                    print("find None!")
                    continue
                role_index_label.append([r_h1, r_t1])
            all_role_label.append(role_index_label)
        
        tokenized_inputs["role_index_labels"] = all_role_label
        return tokenized_inputs
    train_dataset = train_ds.map(
        tokenize_and_align_train_labels,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"FNDEE-train_Datasets-{args.model_type}-{args.max_length}-{args.model_weights}",
    )
    if args.contrastive_method == 'role_description':
        desceiptions_dataset = role_label_description.map(
            prepare_role_description_features,
            batched=True,
            desc="Running tokenizer on role descriptions",
            remove_columns=role_label_description.column_names,
        )
    elif args.contrastive_method == 'event_description' or args.contrastive_method == "No":
        desceiptions_dataset = whole_event_description.map(
            prepare_event_description_features,
            batched=True,
            desc="Running tokenizer on event descriptions",
            remove_columns=whole_event_description.column_names,
        )
    dev_dataset = dev_ds.map(
        tokenize,
        batched=True,
        remove_columns=["id", "events","coref_arguments"],  # 保留text
        desc="Running tokenizer on dev dataset",
        new_fingerprint=f"FNDEE-dev_Datasets-{args.model_type}-{args.max_length}",
    )
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")
    if args.contrastive_method == 'role_description':
        data_collator = DataCollatorForGPLinkerDuEE(
            type_input_ids=desceiptions_dataset["input_ids"],
            type_attention_mask=desceiptions_dataset["attention_mask"],
            type_token_type_ids=desceiptions_dataset["token_type_ids"] if "token_type_ids" in desceiptions_dataset else None,
            tokenizer=tokenizer,
            pad_to_multiple_of=(8 if use_fp16 else None),
            num_labels=args.num_labels,
            event_type_num = args.event_type_num,
            contrastive_method = args.contrastive_method,
        )
    elif args.contrastive_method == 'event_description' or args.contrastive_method == "No":
        data_collator = DataCollatorForGPLinkerDuEE(
            type_input_ids=desceiptions_dataset["input_ids"],
            type_attention_mask=desceiptions_dataset["attention_mask"],
            role_index_labels = desceiptions_dataset["role_index_labels"],
            type_token_type_ids=desceiptions_dataset["token_type_ids"] if "token_type_ids" in desceiptions_dataset else None,
            tokenizer=tokenizer,
            pad_to_multiple_of=(8 if use_fp16 else None),
            num_labels=args.num_labels,
            event_type_num = args.event_type_num,
            contrastive_method = args.contrastive_method,
        )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    dev_dataset.raw_data = dev_ds

    return train_dataloader, dev_dataloader

