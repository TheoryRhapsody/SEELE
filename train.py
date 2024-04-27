import json
import logging
import math
import os
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pprint import pformat

import datasets
import torch
import transformers
import time
from accelerate import Accelerator
from fastcore.all import *
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler, set_seed
from transformers import BertTokenizerFast

from models import AutoModelGPLinker4EE
from utils.args import parse_args
from utils.data import get_dataloader_and_dataset
from utils.postprocess import DedupList, isin, postprocess_gplinker
from utils.utils import get_writer, try_remove_old_ckpt, write_json
from utils.FNDEE_Metrics import FNDEE_judge

logger = logging.getLogger(__name__)
my_args = parse_args()
my_device = torch.device(my_args.my_device)


@torch.no_grad()
def evaluate(
    args,
    model,
    dev_dataloader,
    accelerator,
    global_steps=0,
    threshold=0,
    write_predictions=True,
    current_epcoh = None,
):
    model.eval()
    all_predictions = []
    for batch in tqdm(
        dev_dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Evaluating: ",
        leave=False,
    ):
        offset_mappings = batch.pop("offset_mapping")
        texts = batch.pop("text")
        batch_s = batch.to(my_device)
        if args.contrastive_method == 'role_description':
            outputs = model(
                    input_ids = batch_s['input_ids'],
                    attention_mask = batch_s['attention_mask'],
                    type_inputs_ids = batch_s['type_input_ids'],
                    type_attention_mask = batch_s['type_attention_mask'],
                    current_epoch_id = current_epcoh
                    # labels = batch_s['labels']
                    )[0]
        elif args.contrastive_method == 'event_description'or args.contrastive_method == 'No':
            outputs = model(
                input_ids = batch_s['input_ids'],
                attention_mask = batch_s['attention_mask'],
                type_inputs_ids = batch_s['type_input_ids'],
                type_attention_mask = batch_s['type_attention_mask'],
                role_index_labels = batch_s['role_index_labels'],
                current_epoch_id = current_epcoh
                # labels = batch_s['labels']
                )[0]

        outputs_gathered = postprocess_gplinker(
            args,
            accelerator.gather(outputs),
            offset_mappings,
            texts,
            trigger=False,
            threshold=threshold,
        )
        all_predictions.extend(outputs_gathered)


    if write_predictions:
        pred_dir = os.path.join(args.output_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)
        time_info = time.localtime()
        nowtime  = time.strftime("%Y-%m-%d %H:%M:%S",time_info)
        pred_file = os.path.join(pred_dir, f"FNDEE_{global_steps}_step_{nowtime}_predictions.json")
        f = open(pred_file, "w", encoding="utf-8")
    
    all_predict_result_set = []
    id_count = 0
    for pred_events, events, texts in zip(
        all_predictions,
        dev_dataloader.dataset.raw_data["events"],
        dev_dataloader.dataset.raw_data["text"],
    ):
        id_count += 1

        if write_predictions:
            event_list = DedupList()
            for event in pred_events:
                trg_flag = 0
                final_event = {
                    "event_type": event[0][0],
                    "arguments": DedupList()}
                
                for each_arg in event :
                    if each_arg[1] == "触发词":
                        trg_flag = 1
                        final_event["trigger"]= {
                            "text": each_arg[2],
                            "offset": list(map(int, each_arg[3].split(";"))) 
                        }
                if trg_flag == 0:
                    # print("存在无触发词的预测结果，进行过滤")
                    continue
                for argu in event:
                    if argu[1] != "触发词":
                        final_event["arguments"].append(
                            {"role": argu[1], "text": argu[2], "offset": list(map(int, argu[3].split(";")))}
                        )
                event_list = [
                    event for event in event_list if not isin(event, final_event)
                ]
                if not any([isin(final_event, event) for event in event_list]):
                    event_list.append(final_event)

            # l = json.dumps(
            #     {"text": texts, "event_list": event_list}, ensure_ascii=False
            # )
            l = {
                "id": "{}".format(id_count),
                "event_list": event_list
            }

            all_predict_result_set.append(l)
            # f.write(l + "\n")


    if write_predictions:
        json.dump(all_predict_result_set, f,ensure_ascii=False)
        f.close()
    
        trg_P, trg_R, trg_F1, arg_P, arg_R, arg_F1 = FNDEE_judge(args.ground_truth_test_path, pred_file)



    final_F1 = (trg_F1 + arg_F1)/2
    


    model.train()

    return {
        "trg_f1": trg_F1,
        "trg_Pr": trg_P,
        "trg_Rc": trg_R,
        "arg_f1": arg_F1,
        "arg_Pr": arg_P,
        "arg_Rc": arg_R,
        "Final_f1": final_F1
    }


def main():
    args = parse_args()
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "{}_run.log".format(time.strftime("%m-%d %X", time.localtime()))),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    labels = []
    labels_to_eventtype = []
    event_type_labels = []
    max_role_num_within_one_event = 0
    with open(os.path.join(args.file_path, "FNDEE_schema.json"), "r", encoding="utf-8") as f:
        schema_ = json.load(f)
        for idx1,l in enumerate(schema_) :
            # l = json.loads(l)
            current_role_num =0
            t = l["event_type"]
            event_type_labels.append(t)
            for idx2,r in enumerate(["触发词"] + [s["role"] for s in l["role_list"]]) :
                labels.append((t, r))
                labels_to_eventtype.append(torch.tensor([idx1,idx2]).unsqueeze_(0))
                current_role_num += 1
            if current_role_num >max_role_num_within_one_event:
                max_role_num_within_one_event = current_role_num
    args.labels_to_event_type = torch.cat(labels_to_eventtype)
    args.labels = labels
    args.event_type_labels = event_type_labels
    args.num_labels = len(labels)
    args.max_role_num_within_one_event = max_role_num_within_one_event

    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.pretrained_model_name_or_path
    )
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # model = get_auto_model(args.model_type).from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     num_labels=args.num_labels,
    #     cache_dir=args.model_cache_dir,
    #     use_efficient=args.use_efficient,
    # ).to(my_device)
    my_config = {
        "bert_path": args.pretrained_model_name_or_path,
        "num_labels": args.num_labels,
        "labels_to_event_type": args.labels_to_event_type,
        "event_type_num": args.event_type_num,
        "cache_dir": args.model_cache_dir,
        "use_efficient": args.use_efficient,
        "contrastive_method": args.contrastive_method,
        "contrastive_level": args.contrastive_level,
        "dropout_rate": args.dropout_rate,
        "description_loss_weight": args.description_loss_weight,
        "hidden_size": 768,
        "max_role_num": args.max_role_num_within_one_event
    }
    model = AutoModelGPLinker4EE(my_config).to(my_device)

    (train_dataloader, dev_dataloader) = get_dataloader_and_dataset(
        args,
        tokenizer,
        labels,
        event_type_labels,
        use_fp16=accelerator.use_fp16,
        text_column_name="text",
        label_column_name="events",
    )

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in list(model.named_parameters()):
        space = name.split('.')
        if 'encoder' in space:
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01, 'lr': 2e-5},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': 2e-5},

            # others
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': 2e-5},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': 2e-5},
        ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader
    )
    model.to(my_device)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    args.num_warmup_steps = (
        math.ceil(args.max_train_steps * args.num_warmup_steps_or_radios)
        if isinstance(args.num_warmup_steps_or_radios, float)
        else args.num_warmup_steps_or_radios
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    args.total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        leave=False,
        disable=not accelerator.is_local_main_process,
        desc="Training: ",
    )
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_arg_f1 = 0.0
    max_final_f1 = 0.0
    writer = get_writer(args)
    model.train()

    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    args_copy = copy.deepcopy(args)
    args_dict = vars(args_copy)
    del args_dict["labels_to_event_type"]
    write_json(args_dict, os.path.join(args.output_dir, "model_args.json"))
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batchs = batch.to(my_device)
            if args.contrastive_method == 'role_description':
                outputs = model(
                    input_ids = batchs['input_ids'],
                    attention_mask = batchs['attention_mask'],
                    type_inputs_ids = batchs['type_input_ids'],
                    type_attention_mask = batchs['type_attention_mask'],
                    labels = batchs['labels'],
                    current_epoch_id = epoch,
                    )
            elif args.contrastive_method == 'event_description'or args.contrastive_method == 'No':
                outputs = model(
                    input_ids = batchs['input_ids'],
                    attention_mask = batchs['attention_mask'],
                    type_inputs_ids = batchs['type_input_ids'],
                    role_index_labels = batchs['role_index_labels'],
                    type_attention_mask = batchs['type_attention_mask'],
                    labels = batchs['labels'],
                    current_epoch_id = epoch,
                    )
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                if epoch < 1 or args.contrastive_method == 'No':
                    loss = outputs[1]
                else:
                    loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            accelerator.backward(loss)

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1

                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    writer.add_scalar(
                        "lr", lr_scheduler.get_last_lr()[-1], global_steps
                    )
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.8f}  loss: {:.8f}".format(
                            global_steps,
                            lr_scheduler.get_last_lr()[-1],
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    accelerator.print(
                        "global_steps {} - lr: {:.8f}  loss: {:.8f}".format(
                            global_steps,
                            lr_scheduler.get_last_lr()[-1],
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    logging_loss = tr_loss
        accelerator.wait_for_everyone()

        if (
            epoch >= 5
        ) or epoch == args.num_train_epochs:
            logger.info(
                f"********** Evaluate Step {global_steps} **********")
            accelerator.print("##--------------------- Dev")
            logger.info("##--------------------- Dev")
            dev_metric = evaluate(
                args, model, dev_dataloader, accelerator, global_steps, 0, True, epoch
            )
            accelerator.print("-" * 80)
            logger.info("-" * 80)
            for k, v in dev_metric.items():
                accelerator.print(f"{k} = {v}")
                logger.info(f"{k} = {v}")
                writer.add_scalar(
                    f"dev/{k}",
                    v,
                    global_steps,
                )
            accelerator.print("-" * 80)
            logger.info("-" * 80)
            accelerator.print("**--------------------- Dev End")
            logger.info("**--------------------- Dev End")

            arg_f1, final_f1 = dev_metric["arg_f1"], dev_metric["Final_f1"]


            if final_f1 >= max_final_f1:
                max_final_f1 = final_f1
                savefile = Path(args.output_dir) / "Final_f1_results_{}_{}.txt".format(args.contrastive_method,args.description_loss_weight)
                savefile.write_text(
                    pformat(dev_metric), encoding="utf-8")
                
                curtime = time.localtime()
                now_time  = time.strftime("%Y-%m-%d %H:%M:%S",curtime)

                a_output_dir = os.path.join(
                    args.output_dir,
                    "contrastive",
                    "ckpt",
                    f"step-{global_steps}-f1-{final_f1}",
                    f"{now_time}",
                )
                os.makedirs(a_output_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                tokenizer.save_pretrained(a_output_dir)
                torch.save(model.state_dict(), a_output_dir+"/pytorch_model.bin")
                # accelerator.unwrap_model(model).save_pretrained(
                #     a_output_dir, save_function=accelerator.save
                # )
                # try_remove_old_ckpt(
                #     os.path.join(args.output_dir, "Final_F1"), topk=args.topk
                # )
                logger.info('>> saved: {}'.format(a_output_dir))

            logger.info("*************************************")

        if epoch >= args.num_train_epochs:
            return
        


if __name__ == "__main__":
    main()
