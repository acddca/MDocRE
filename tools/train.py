# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2022/11/16 23:20
# @File: train.py
# @Email: wangjl.nju.2020@gmail.com.
import sys
sys.path.append("../")

import numpy as np
import wandb
import os
import json
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers import HfArgumentParser
from datetime import datetime
from arguments import DataTrainingArguments, ModelArguments, CustomizedTrainingArguments
from utils import classification_report
from src.data.docred_dataset import DocREDDataset, DocREDDatasetForBert2Phase
from src.data.mdocred_dataset import MDocREDDataset
from src.data.mdocred_dataset_lsr import LSRDataset
from src.models.trainer import Trainer
from src.models.bert_trainer import BertTrainer
from src.models.atlop_trainer import ATLOPTrainer
from src.models.baseline_cnn_trainer import CnnTrainer
from src.models.baseline_bilstm_trainer import BilstmTrainer
from src.models.bert2phase_trainer import Bert2PhaseTrainer
from src.models.clipbert_trainer import ClipBertTrainer
from src.models.maebert_trainer import MaeBertTrainer
from src.models.maebert_base_trainer import MaeBertBaseTrainer
from src.models.mdocre_trainer import MDocRETrainer
from src.models.mdocre2stream_trainer import MDocRE2StreamTrainer
from src.models.mdocrecross_trainer import MDocRECrossTrainer
from src.models.mdocre_coattn_trainer import MDocRECoAttentionTrainer
from src.models.textual_prefix_trainer_klx import TextualPrefixTrainerKLX
from src.models.lsr_trainer import LSRBertTrainer
from src.models.vitbert_trainer import ViTBertTrainer
from src.models.textual_prefix_trainer import TextualPrefixTrainer
from src.utils.evaluate_utils import to_official, official_evaluate
from src.utils.train_utils import (
    set_seed, collate_fn, clip_bert_collate_fn, mae_bert_collate_fn, mdocre_collate_fn,
    mdocre_cross_collate_fn, textual_prefix_collate_fn, lsr_collate_fn, EarlyStopping
)


def get_trainer(model_args, train_args) -> Trainer:
    model_type = model_args.model_type
    if model_type == "enhanced_bert":
        return BertTrainer(train_args, model_args)
    elif model_type == "atlop":
        return ATLOPTrainer(train_args, model_args)
    elif model_type == "bert2phase":
        return Bert2PhaseTrainer(train_args, model_args)
    elif model_type == "clip_bert":
        return ClipBertTrainer(train_args, model_args)
    elif model_type == "mae_bert":
        return MaeBertTrainer(train_args, model_args)
    elif model_type == "mae_bert_base":
        return MaeBertBaseTrainer(train_args, model_args)
    elif model_type == "mdocre" or model_type == "mdocre_unimodal":
        return MDocRETrainer(train_args, model_args)
    elif model_type == "mdocre2stream":
        return MDocRE2StreamTrainer(train_args, model_args)
    elif model_type == "mdocre_cross":
        return MDocRECrossTrainer(train_args, model_args)
    elif model_type == "mdocre_coattn":
        return MDocRECoAttentionTrainer(train_args, model_args)
    elif model_type == "vit_bert":
        return ViTBertTrainer(train_args, model_args)
    elif model_type == "textual_prefix":
        return TextualPrefixTrainer(train_args, model_args)
    elif model_type == "textual_prefix_klx":
        return TextualPrefixTrainerKLX(train_args, model_args)
    elif model_type == "lsr_unimodal":
        return LSRBertTrainer(train_args, model_args)
    elif model_type == "cnn":
        return CnnTrainer(train_args, model_args)
    elif model_type == "bilstm":
        return BilstmTrainer(train_args, model_args)
    raise RuntimeError(f"Unsupported model type: {model_type}")


def get_dataset(split, data_args: DataTrainingArguments, model_args: ModelArguments, device_type="cuda"):
    assert split in ["train", "eval", "test"]

    dataset_name = data_args.dataset_name
    if dataset_name == "docred":
        model_type = model_args.model_type
        if model_type == "atlop" or model_type == "enhanced_bert":
            return DocREDDataset(data_args, split, data_args.max_len, True)
        elif model_type == "bert2phase":
            first_phase = model_args.first_phase
            if split == "train":
                if first_phase:
                    return DocREDDataset(data_args, split, data_args.max_len, True)

                neg_pos_ratio = data_args.neg_pos_ratio
                only_pos_sample = data_args.only_pos_sample
                return DocREDDatasetForBert2Phase(
                    data_args, split, data_args.max_len, True, only_pos_sample, neg_pos_ratio
                )
            else:
                return DocREDDataset(data_args, split, data_args.max_len, True)
    elif dataset_name == "mdocred":
        model_type = model_args.model_type
        if (
                model_type == "atlop"
                or model_type == "enhanced_bert"
                or model_type == "cnn"
                or model_type == "bilstm"
        ):
            return DocREDDataset(data_args, split, data_args.max_len, True)
        elif model_type == "lsr_unimodal":
            return LSRDataset(data_args, split, data_args.max_len, data_args.char_limit, device_type)
        elif (
                model_type == "clip_bert"
                or model_type == "mae_bert"
                or model_type == "mae_bert_base"
                or model_type == "mdocre"
                or model_type == "mdocre2stream"
                or model_type == "mdocre_unimodal"
                or model_type == "mdocre_cross"
                or model_type == "mdocre_coattn"
                or model_type == "vit_bert"
                or model_type == "textual_prefix"
                or model_type == "textual_prefix_klx"
        ):
            num_frames = model_args.num_frames
            image_size = model_args.image_size
            use_mosaic = model_args.use_mosaic
            return MDocREDDataset(
                data_args, split, data_args.max_len, num_frames, image_size=image_size, use_mosaic=use_mosaic
            )

    raise RuntimeError(f"Unsupported dataset name: {dataset_name}")


def train(train_args: CustomizedTrainingArguments, trainer: Trainer, train_dataloader, eval_dataloader, data_args):
    best_score = -1
    num_steps = 0
    num_train_epochs = train_args.num_train_epochs
    early_stopping = EarlyStopping(train_args.early_stopping_patience)
    num_pretrain_epoch = train_args.num_pretrain_epoch
    for epoch in range(int(num_train_epochs)):
        epoch_loss = []
        p_bar = tqdm(enumerate(train_dataloader))
        num_batch = len(train_dataloader)
        for step, batch in p_bar:
            p_bar.set_description(f"Training Example: {step + 1} / {num_batch}")
            loss_val = trainer.train_step(batch, epoch)
            epoch_loss.append(loss_val)
            num_steps += 1

            wandb.log({"loss": loss_val}, step=num_steps)

            if (
                    (step + 1) == len(train_dataloader) - 1
                    or (train_args.eval_steps > 0 and num_steps % train_args.eval_steps == 0)
            ):
                if 0 <= epoch < num_pretrain_epoch:
                    dev_score, dev_output, dev_preds = evaluate_pretrain(trainer, eval_dataloader, epoch, tag="dev")
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                else:
                    dev_score, dev_output, dev_preds = evaluate(
                        trainer, eval_dataloader, data_args.rel_info_path, epoch, tag="dev"
                    )
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)

                    if dev_score > best_score:
                        best_score = dev_score
                        wandb.log({"best_dev_score": best_score}, step=num_steps)
                        if not os.path.exists(train_args.output_dir):
                            os.makedirs(train_args.output_dir)

                        with open(os.path.join(train_args.output_dir, "best_eval_result.json"), 'w') as fp:
                            json.dump(dev_preds.tolist(), fp)

                        print(f"### Save best dev model to {train_args.output_dir}")
                        trainer.save(os.path.join(train_args.output_dir, "best_model.bin"))

        epoch_loss_val = sum(epoch_loss) / num_batch
        print(f"epoch: {epoch + 1}, loss: {epoch_loss_val}")
        if train_args.save_last_model:
            if not os.path.exists(train_args.output_dir):
                os.makedirs(train_args.output_dir)

            print(f"### Epoch: {epoch + 1}, save last epoch model to {train_args.output_dir}")
            trainer.save(os.path.join(train_args.output_dir, "last_model.bin"))

        if epoch > num_pretrain_epoch:
            early_stopping(loss=epoch_loss_val)
            if early_stopping.early_stop():
                print("### early stopping.")
                break


def evaluate(trainer: Trainer, dataloader: DataLoader, rel_info_path, epoch=-1, tag="dev"):
    preds = []
    for batch in dataloader:
        pred = trainer.predict_step(batch, epoch)
        pred[np.isnan(pred)] = 0
        preds.append(pred)

    eval_dataset = dataloader.dataset
    preds = np.concatenate(preds, axis=0).astype(np.float32)

    ans = to_official(preds, eval_dataset.sample_list, rel_info_path)

    if len(ans) > 0:
        data_dir = os.path.dirname(eval_dataset.data_path)
        dev_path = os.path.basename(eval_dataset.data_path)
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, data_dir, dev_path)
    else:
        print(f"Warning: all preds are 0!!!")
        best_f1, best_f1_ign = 0, 0

    output = {
        tag + "_F1": best_f1,
        tag + "_F1_ign": best_f1_ign,
    }
    return best_f1, output, preds


def evaluate_pretrain(trainer: Trainer, dataloader: DataLoader, epoch, tag="dev"):
    preds = []
    labels = []
    for batch in dataloader:
        pred_ids, label_ids = trainer.predict_step(batch, epoch)
        preds.extend(pred_ids)
        labels.extend(label_ids)

    reports = classification_report(labels, preds)
    best_f1 = reports["fscore"]

    output = {}
    for k, v in reports.items():
        output[f"{tag}_{k}"] = v
    return best_f1, output, preds


def report(trainer: Trainer, dataloader: DataLoader, data_args):
    preds = []
    for batch in dataloader:
        pred = trainer.predict_step(batch)
        pred[np.isnan(pred)] = 0
        preds.append(pred)

    test_dataset = dataloader.dataset
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, test_dataset.sample_list, data_args.rel_info_path)

    return preds


def get_collate_fn(data_args: DataTrainingArguments, model_args: ModelArguments):
    dataset_name = data_args.dataset_name
    model_type = model_args.model_type
    if dataset_name == "docred":
        if model_type == "atlop" or model_type == "enhanced_bert" or model_type == "bert2phase":
            return collate_fn
    elif dataset_name == "mdocred":
        if model_type == "atlop" or model_type == "enhanced_bert" or model_type=="cnn" or model_type=="bilstm":
            return collate_fn
        elif model_type == "clip_bert":
            return clip_bert_collate_fn
        elif model_type == "mae_bert" or model_type == "mae_bert_base":
            return mae_bert_collate_fn
        elif (
                model_type == "mdocre"
                or model_type == "mdocre2stream"
                or model_type == "mdocre_unimodal"
                or model_type == "mdocre_coattn"
                or model_type == "vit_bert"
        ):
            return mdocre_collate_fn
        elif model_type == "mdocre_cross":
            return mdocre_cross_collate_fn
        elif model_type == "textual_prefix" or model_type == "textual_prefix_klx":
            return textual_prefix_collate_fn
        elif model_type == "lsr_unimodal":
            return lsr_collate_fn

    raise RuntimeError(f"Unsupported dataset and model type: {dataset_name} - {model_type}")


def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, ModelArguments, CustomizedTrainingArguments)
    )
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    train_dataset = get_dataset("train", data_args, model_args, training_args.device_type) if training_args.do_train else None
    eval_dataset = get_dataset("eval", data_args, model_args, training_args.device_type) if training_args.do_eval else None
    test_dataset = get_dataset("test", data_args, model_args, training_args.device_type) if training_args.do_predict else None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=get_collate_fn(data_args, model_args),
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory
    ) if train_dataset is not None else None

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(data_args, model_args),
        drop_last=False,
        pin_memory=True,
    ) if eval_dataset is not None else None

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(data_args, model_args),
        drop_last=False
    ) if test_dataset is not None else None

    num_train_epochs = training_args.num_train_epochs
    total_steps = int(len(train_dataloader) * num_train_epochs // training_args.gradient_accumulation_steps) \
        if train_dataloader is not None else 1
    warmup_steps = int(total_steps * training_args.warmup_ratio) \
        if train_dataloader is not None else 1
    training_args.num_training_steps = total_steps
    training_args.num_warmup_steps = warmup_steps
    print(f"### num_training_steps: {total_steps}")
    print(f"### warmup steps: {warmup_steps}")

    trainer = get_trainer(model_args, training_args)
    if training_args.load_checkpoint:
        if not os.path.exists(training_args.last_check_point_path):
            print("### Warning: last_check_point_path is empty.")
            f_dir = os.path.dirname(training_args.last_check_point_path)
            os.makedirs(f_dir, exist_ok=True)
        else:
            trainer.load(training_args.last_check_point_path)

    now = datetime.now()
    time_stamp = datetime.timestamp(now)
    run_name = f"{model_args.model_type}-{data_args.dataset_name}-{time_stamp}"
    wandb.init(project="docred", name=run_name)
    wandb.config.update(data_args)
    wandb.config.update(model_args)
    wandb.config.update(training_args)

    if training_args.do_train:
        train(training_args, trainer, train_dataloader, eval_dataloader, data_args)

    if training_args.do_eval:
        trainer.load(os.path.join(training_args.output_dir, "best_model.bin"))
        dev_score, dev_output, _ = evaluate(trainer, eval_dataloader, data_args.rel_info_path)
        print(dev_output)

    if training_args.do_predict:
        trainer.load(os.path.join(training_args.output_dir, "best_model.bin"))
        preds = report(trainer, test_dataloader, data_args)
        with open(os.path.join(training_args.output_dir, "result.json"), "w") as fp:
            json.dump(preds, fp)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()