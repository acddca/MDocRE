# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2022/11/16 23:08
# @File: arguments.py
# @Email: wangjl.nju.2020@gmail.com.
from dataclasses import dataclass, field, asdict
from transformers import TrainingArguments


@dataclass
class CustomizedTrainingArguments(TrainingArguments):

    early_stopping_patience: int = field(
        default=5,
        metadata={
            "help": "early stopping patience"
        }
    )

    device_type: str = field(
        default="cpu",
        metadata={
            "help": "use gpu or cpu"
        }
    )

    save_last_model: bool = field(
        default=True,
        metadata={
            "help": "whether to save model with training finish"
        }
    )

    load_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "whether to load model from checkpoint"
        }
    )

    last_check_point_path: str = field(
        default="",
        metadata={
            "help": "path to save target checkpoint"
        }
    )

    use_multi_gups: bool = field(
        default=False,
        metadata={
            "help": "use multiple gpus to train"
        }
    )

    num_pretrain_epoch: int = field(
        default=-1,
        metadata={
            "help": "num of pretrain epochs"
        }
    )

    pretrain_learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "learning rate for pretraining"
        }
    )

    use_token_loss: bool = field(
        default=True,
        metadata={
            "help": "if True, model will calculate token-wise matching loss"
        }
    )

    use_sent_loss: bool = field(
        default=True,
        metadata={
            "help": "if True, model will calculate sentence-wise matching loss"
        }
    )

    use_textual_guided_loss: bool = field(
        default=False,
        metadata={
            "help": "if True, model will calculate kl_div_loss"
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.early_stopping_patience < 0:
            self.early_stopping_patience = 5

    def as_dict(self):
        return asdict(self)


@dataclass
class ModelArguments:

    model_type: str = field(
        default="enhanced_bert",
        metadata={
            "help": "model type"
        }
    )

    use_final_output: bool = field(
        default=False,
        metadata={
            "help": "use final text output to fusion visual feats"
        }
    )

    use_textual_prefix: bool = field(
        default=True,
        metadata={
            "help": "whether to use prefix when fusion textual and visual features"
        }
    )

    use_gated_fusion: bool = field(
        default=True,
        metadata={
            "help": "whether to use gate mechanism when fusion textual and visual entity embeddings"
        }
    )

    use_global_gated_fusion: bool = field(
        default=False,
        metadata={
            "help": "whether to use global gete mechanism when fusion textual and visual entity embeddings"
        }
    )

    use_structural_prefix: bool = field(
        default=False,
        metadata={
            "help": "whether to use prefix when fusion structural information"
        }
    )

    use_pretrained_encoder: bool = field(
        default=False,
        metadata={
            "help": "whether to use pretrained visual features encoder"
        }
    )

    use_unimodal: bool = field(
        default=False,
        metadata={
            "help": "use unimodal in multimodal model"
        }
    )

    use_mosaic: bool = field(
        default=False,
        metadata={
            "help": "use mosaic data augment for visual input"
        }
    )

    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "path to pretrained model"
        }
    )

    video_model_name_or_path: str = field(
        default="MCG-NJU/videomae-base",
        metadata={
            "help": "path to pretrained video model"
        }
    )

    hidden_size: int = field(
        default=768,
        metadata={
            "help": "bert hidden size"
        }
    )

    entity_embed_size: int = field(
        default=768,
        metadata={
            "help": "entity embedding size"
        }
    )

    block_size: int = field(
        default=64,
        metadata={
            "help": "grouped block embedding size"
        }
    )

    num_classes: int = field(
        default=97,
        metadata={
            "help": "number of relation classes"
        }
    )

    num_labels: int = field(
        default=4,
        metadata={
            "help": "number of labels for multi-label classification task"
        }
    )

    loss_type: str = field(
        default="focal",
        metadata={
            "help": "loss type of (focal, ce)"
        }
    )

    first_phase: bool = field(
        default=False,
        metadata={
            "help": "param for bert2phase to locate phase"
        }
    )

    v_dropout: float = field(
        default=0.2,
        metadata={
            "help": "dropout rate of visual transformer hidden layer"
        }
    )

    t_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout rate of text transformer hidden layer"
        }
    )

    num_heads: int = field(
        default=12,
        metadata={
            "help": "numbers of heads for Transformer"
        }
    )

    num_co_attn_layers: int = field(
        default=2,
        metadata={
            "help": "numbers of co-attention layers"
        }
    )

    num_frames: int = field(
        default=32,
        metadata={
            "help": "numbers of video frames"
        }
    )

    image_size: int = field(
        default=224,
        metadata={
            "help": "image size for visual model"
        }
    )

    patch_size: int = field(
        default=16,
        metadata={
            "help": "patch size for visual model"
        }
    )

    tubelet_size: int = field(
        default=2,
        metadata={
            "help": "tubelet size for mae"
        }
    )

    detectron2_model_cfg: str = field(
        default="",
        metadata={
            "help": "config path to detectron visual model"
        }
    )

    num_visual_tokens: int = field(
        default=9,
        metadata={
            "help": "number of visual tokens, default 9 for image 224 x 224"
        }
    )

    num_structural_fusion_layers: int = field(
        default=1,
        metadata={
            "help": "number of transformer fusion layers for struct encoder"
        }
    )

    num_semantic_fusion_layers: int = field(
        default=1,
        metadata={
            "help": "number of transformer fusion layers for semantic encoder"
        }
    )

    in_channels: int = field(
        default=3,
        metadata={
            "help": "channels of image"
        }
    )

    def as_dict(self):
        return asdict(self)


@dataclass
class DataTrainingArguments:

    dataset_name: str = field(
        default="docred",
        metadata={
            "help": "Name of target dataset"
        }
    )

    train_path: str = field(
        default="",
        metadata={
            "help": "path of trainset"
        }
    )

    test_path: str = field(
        default="",
        metadata={
            "help": "path of testset"
        }
    )

    dev_path: str = field(
        default="",
        metadata={
            "help": "path of devset"
        }
    )

    rel_info_path: str = field(
        default="",
        metadata={
            "help": "rel2id config file for relation extraction"
        }
    )

    visual_ann_path: str = field(
        default=r"D:\Dataset\MDocRED\frame_label",
        metadata={
            "help": "path to store visual annotation files"
        }
    )

    visual_data_path: str = field(
        default=r"D:\Dataset\MDocRED\frame",
        metadata={
            "help": "path to store visual data"
        }
    )

    tokenizer_path: str = field(
        default="",
        metadata={
            "help": "model path or pretrained directory of tokenizer"
        }
    )

    max_len: int = field(
        default=1024,
        metadata={
            "help": "max sequence len"
        }
    )

    override_cache: bool = field(
        default=True,
        metadata={
            "help": "rebuild dataset or not"
        }
    )

    data_cache_path: str = field(
        default="",
        metadata={
            "help": "path to cache dataset"
        }
    )

    only_pos_sample: bool = field(
        default=False,
        metadata={
            "help": "whether to only train positive samples"
        }
    )

    neg_pos_ratio: float = field(
        default=1.0,
        metadata={
            "help": "sample ration between negative and positive"
        }
    )

    truncate_text_inputs: bool = field(
        default=True,
        metadata={
            "help": "whether to truncate text inputs to max_len"
        }
    )

    def __post_init__(self):
        # se† attributes
        if len(self.dataset_name) == 0:
            # set train_path, dev_path, test_path
            pass

    def as_dict(self):
        return asdict(self)


if __name__ == "__main__":
    model_args = ModelArguments()
    print(model_args.as_dict())
