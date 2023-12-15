cd ../tools
dataset_name="mdocred"
model_name="mdocre"
output_name="$dataset_name-$model_name"
used_batch_size=2
num_workers=2

export CUDA_VISIBLE_DEVICES=2

echo "### Dataset Name:                              $dataset_name"
echo "### Model Name:                                $model_name"
echo "### Batch Size:                                $used_batch_size"
echo "### Output Name:                               $output_name"

python -u train.py \
--seed 66 \
--model_name_or_path "/home/nlper_data/wangjl/pretrained/bert-base-uncased" \
--tokenizer_path "/home/nlper_data/wangjl/pretrained/bert-base-uncased" \
--output_dir "../checkpoints/$output_name" \
--dataset_name "mdocred" \
--train_path "/home/nlper_data/wangjl/dataset/MDocRED/train_annotated.json" \
--dev_path "/home/nlper_data/wangjl/dataset/MDocRED/dev.json" \
--test_path "/home/nlper_data/wangjl/dataset/MDocRED/test.json" \
--rel_info_path "/home/nlper_data/wangjl/dataset/MDocRED/rel_info.json" \
--visual_ann_path "/home/nlper_data/wangjl/dataset/MDocRED/frame_label" \
--visual_data_path "/home/nlper_data/wangjl/dataset/MDocRED/frame" \
--override_cache False \
--data_cache_path "/home/nlper_data/wangjl/dataset/MDocRED/mdocre" \
--model_type "mdocre" \
--save_strategy "epoch" \
--evaluation_strategy "epoch" \
--per_device_train_batch_size $used_batch_size \
--per_device_eval_batch_size $used_batch_size \
--max_len 512 \
--hidden_size 768 \
--entity_embed_size 768 \
--block_size 32 \
--num_heads 12 \
--num_co_attn_layers 2 \
--v_dropout 0.5 \
--t_dropout 0.5 \
--num_frames 128 \
--image_size 352 \
--in_channels 3 \
--num_train_epochs 100 \
--early_stopping_patience 10 \
--num_structural_fusion_layers 2 \
--num_semantic_fusion_layers 2 \
--warmup_steps 100 \
--warmup_ratio 0.06 \
--max_grad_norm 1.0 \
--learning_rate 1e-5 \
--gradient_accumulation_steps 1 \
--num_classes 22 \
--eval_steps 100 \
--dataloader_num_workers $num_workers \
--dataloader_drop_last True \
--dataloader_pin_memory True \
--load_best_model_at_end True \
--loss_type "ce" \
--do_train \
--do_eval \
--use_multi_gups True \
--device_type "cuda"

#indices=(0 1 2 3 4)
#lrs=(5e-5 4e-5 3e-5 2e-5 1e-5)
#
#for i in ${indices[@]}
#do
#  runs=$(expr $i + 1)
#  echo "### start run $runs"
#  lr=${lrs[$i]}
#
#  python -u train.py \
#  --seed 66 \
#  --model_name_or_path "/home/nlper_data/wangjl/pretrained/bert-base-uncased" \
#  --tokenizer_path "/home/nlper_data/wangjl/pretrained/bert-base-uncased" \
#  --output_dir "../checkpoints/$output_name" \
#  --dataset_name "mdocred" \
#  --train_path "/home/nlper_data/wangjl/dataset/MDocRED/train_annotated.json" \
#  --dev_path "/home/nlper_data/wangjl/dataset/MDocRED/dev.json" \
#  --test_path "/home/nlper_data/wangjl/dataset/MDocRED/test.json" \
#  --rel_info_path "/home/nlper_data/wangjl/dataset/MDocRED/rel_info.json" \
#  --visual_ann_path "/home/nlper_data/wangjl/dataset/MDocRED/frame_label" \
#  --visual_data_path "/home/nlper_data/wangjl/dataset/MDocRED/frame" \
#  --override_cache False \
#  --data_cache_path "/home/nlper_data/wangjl/dataset/MDocRED/mdocre" \
#  --model_type "mdocre" \
#  --save_strategy "epoch" \
#  --evaluation_strategy "epoch" \
#  --per_device_train_batch_size $used_batch_size \
#  --per_device_eval_batch_size $used_batch_size \
#  --max_len 512 \
#  --hidden_size 768 \
#  --entity_embed_size 768 \
#  --block_size 32 \
#  --num_heads 12 \
#  --num_co_attn_layers 2 \
#  --v_dropout 0.5 \
#  --t_dropout 0.5 \
#  --num_frames 128 \
#  --image_size 352 \
#  --in_channels 3 \
#  --num_train_epochs 100 \
#  --early_stopping_patience 10 \
#  --num_structural_fusion_layers 2 \
#  --num_semantic_fusion_layers 2 \
#  --warmup_steps 100 \
#  --warmup_ratio 0.06 \
#  --max_grad_norm 1.0 \
#  --learning_rate $lr \
#  --gradient_accumulation_steps 1 \
#  --num_classes 22 \
#  --eval_steps 100 \
#  --dataloader_num_workers $num_workers\
#  --dataloader_drop_last True \
#  --dataloader_pin_memory True \
#  --load_best_model_at_end True \
#  --loss_type "ce" \
#  --do_train \
#  --do_eval \
#  --use_multi_gups True \
#  --device_type "cuda"
#done
