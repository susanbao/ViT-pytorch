cuda=$1
data_type=$2
dim=$3
class=$4

# CUDA_VISIBLE_DEVICES=$cuda python3 train_without_eval.py --name ViT_${data_type}_${class}_all --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range all --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --num_steps 20000 --eval_every 5000 --warmup_steps 2000 --learning_rate 0.03 --enable_backbone_grad --max_grad_norm 10 --ordinal_class_num ${class} --region_size 16 --enable_wandb

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_all --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range all --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 5000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_all --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range all --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 10000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_all --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range all --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 15000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_all --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range all --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 20000

# CUDA_VISIBLE_DEVICES=$cuda python3 train_without_eval.py --name ViT_${data_type}_${class}_region --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range region --train_batch_size 16 --eval_batch_size 16  --data_name feature_box_level_ViT --input_feature_dim $dim --num_steps 100000 --eval_every 5000 --warmup_steps 2000 --learning_rate 0.03 --enable_backbone_grad --max_grad_norm 10 --ordinal_class_num ${class} --region_size 16 --enable_wandb

# for step in 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000
for step in 55000
do
    CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step $step
done

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 5000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 10000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 15000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_16 --pretrained_dir ../ViT_ckpts/ViT-B_16.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 16 --checkpoint_step 20000

# CUDA_VISIBLE_DEVICES=$cuda python3 train_without_eval.py --name ViT_${data_type}_${class}_region_32_32 --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_32 --pretrained_dir ../ViT_ckpts/ViT-B_32.npz --loss_range region --train_batch_size 16 --eval_batch_size 16  --data_name feature_box_level_ViT --input_feature_dim $dim --num_steps 20000 --eval_every 5000 --warmup_steps 2000 --learning_rate 0.03 --enable_backbone_grad --max_grad_norm 10 --ordinal_class_num ${class} --region_size 32 --enable_wandb

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region_32_32 --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_32 --pretrained_dir ../ViT_ckpts/ViT-B_32.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 32 --checkpoint_step 5000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region_32_32 --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_32 --pretrained_dir ../ViT_ckpts/ViT-B_32.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 32 --checkpoint_step 10000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region_32_32 --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_32 --pretrained_dir ../ViT_ckpts/ViT-B_32.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 32 --checkpoint_step 15000

# CUDA_VISIBLE_DEVICES=$cuda python3 test.py --name ViT_${data_type}_${class}_region_32_32 --data_dir ../pytorch-segmentation/pro_data/$data_type/ --model_data_type $data_type --gradient_accumulation_steps 4 --model_type ViT-B_32 --pretrained_dir ../ViT_ckpts/ViT-B_32.npz --loss_range region --train_batch_size 16 --eval_batch_size 16 --data_name feature_box_level_ViT --input_feature_dim $dim --ordinal_class_num ${class} --region_size 32 --checkpoint_step 20000