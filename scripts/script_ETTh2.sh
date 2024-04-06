# export CUDA_VISIBLE_DEVICES=2

experiment_name=KUNetTest_20240405

# input sequence length
seq_len=720

# model information
model_name=NKUNet
model_id_name=KUN

# dataset information
root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
data_name=ETTh2

# make dir for logs
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$experiment_name ]; then
    mkdir ./logs/$experiment_name
fi

random_seed=2021

# Main experiments loop
for pred_len in  96 192 336 720
do
    python -u run_kun.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len $seq_len \
    --pred_len $pred_len \
                         \
    --enc_in 7 \
    --dropout 0.3\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --lradj CARD \
    
    2>&1 | tee logs/$experiment_name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log \

done