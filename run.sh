#!/bin/bash
set -e 

stage=1
stop_stage=3
gpu_num=0

. ./utils/parse_options.sh

data=./data_v2

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    bash local/run_speech_command_v1_prepare.sh
fi

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for x in train valid test; do
        python local/generate_labels_scp.py $data/$x/text $data/$x/labels.scp
        python hkws/bins/prepare_torch_scp.py $data/$x/wav.scp $data/$x/labels.scp $data/$x/torch.scp
    done

fi

train_scp=$data/train/torch.scp
dev_scp=$data/valid/torch.scp
test_scp=$data/test/torch.scp

seed=15
dropout=0.3
for seed in 10; do
for learning_rate in 0.020; do
#for learning_rate in 0.010 0.020 0.002; do
#for learning_rate in 0.004 0.006 0.008 0.010 0.020; do
for layer_type in "multi_scale_tcn"; do
weight_decay=5e-5
batch_size=100
warmup=2000
optimizer="noam" # noam, sgd

input_dim=80
output_dim=11
num_layers=3
block_size=4
hidden_dim=20
previous_model=""
kernel_size=3


#config=conf/train_config_sa_mfcc80.yml
config=conf/train_config_harder_sa_mfcc80.yml
#config=conf/train_config_mfcc80.yml
#config=conf/train_config.yml
sa_c=mfcc80_sa2_t20_f40
#sa_c=mfcc80_sa1_t10_f40
#sa_c=mfcc80_sa0_t0_f0
date_c=`date -I`
#date_c="2021-03-27"
data_c=speech_command_v2_symmetric${sa_c}
proto_c=${layer_type}_nl${num_layers}_bls${block_size}_hd${hidden_dim}_ks${kernel_size}_dp${dropout}
opt_c=bs${batch_size}_op-${optimizer}_lr${learning_rate}_wd${weight_decay}_wu${warmup}

save_dir=exp/${date_c}_${data_c}_${proto_c}_${opt_c}_${seed}
#previous_model=$save_dir/1.pt

mkdir -p $save_dir
debug="-m pdb"
debug="-u"

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "training at ${save_dir}/log_train.txt"
    #$cuda_cmd $save_dir/log_train.txt python $debug train.py \
    CUDA_VISIBLE_DEVICES=$gpu_num python $debug hkws/bins/train.py \
        --seed=$seed --train=1 --test=0 \
        --config=$config \
        --encoder=$layer_type \
        --num-layers=$num_layers \
        --block-size=$block_size \
        --dropout=$dropout \
        --input-dim=$input_dim \
        --output-dim=$output_dim \
        --hidden-dim=$hidden_dim \
        --kernel-size=$kernel_size \
        --batch-size=$batch_size \
        --warm-up-steps=$warmup \
        --learning-rate=$learning_rate \
        --optimizer=$optimizer \
        --init-weight-decay=$weight_decay \
        --load-model=$previous_model \
        --max-epochs=100 \
        --use-cuda=1 \
        --train-scp=$train_scp \
        --num-workers=6 \
        --dev-scp=$dev_scp \
        --save-dir=$save_dir \
        --log-interval=10 | tee $save_dir/log_train.txt
fi

num_avg=10
#save_dir=exp/2021-03-28_speech_command_symmetricmfcc80_sa2_t20_f40_multi_scale_tcn_nl1_bls4_hd32_ks3_dp0.3_bs100_op-noam_lr0.004_wd5e-5_wu2000_10/
#save_dir=exp/2021-03-28_speech_command_symmetricmfcc80_sa2_t20_f40_multi_scale_tcn_nl2_bls4_hd20_ks3_dp0.3_bs100_op-noam_lr0.004_wd5e-5_wu2000_10/
#save_dir=exp/2021-03-29_speech_command_symmetricmfcc80_sa2_t20_f40_multi_scale_tcn_nl3_bls4_hd20_ks3_dp0.3_bs100_op-noam_lr0.004_wd5e-5_wu2000_10/
#save_dir=exp/2021-03-29_speech_command_symmetricmfcc80_sa2_t20_f40_multi_scale_tcn_nl4_bls4_hd20_ks3_dp0.3_bs100_op-noam_lr0.004_wd5e-5_wu2000_10/
#save_dir=exp/2021-03-29_speech_command_symmetricmfcc80_sa2_t20_f40_multi_scale_tcn_nl5_bls4_hd20_ks3_dp0.3_bs100_op-noam_lr0.004_wd5e-5_wu2000_10/
#save_dir=exp/2021-04-01_speech_command_symmetricmfcc80_sa2_t20_f40_multi_scale_tcn_nl4_bls4_hd100_ks5_dp0.3_bs100_op-noam_lr0.004_wd5e-5_wu2000_10/
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python $debug hkws/bins/average_model.py \
                        --dst_model=$save_dir/avg.pt \
                        --src_path=$save_dir \
                        --num=$num_avg --val_best 
fi

best_model=$save_dir/avg.pt
if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    #$cuda_cmd $save_dir/log_test.txt python $debug train.py \
    echo "testing at ${save_dir}/log_test.txt"
    CUDA_VISIBLE_DEVICES=$gpu_num python $debug hkws/bins/train.py \
            --seed=10 --train=0 --test=1 \
            --config=$config \
            --encoder=$layer_type \
            --num-layers=$num_layers \
            --block-size=$block_size \
            --input-dim=$input_dim \
            --hidden-dim=$hidden_dim \
            --kernel-size=$kernel_size \
            --num-layers=$num_layers \
            --output-dim=$output_dim \
            --dropout=$dropout \
            --batch-size=$batch_size \
            --load-model=$best_model \
            --use-cuda=1 \
            --test-scp=$test_scp \
            --num-workers=6 \
            --log-interval=100 | tee $save_dir/log_test.txt
fi
done
done
done

