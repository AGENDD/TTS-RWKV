#3M
# dim=128
# blocks=5
#10M
# dim=320
# blocks=5
#12M
dim=256
blocks=12

learning_rate=1e-5
num_epochs=8000
batch_size=32

# export CUDA_VISIBLE_DEVICES=1,2,3
export HF_ENDPOINT="https://hf-mirror.com" 
python pretrain_multiGPU.py --dim $dim --n_blocks $blocks --learning_rate $learning_rate --num_epochs $num_epochs --batch_size $batch_size