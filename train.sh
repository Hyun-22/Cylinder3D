name=cylinder_asym_networks
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym.py \
2>&1 | tee ./logs/${name}_logs_tee.txt