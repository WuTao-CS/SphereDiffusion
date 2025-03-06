conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai
cd /group/30042/jerryxwli/code/ControlNet/
# python train.py --save-path 'perspect_checkpoints' --num-workers 10
python train.py --save-path 'ablation_checkpoints/clip_dcn_new_bs4' \
    --num-workers 10 \
    --config-file 'models/cldm_v15_clip_new_resize_dcn3.yaml' \
    --pretrain 'models/control_sd15_ini_resize_zero_dcn3.ckpt' \
    --bs 4
