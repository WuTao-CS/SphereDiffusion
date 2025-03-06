conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai
cd /group/30042/jerryxwli/code/ControlNet/

python train.py --save-path 'perspect_resize_clip_mask_checkpoints_dcn3_constrastive_rotate_final' \
    --num-workers 10 \
    --config-file 'models/cldm_v15_clip_new_resize_contrastive_dcn3_rotate.yaml' \
    --pretrain 'models/control_sd15_ini_resize_zero_dcn3_constractive.ckpt' \
    --bs 4 \
    --rotate \
    --x 10 \
    --y 10