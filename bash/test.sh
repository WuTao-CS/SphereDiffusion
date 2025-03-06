python train.py --save-path 'perspect_new_clip_mask_checkpoints_temp' \
    --num-workers 8 \
    --config-file 'models/cldm_v15_clip_new.yaml' \
    --pretrain 'models/control_sd15_clip_new_ini.ckpt' \
    --bs 8