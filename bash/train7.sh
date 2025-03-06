conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai
cd /group/30042/jerryxwli/code/ControlNet/
python train2.py --save-path 'baseline_final_checkpoints' \
    --num-workers 10 \
    --mode 'perspectmask' \
    --config-file 'models/cldm_v15.yaml' \
    --pretrain 'models/control_sd15_ini.ckpt' \
    --bs 6