from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, CannyDataset, Perspect2PanoDataset, PerspectMask2PanoDataset, PerspectMaskCLIP2PanoDataset, NewPerspectMaskCLIP2PanoDataset, NewMaskCLIP2PanoDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ControlNet')
    parser.add_argument('--config-file', metavar="FILE",
                        default='./models/cldm_v15_clip.yaml',
                        help='config file path')
    # checkpoint and log
    parser.add_argument('--mode', type=str, default='newperspectmaskclip',
                        help='data format')
    parser.add_argument('--pretrain', type=str, default='./models/control_sd15_clip_ini.ckpt',
                        help='put the path to pre_train weight if needed')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--log-iter', type=int, default=1000,
                        help='print log every log-iter')
    parser.add_argument('--save-path', type=str, default='./checkpoints_debug/',
                        help='path to save checkpoints')
    parser.add_argument('--rotate', action='store_true', default=False)
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--y', type=int, default=0)
    parser.add_argument('--z', type=int, default=360)
    # training
    parser.add_argument('--bs', type=int, default=4,
                        help='per card batchsize')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='dataloader num_workers')

    args = parser.parse_args()
    return args


args = parse_args()
# Configs
resume_path = args.pretrain
batch_size = args.bs
logger_freq = args.log_iter
learning_rate = args.lr
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(args.config_file).cpu()
model.load_state_dict(load_state_dict(
    resume_path, location='cpu'), strict=False)
# if args.mode == 'perspectmaskclip':
#     # pretrain = load_state_dict(resume_path, location='cpu')
#     # new_pretrain = {}
#     # for k, v in pretrain.items():
#     #     print(k)
#     model.load_state_dict(load_state_dict(
#         resume_path, location='cpu'), strict=False)
# else:
#     model.load_state_dict(load_state_dict(
#         resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# saves last-K checkpoints based on "global_step" metric
# make sure you log it inside your LightningModule


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = "model_last.ckpt"
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="epoch",
    mode="max",
    dirpath=args.save_path,
    filename="ControlNet-{epoch:02d}-{global_step}",
)
# Misc
if args.mode == 'canny':
    print("Using Canny Dataset")
    dataset = CannyDataset()
elif args.mode == 'perspect':
    print("Using Perspective Dataset")
    dataset = Perspect2PanoDataset()
elif args.mode == 'perspectmask':
    print("Using Perspective Mask Dataset")
    dataset = PerspectMask2PanoDataset()
elif args.mode == 'perspectmaskclip':
    print("Using Perspective Mask CLIP Dataset")
    print("args.rotate: ", args.rotate)
    dataset = PerspectMaskCLIP2PanoDataset(args.rotate, args.x, args.y, args.z)
elif args.mode == 'newperspectmaskclip':
    print("Using New Perspective Mask CLIP Dataset")
    print("args.rotate: ", args.rotate)
    dataset = NewPerspectMaskCLIP2PanoDataset(
        args.rotate, args.x, args.y, args.z)
elif args.mode == 'newmaskclip':
    print("Using New Mask CLIP Dataset")
    print("args.rotate: ", args.rotate)
    dataset = NewMaskCLIP2PanoDataset(
        args.rotate, args.x, args.y, args.z)
else:
    dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=args.num_workers,
                        batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
save_model = CheckpointEveryNSteps(save_step_frequency=50000)


if __name__ == '__main__':
    # Train! 5 20
    # trainer = pl.Trainer(accelerator="gpu", devices=-1, strategy="ddp", max_time={
    #                      "days": 7, "hours": 1}, default_root_dir=args.save_path, callbacks=[logger, checkpoint_callback])
    trainer = pl.Trainer(accelerator="gpu", devices=-1, strategy="ddp",
                         max_epochs=25, callbacks=[checkpoint_callback])
    if args.resume is not None:
        trainer.fit(model, dataloader, ckpt_path=args.resume)
    else:
        trainer.fit(model, dataloader)
