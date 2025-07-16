import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.seas import SeaS
from utils.load_config import load_yaml

def get_args():
    parser = argparse.ArgumentParser(description='SeaS')
    parser.add_argument('--config', type=str, default='./configs/seas.yaml', help='config file path')
    # ----------------------
    # dataset setting
    # ----------------------
    parser.add_argument('--instance_data_dir', type=str, default=None, help="The directory of anomaly images in datasets.")
    parser.add_argument('--mask_dir', type=str, default=None, help="The directory of the corresponding masks of defect images in datasets.")
    parser.add_argument('--normal_data_dir', type=str, default=None, help="The directory of normal images in datasets.")
    parser.add_argument('--num_normal_images', type=int, default=None, help="Minimal number of normal images for normal image alignment.")
    parser.add_argument('--resolution', type=int, default=None, help= "The size to which the image is resized.")
    parser.add_argument('--rotation', action='store_true', help= "Whether to rotate the input images.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."))
    parser.add_argument('--gen_train_batch_size', type=int, default=None, help="The batch_size of the training generation process (if `--with_object_preserevation`, the actual batch_size on the device is twice of this parameter)")
    parser.add_argument('--mask_train_batch_size', type=int, default=None, help="The batch_size of the training mask process (if `--with_object_preserevation`, the actual batch_size on the device is twice of this parameter).")
    # ----------------------
    # models setting
    # ----------------------
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help="Path to pretrained model or model identifier from huggingface.co/models.")   
    parser.add_argument('--seas_trained_model_path', type=str, default=None, help="Path to SeaS trained generation models.")
    parser.add_argument('--normal_token_num', type=int, default=None, help="The number of normal tokens")
    parser.add_argument('--anomaly_token_num', type=int, default=None, help="The number of anomaly tokens")
    parser.add_argument('--initializer_token', type=str, default=None, help="A token to use as initializer word.")
    parser.add_argument('--revision',type=str,default=None,required=False, help="Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision.")
    parser.add_argument('--text_encoder_use_attention_mask',action="store_true",help="Whether to use attention mask for the text encoder")
    parser.add_argument('--tokenizer_max_length',type=int,default=None, help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.")
    # ----------------------
    # optimizer hyperparameters
    # ----------------------
    parser.add_argument('--gen_learning_rate', type=float, default=None, help="Initial learning rate (after the potential warmup period) to use, for the training generation process.")
    parser.add_argument('--rmp_learning_rate', type=float, default=None, help="Initial learning rate (after the potential warmup period) to use, for the training mask process.")
    parser.add_argument('--encoder_learning_rate', type=float, default=None, help="Initial learning rate (after the potential warmup period) to use, for the text embedings in the text enoder.")
    parser.add_argument('--lr_scheduler', type=str, default=None, help="Learning rate scheduler")
    parser.add_argument('--lr_warmup_steps', type=int, default=None, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument('--adam_beta2', type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument('--lr_num_cycles',type=int, default=1,help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument('--lr_power', type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument('--use_8bit_adam', action='store_true', help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument(
        '--set_grads_to_none',
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    # ----------------------
    # steps & checkpointing
    # ----------------------
    parser.add_argument('--gen_train_steps', type=int, default=None, help="Total number of training steps to perform, for the training of image generation model.")
    parser.add_argument('--mask_train_steps', type=int, default=None, help="Total number of training steps to perform, for the training of mask generation model.")
    parser.add_argument('--checkpointing_steps', type=int, default=None, help='Checkpointing steps')
    # ----------------------
    # training setting
    # ----------------------
    parser.add_argument('--output_dir', type=str, default=None, help="The directory to save checkpoints.")
    parser.add_argument('--train_text_encoder', type=bool, default=True, help="Whether or not to train the text_encoder.")
    parser.add_argument('--with_Ni_Alignment', type=bool, default=False, help="Whether to use normal images for normal image alignment.")
    parser.add_argument('--offset_noise', type=bool, default=False, help=("Fine-tuning against a modified noise"" See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."))
    parser.add_argument(
        '--mixed_precision', type=str, default=None, choices=["no", "fp16", "bf16"],
        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."))
    parser.add_argument(
        '--logging_dir',
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    args = parser.parse_args()
    return args

def load_args(cfg, args):
    """ Load args from the config file """
    cfg_train=cfg['training']
    # ----------------------
    # datasets settings
    # ----------------------
    args.resolution=cfg_train['datasets']['resolution']
    args.num_normal_images=cfg_train['datasets']['num_normal_images']
    args.rotation=cfg_train['datasets']['rotation']
    args.dataloader_num_workers=cfg_train['datasets']['dataloader_num_workers']
    args.gen_train_batch_size=cfg_train['datasets']['batch_size']['gen_train_batch_size']
    args.mask_train_batch_size=cfg_train['datasets']['batch_size']['mask_train_batch_size']
    # ----------------------
    # model settings
    # ----------------------
    args.pretrained_model_name_or_path=cfg_train['models']['pretrained_model_name_or_path']
    args.normal_token_num=cfg_train['models']['prompt']['normal_token_num']
    args.anomaly_token_num=cfg_train['models']['prompt']['anomaly_token_num']
    args.initializer_token=cfg_train['models']['prompt']['initializer_token']
    # ----------------------
    # optimizer hyperparameters
    # ----------------------
    args.gen_learning_rate=cfg_train['optimizer']['gen_learning_rate']
    args.rmp_learning_rate=cfg_train['optimizer']['rmp_learning_rate']
    args.encoder_learning_rate=cfg_train['optimizer']['encoder_learning_rate']
    args.lr_scheduler=cfg_train['optimizer']['lr_scheduler']
    args.lr_warmup_steps=cfg_train['optimizer']['lr_warmup_steps']
    args.gradient_accumulation_steps=cfg_train['optimizer']['gradient_accumulation_steps']
    args.adam_beta1=cfg_train['optimizer']['adam_beta1']
    args.adam_beta2=cfg_train['optimizer']['adam_beta2']
    args.adam_weight_decay=cfg_train['optimizer']['adam_weight_decay']
    args.adam_epsilon1e_08=float(cfg_train['optimizer']['adam_epsilon1e-08'])
    args.lr_num_cycles=cfg_train['optimizer']['lr_num_cycles']
    args.lr_power=cfg_train['optimizer']['lr_power']
    args.use_8bit_adam=cfg_train['optimizer']['use_8bit_adam']
    # ----------------------
    # training setting
    # ----------------------
    args.train_text_encoder=cfg_train['train_text_encoder']
    args.with_Ni_Alignment=cfg_train['with_Ni_Alignment']
    args.mixed_precision=cfg_train['mixed_precision']
    args.offset_noise=cfg_train['offset_noise']

    return args

if __name__ == "__main__":
    os.environ["ACCELERATE_FORCE_NUM_PROCESSES"] = "1"
    args = get_args()
    cfg = load_yaml(args.config)
    cfg = load_args(cfg, args)
    model = SeaS(args)
    model.main()
    