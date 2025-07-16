import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.seas_infer import SeaSInfer
from utils.load_config import load_yaml

def get_args_infer():
    parser = argparse.ArgumentParser(description='SeaS_infer')
    parser.add_argument('--config', type=str, default='./configs/seas.yaml', help='config file path')
    parser.add_argument('--output_dir', type=str, default=None, required=False, help='The directory to save images.')
    # ----------------------
    # datasets settings
    # ----------------------
    parser.add_argument('--ref_data_dir', type=str, default=None, help='The directory of reference (normal) images.')
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size of the inference stage.')
    parser.add_argument('--add_noise_step', type=int, default=1500, help='The steps to add noise to normal images.')
    # ----------------------
    # model settings
    # ----------------------
    parser.add_argument('--gen_model_path', type=str, help='The path of the weight of the trained generation model.')
    parser.add_argument('--stable_diffusion_model_path', type=str, help='The directory of the pretrained stable-diffusion model.')
    parser.add_argument('--rmp_model_path', type=str, default=None, required=False, help='The directory of the Refined Mask Prediction (RMP) models.')
    parser.add_argument('--prompt', type=str, help='The prompt for inference.')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='The number of ddim/ddpm sampling steps.')
    parser.add_argument('--guidance_scale', type=int, default=8, help='The guidance scale of sampling.')
    # ----------------------
    # inference settings
    # ----------------------
    parser.add_argument('--total_infer_num', type=int, default=500, help='The total number of images to generate.')
    parser.add_argument('--seed_start', type=int, default=0, help='The starting random seed.')
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold to binarize the mask generation model.')
    parser.add_argument('--onlyfinal', action='store_true', help='Save only the final step results.')
    parser.add_argument('--gen_mask', action='store_true', help='Generate images with corresponding masks.')

    parser.add_argument('--device', type=str, default='cuda:0', help='The device used to generate images.')

    args = parser.parse_args()
    return args


def load_args(cfg, args):
    """ Load args from the config file """
    cfg_inf=cfg['inference']
    # ----------------------
    # datasets settings
    # ----------------------
    args.batch_size=cfg_inf['datasets']['batch_size']
    args.add_noise_step=cfg_inf['datasets']['add_noise_step']
    # ----------------------
    # model settings
    # ----------------------
    args.stable_diffusion_model_path=cfg_inf['models']['stable_diffusion_model_path']
    args.num_inference_steps=cfg_inf['models']['num_inference_steps']
    args.guidance_scale=cfg_inf['models']['guidance_scale']
    # ----------------------
    # inference settings
    # ----------------------
    args.seed_start=cfg_inf['seed_start']
    args.threshold=cfg_inf['threshold']
    args.onlyfinal=cfg_inf['onlyfinal']
    args.gen_mask=cfg_inf['gen_mask']

    args.device=cfg_inf['device']

    return args


if __name__ == "__main__":
    os.environ["ACCELERATE_FORCE_NUM_PROCESSES"] = "1"
    args = get_args_infer()
    cfg = load_yaml(args.config)
    cfg = load_args(cfg, args)
    model = SeaSInfer(args)
    model.main()