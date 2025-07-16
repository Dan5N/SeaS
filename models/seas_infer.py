import os
import sys
sys.path.append(os.getcwd())
import time

import torch

from diffusers import UNet2DConditionModel

from datasets.seas_dataset import SeaSTestDataset, collate_fn_test
from models.modules._unet_2d_condition import UNet2DConditionModel
from models.modules._autoencoder_kl import AutoencoderKL
from models.modules._RMP import RMP
from models.modules._SeaSpipeline import SeaSPipeline
from models.modules._scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from utils.show_attn_utils import AttentionStore, register_attention_control


class SeaSInfer():
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        self.device = args.device
        # ----------------------
        # datasets settings
        # ----------------------
        self.ref_data_dir = args.ref_data_dir
        self.num_samples = args.batch_size
        self.add_noise_step = args.add_noise_step
        # ----------------------
        # model settings
        # ----------------------
        self.rmp_model_path = args.rmp_model_path
        self.prompt = args.prompt
        self.num_inference_steps = args.num_inference_steps
        self.guidance_scale = args.guidance_scale
        # ----------------------
        # inference settings
        # ----------------------
        self.total_infer_num = args.total_infer_num
        self.seed_start = args.seed_start
        self.only_final = args.onlyfinal
        self.threshold = args.threshold

        # Initialize the AttentionStore controller if needed.
        self.controller = AttentionStore()

        # Load the anomaly mask generator if specified.
        if args.gen_mask:
            self.rmp = RMP(resolution=512, out_dim=1, emb_dim=1280)
            self.rmp.load_state_dict(torch.load(args.rmp_model_path))

        # Create the StableDiffusionPipeline.
        self.pipe = SeaSPipeline.from_pretrained(
            args.gen_model_path,
            unet=UNet2DConditionModel.from_pretrained(args.gen_model_path, subfolder="unet"),
            scheduler=DPMSolverMultistepScheduler.from_pretrained(args.stable_diffusion_model_path, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.stable_diffusion_model_path, subfolder="vae"),
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)

        # Use the normal images as the initial noise after adding noise.
        if self.ref_data_dir:
            self.infer_dataset = SeaSTestDataset(
                instance_data_root=self.ref_data_dir,
                size=512,
                center_crop=True
            )
            self.infer_dataloader = torch.utils.data.DataLoader(
                self.infer_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda examples: collate_fn_test(examples)
            )
            
    def main(self):
        controller = self.controller
        pipe = self.pipe
        ref_data_dir = self.ref_data_dir
        output_dir = self.output_dir
        rmp_model_path = self.rmp_model_path
        num_samples = self.num_samples
        total_infer_num  = self.total_infer_num
        infer_dataloader = self.infer_dataloader
        prompt = self.prompt
        seed_start = self.seed_start
        rmp = self.rmp
        device = self.device
        add_noise_step = self.add_noise_step
        onlyfinal = self.only_final
        threshold = self.threshold

        all_images = []
        all_masks = []
        times = []

        pipe = pipe.to(device)
        register_attention_control(pipe, controller)
        device = torch.device(device)
        
        for index in range(total_infer_num // num_samples):
            generator = [torch.Generator(device=device).manual_seed(i) for i in range(seed_start, seed_start+num_samples)]
            if ref_data_dir: # if use the real normal images as initial noise after adding the noise.
                data = iter(infer_dataloader)
                first_item = next(data)
                start_time = time.time()
                images = pipe(prompt=prompt, 
                            num_images_per_prompt=num_samples, 
                            num_inference_steps=self.num_inference_steps, 
                            guidance_scale=self.guidance_scale,
                            add_noise=True,
                            generator=generator,
                            ref_sample=first_item["pixel_values"].to(dtype=torch.float16), 
                            onlyfinal=onlyfinal, 
                            add_noise_step=int(add_noise_step),
                            threshold=threshold,
                            rmp=rmp)
                end_time = time.time()
                times.append(end_time - start_time)
            else :  # if use the random noise
                start_time = time.time()
                images = pipe(prompt=prompt, 
                            num_images_per_prompt=num_samples, 
                            num_inference_steps=self.num_inference_steps, 
                            guidance_scale=self.guidance_scale,
                            generator=generator, 
                            onlyfinal=onlyfinal,
                            rmp=rmp)
                end_time = time.time()
                times.append(end_time - start_time)

            # We only use this to save the generated images.
            if onlyfinal: 
                all_images.extend(images["image"])
                if rmp_model_path:
                    all_masks = images["anomask"]
    
            seed_start = seed_start + num_samples
            if output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                
            if not os.path.exists(f'{output_dir}/image'):
                os.mkdir(f'{output_dir}/image')
            if rmp_model_path:
                if not os.path.exists(f'{output_dir}/mask'):
                    os.mkdir(f'{output_dir}/mask')

            num = index * num_samples
            for img in all_images:
                img.save(f'{output_dir}/image/{num:03}.png', dpi=(600,600), bbox_inches='tight')
                num += 1
            
            print(len(all_masks))
            if rmp_model_path:
                num = index * num_samples
                threshold_folder = os.path.join(output_dir, f'mask')
                if not os.path.exists(threshold_folder):
                    os.mkdir(threshold_folder)
                for mask in all_masks:            
                    mask.save(f'{threshold_folder}/{num:03}.png', dpi=(600,600), bbox_inches='tight')
                    num += 1
            
            all_images.clear()
            all_masks.clear()

            del images
            average = sum(times) / len(times)
            print(f"average:{average}")       
        return