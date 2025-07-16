import logging
import math
import itertools
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path

import torch
import torch.nn.functional as F

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import transformers
from transformers import AutoTokenizer, PretrainedConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm

from datasets.seas_dataset import SeaSTrainDataset, collate_fn
from models.modules._unet_2d_condition import UNet2DConditionModel
from models.modules._autoencoder_kl import AutoencoderKL
from models.modules._RMP import RMP
from models.modules._focal_loss import FocalLoss

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.19.0.dev0")
logger = get_logger(__name__)

class SeaS_mask():
    def __init__(self, args):
        # init args
        self.args = args
        self.output_dir = args.output_dir
        self.logging_dir = args.logging_dir
        # ----------------------
        # training settings
        # ----------------------
        self.mixed_precision = args.mixed_precision
        self.offset_noise = args.offset_noise
        self.with_Ni_Alignment = args.with_Ni_Alignment
        # ----------------------
        # datasets settings
        # ----------------------
        self.instance_data_dir = args.instance_data_dir
        self.mask_dir = args.mask_dir
        self.normal_data_root = args.normal_data_dir if args.with_Ni_Alignment else None
        self.resolution = args.resolution
        self.num_normal_images = args.num_normal_images
        self.rotation = args.rotation
        self.dataloader_num_workers = args.dataloader_num_workers
        self.train_batch_size = args.gen_train_batch_size
        # ----------------------
        # model & prompt settings
        # ----------------------
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.seas_trained_model_path = args.seas_trained_model_path
        self.normal_token_num = args.normal_token_num
        self.anomaly_token_num = args.anomaly_token_num
        self.tokenizer_max_length = args.tokenizer_max_length
        self.revision = args.revision
        self.text_encoder_use_attention_mask = args.text_encoder_use_attention_mask
        # ----------------------
        # optimizer hyperparameters
        # ----------------------
        self.rmp_learning_rate = args.rmp_learning_rate
        self.lr_scheduler = args.lr_scheduler
        self.lr_warmup_steps = args.lr_warmup_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.adam_beta1, self.adam_beta2, self.adam_weight_decay, self.adam_epsilon = \
            args.adam_beta1, args.adam_beta2, args.adam_weight_decay, args.adam_epsilon
        self.lr_num_cycles = args.lr_num_cycles
        self.lr_power = args.lr_power
        self.use_8bit_adam = args.use_8bit_adam
        self.set_grads_to_none = args.set_grads_to_none
        # ----------------------
        # steps & checkpointing
        # ----------------------
        self.max_train_steps = args.mask_train_steps
        self.checkpointing_steps = args.checkpointing_steps

        # For VisA, we don't rotate the images
        if "visa" in str(self.instance_data_dir):
            self.rotation = False

    # Import the textencoder used in stable diffusion from model's name or path.
    def import_model_class_from_model_name_or_path(self, pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")
    
    # Get the embeedings for prompt.
    def encode_prompt(self, text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_input_ids = input_ids.to(text_encoder.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(text_encoder.device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    def main(self):
        logging_dir = Path(self.output_dir, self.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=logging_dir)
        print(os.environ.get("ACCELERATE_FORCE_NUM_PROCESSES"))

        # init
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

        num_warmup_steps = self.lr_warmup_steps * accelerator.num_processes
        num_training_steps = self.max_train_steps * accelerator.num_processes

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.seas_trained_model_path, subfolder="tokenizer", revision=self.revision, use_fast=False)
        # import correct text encoder class
        text_encoder_cls = self.import_model_class_from_model_name_or_path(self.seas_trained_model_path, self.revision)
        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(self.seas_trained_model_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(self.seas_trained_model_path, subfolder="text_encoder", revision=self.revision)
        unet = UNet2DConditionModel.from_pretrained(self.seas_trained_model_path, subfolder="unet", revision=self.revision)
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision)
        
        rmp = RMP(resolution = self.resolution, out_dim=1, emb_dim=1280)
        rmp = rmp.to(accelerator.device, dtype=torch.float32)

        if vae is not None:
            vae.requires_grad_(False)

        text_encoder.requires_grad_(False)

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if accelerator.unwrap_model(unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        if vae is not None:
            vae.to(accelerator.device, dtype=weight_dtype)
        if text_encoder is not None:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        unet.to(accelerator.device, dtype=torch.float32)

        logger.info(f"***** Running Generation Model *****")
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # optimizer
        optimizer_class = bnb.optim.AdamW8bit
        optimizer_rmp = optimizer_class(
            itertools.chain(rmp.parameters()),
            lr=self.rmp_learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )
        
        # scheduler
        lr_scheduler_rmp = get_scheduler(self.lr_scheduler, optimizer=optimizer_rmp, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=self.lr_num_cycles, power=self.lr_power)

        # dataset and dataloader
        train_dataset = SeaSTrainDataset(
            instance_data_root=self.instance_data_dir,
            mask_root=self.mask_dir, 
            normal_data_root=self.normal_data_root,
            normal_num=self.num_normal_images, 
            tokenizer=tokenizer, 
            size=self.resolution, 
            rotation=self.rotation,
            tokenizer_max_length=self.tokenizer_max_length,
            normal_token_num=self.normal_token_num, 
            anomaly_token_num=self.anomaly_token_num,      
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True, 
            collate_fn=lambda examples: collate_fn(examples, self.with_Ni_Alignment), 
            num_workers=self.dataloader_num_workers
        )

        # Prepare everything with our `accelerator`.
        rmp, optimizer_rmp, train_dataloader, lr_scheduler_rmp = accelerator.prepare(
                rmp, optimizer_rmp, train_dataloader, lr_scheduler_rmp
            )

        # Training info#
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
            
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps
        logger.info(f"***** Running generation training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.num_train_epochs):
            #forward
            rmp.train()
            unet.train()
            for step, batch in enumerate(train_dataloader):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)               
                # Skip steps until we reach the resumed step
                with accelerator.accumulate(unet):         
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (4,), device=accelerator.device)
                    timesteps = timesteps.long()

                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    mask = batch["mask"].to(dtype=weight_dtype)
                    mask = mask.unsqueeze(1)
                    if vae is not None:
                        # Convert images to latent space
                        model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        model_input = model_input * vae.config.scaling_factor  #[batch,4,64,64]
                    else:
                        model_input = pixel_values
                    # Sample noise that we'll add to the model input
                    if self.offset_noise:
                        noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                            model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                        )
                    else:
                        noise = torch.randn_like(model_input)
                    _, channels, _, _ = model_input.shape
                    # Add noise to the model input according to the noise magnitude at each timestep
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.encode_prompt(text_encoder, batch["input_ids"], batch["attention_mask"], text_encoder_use_attention_mask=self.text_encoder_use_attention_mask)

                    if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    class_labels = None

                    # Predict the noise residual
                    model_pred, model_pred_list, emb, _ = unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels)
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    if self.with_Ni_Alignment:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, _ = torch.chunk(model_pred, 2, dim=0)
                    
                    # loss
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = torch.zeros_like(model_pred)
                    for index in range(model_pred.shape[0]):
                        latents[index] = noise_scheduler.step(model_pred[index].unsqueeze(0), timesteps[index], noisy_model_input[index], return_dict=False)[0]
                    image, vae_features = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]                          
                    
                    # generate the anomaly mask (short for anomask).
                    anomask_pred, anomask_pred_64 = rmp([b.clone().detach() for b in model_pred_list], emb=emb, vae_features=vae_features)
                    
                    # For tiny anomaly in VisA and MVTec-3D-AD, we use this method to prevent the values of tiny anomaly region 
                    # in mask which should be 1 but are affected by the interpolation.
                    # So we first normalize the mask to make sure the values of tiny anomaly region in mask are 1.
                    if "visa" in self.instance_data_dir or "3d" in self.instance_data_dir:
                        if torch.max(mask) == 0:
                            continue
                        mask = mask / torch.max(mask)
                        mask_64 = F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False)
                        if torch.max(mask_64) == 0:
                            continue
                        mask_64 = mask_64 / torch.max(mask_64)

                        mask_for_good = torch.zeros_like(mask)
                        mask_for_good_64 = torch.zeros_like(mask_64)
                        mask_all = torch.cat((mask, mask_for_good), dim=0)
                        mask_64 = torch.cat((mask_64, mask_for_good_64), dim=0)
                    else:
                        mask_for_good = torch.zeros_like(mask)
                        mask_all = torch.cat((mask, mask_for_good), dim=0)
                        mask_64 = F.interpolate(mask_all, size=(64, 64), mode='bilinear', align_corners=False)

                    focal = FocalLoss()                                               
                    focal_loss = focal(anomask_pred, mask_all)
                    focal_loss_64 = focal(anomask_pred_64, mask_64)
                    loss = focal_loss + focal_loss_64
                    accelerator.backward(loss)

                    optimizer_rmp.step()
                    lr_scheduler_rmp.step()
                    optimizer_rmp.zero_grad(set_to_none=self.set_grads_to_none)

                    logs = {
                        "loss:":loss.detach().item(),
                        "focal_loss":focal_loss.detach().item(),
                        "focal_loss_64":focal_loss_64.detach().item(),
                        "lr": lr_scheduler_rmp.get_last_lr()[0]}               
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # save models
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if accelerator.is_main_process:
                        if global_step % self.checkpointing_steps == 0:
                            save_path = os.path.join(self.output_dir, f"mask-checkpoint")
                            os.makedirs(save_path)
                            sub_dir = "rmp"
                            torch.save(rmp.state_dict(), os.path.join(save_path, sub_dir))

                if global_step >= self.max_train_steps:
                    break

        accelerator.wait_for_everyone()
        accelerator.end_training()