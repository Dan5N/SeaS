import logging
import math
import itertools
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path

import torch
import torch.nn.functional as F
import shutil

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

check_min_version("0.19.0.dev0")
logger = get_logger(__name__)

class SeaS():
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
        self.train_text_encoder = args.train_text_encoder
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
        self.normal_token_num = args.normal_token_num
        self.anomaly_token_num = args.anomaly_token_num
        self.initializer_token = args.initializer_token
        self.tokenizer_max_length = args.tokenizer_max_length
        self.revision = args.revision
        self.text_encoder_use_attention_mask = args.text_encoder_use_attention_mask
        # ----------------------
        # optimizer hyperparameters
        # ----------------------
        self.learning_rate = args.gen_learning_rate
        self.learning_rate_text_encoder = args.encoder_learning_rate
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
        self.max_train_steps = args.gen_train_steps
        self.checkpointing_steps = args.checkpointing_steps

        # For VisA, we don't rotate the images
        if "visa" in self.instance_data_dir:
            self.rotation = False
        
    # Normalize attention maps for anomaly and normal according to their token numbers respectively.
    def normalize_attention_maps(self, cross_map_final, attn_resolution, normal_token_num, anomaly_token_num):
        normal_cross_map = cross_map_final[attn_resolution][2 : 1 + normal_token_num].mean(dim=0) if normal_token_num > 1 else cross_map_final[attn_resolution][2]
        anomaly_cross_map = cross_map_final[attn_resolution][3 + normal_token_num : 3 + normal_token_num + anomaly_token_num].mean(dim=0) if anomaly_token_num > 1 else cross_map_final[attn_resolution][3 + normal_token_num]
        normal_cross_map = (normal_cross_map - normal_cross_map.min()) / (normal_cross_map.max() - normal_cross_map.min())
        anomaly_cross_map = (anomaly_cross_map - anomaly_cross_map.min()) / (anomaly_cross_map.max() - anomaly_cross_map.min())

        return normal_cross_map, anomaly_cross_map
    
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
        args = self.args
        logging_dir = Path(self.output_dir, self.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=logging_dir)
        print(os.environ.get("ACCELERATE_FORCE_NUM_PROCESSES"))

        # Initianize the accelerator. 
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

        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.revision, use_fast=False)
        # Import correct text encoder class
        text_encoder_cls = self.import_model_class_from_model_name_or_path(self.pretrained_model_name_or_path, self.revision)

        # Load scheduler and models#
        noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.revision)
        unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet", revision=self.revision)
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision)
        
        # To save the models
        def save_model_hook(models, weights, output_dir):
            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    sub_dir = "unet"  
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    sub_dir = "text_encoder"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
        accelerator.register_save_state_pre_hook(save_model_hook)

        # Embeddings initialization.
        # Calculate the number of anomaly categories
        anomaly_categories_num = 0
        for subfolder in sorted(Path(self.instance_data_dir).iterdir()):
            if subfolder.is_dir() and subfolder.name != "good":
                anomaly_categories_num += 1
        if self.train_text_encoder:
            # Add the new placeholder token in tokenizer according to the number of anomaly categories.
            # The new tokens include normal_token_num 'ob' tokens and anomaly_token_num * anomaly_categories_num 'sks' tokens. 
            normal_token_num = self.normal_token_num
            anomaly_token_num = self.anomaly_token_num
            placeholder_tokens = [f"ob{i}" for i in range(1, normal_token_num + 1)] + [f"sks{i}" for i in range(1, anomaly_token_num * anomaly_categories_num + 1)]

            # The number of newly added tokens (placeholder_tokens)
            num_vectors = normal_token_num + anomaly_token_num * anomaly_categories_num
            if num_vectors < 1:
                raise ValueError(f"num_vectors has to be larger or equal to 1, but is {num_vectors}")

            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            if num_added_tokens != num_vectors:
                raise ValueError(
                    f"The tokenizer already contains the token {self.placeholder_token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids.
            token_ids = tokenizer.encode(self.initializer_token, add_special_tokens=False)
            token_ids_ob = tokenizer.encode("object", add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            initializer_token_id_ob = token_ids_ob[0]
            placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for index, token_id in enumerate(placeholder_token_ids):
                    if index < normal_token_num : 
                        token_embeds[token_id] = token_embeds[initializer_token_id_ob].clone()
                    else:
                        token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        if vae is not None:
            vae.requires_grad_(False)

        if not self.train_text_encoder:
            text_encoder.requires_grad_(False)

        # Freeze all parameters except for the token embeddings in text encoder
        if self.train_text_encoder:
            text_encoder.text_model.encoder.requires_grad_(False)
            text_encoder.text_model.final_layer_norm.requires_grad_(False)
            text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if accelerator.unwrap_model(unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )

        if self.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                f" {low_precision_error_string}"
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
        if not self.train_text_encoder and text_encoder is not None:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        unet.to(accelerator.device, dtype=torch.float32)
        
        logger.info(f"***** Running Generation Model *****")

        # optimizer
        optimizer_class = bnb.optim.AdamW8bit
        optimizer_unet = optimizer_class(
            itertools.chain(unet.parameters()),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )
        
        optimizer_text_encoder = optimizer_class(
            itertools.chain(text_encoder.parameters()),
            lr=self.learning_rate_text_encoder,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        # scheduler
        lr_scheduler_unet = get_scheduler(self.lr_scheduler, optimizer=optimizer_unet, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,  num_cycles=self.lr_num_cycles, power=self.lr_power)
        lr_scheduler_text_encoder = get_scheduler(self.lr_scheduler, optimizer=optimizer_text_encoder, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,  num_cycles=self.lr_num_cycles, power=self.lr_power)

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
            normal_token_num=normal_token_num, 
            anomaly_token_num=anomaly_token_num  
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True, 
            collate_fn=lambda examples: collate_fn(examples, self.with_Ni_Alignment), 
            num_workers=self.dataloader_num_workers
        )

        # Prepare everything with our `accelerator`.
        if self.train_text_encoder:
            unet, text_encoder, optimizer_unet, train_dataloader, lr_scheduler_unet, optimizer_text_encoder, lr_scheduler_text_encoder = accelerator.prepare(
                unet, text_encoder, optimizer_unet, train_dataloader, lr_scheduler_unet, optimizer_text_encoder, lr_scheduler_text_encoder
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
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

        args_dict = vars(args)
        if "placeholder_token" in args_dict:
            del args_dict["placeholder_token"]
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=args_dict)
            
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps
        logger.info(f"***** Running generation training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        logger.info(f"  placeholder_token = {placeholder_tokens}")
        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.num_train_epochs):
            #forward#
            unet.train()
            if self.train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)               
                # Skip steps until we reach the resumed step
                with accelerator.accumulate(unet):         
                    if self.with_Ni_Alignment:
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (self.train_batch_size * 2,), device=accelerator.device)
                    else:
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (self.train_batch_size,), device=accelerator.device)
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
                    bsz, channels, height, width = model_input.shape
                    # Add noise to the model input according to the noise magnitude at each timestep
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.encode_prompt(text_encoder,batch["input_ids"],batch["attention_mask"],text_encoder_use_attention_mask=self.text_encoder_use_attention_mask)

                    if accelerator.unwrap_model(unet).config.in_channels == channels * 2: #inchannels = 4
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    class_labels = None

                    # Predict the noise residual
                    model_pred, model_pred_list, emb, cross_map_final = unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels)
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon": 
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    # Normal Image Alignent Loss
                    if self.with_Ni_Alignment:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_normal = torch.chunk(model_pred, 2, dim=0)
                        target, target_normal = torch.chunk(target, 2, dim=0) 
                        for key, _ in cross_map_final.items():
                            cross_map_final[key], _ = torch.chunk(cross_map_final[key], 2, dim=1) 
                        # Traning Loss of SD for normal image
                        na_loss = F.mse_loss(model_pred_normal.float(), target_normal.float(), reduction="mean")

                    # Decoupled Anomaly Alignment Loss
                    da_loss = {"32":None, "16":None}
                    attn_resolutions = ["32", "16"]

                    ground_truth = mask.clone()
                    ground_truth = F.interpolate(ground_truth, size=(64, 64), mode='bilinear', align_corners=False)
                    ground_truth = ground_truth.to(model_input.device)   
                    for attn_resolution in attn_resolutions:
                        ground_truth = F.max_pool2d(ground_truth, kernel_size=2, stride=2)
                        ground_truth_mask = ground_truth==1
                        # We apply this to prevent all zero mask because of the values of tiny anomaly region in VisA and MVTec-3D-AD
                        # in mask which should be 1 but are affected by the interpolation.
                        if "visa" in str(self.instance_data_dir) or "3d" in str(self.instance_data_dir):
                            ground_truth_mask = ground_truth > 0
                        cross_map_final[attn_resolution] = cross_map_final[attn_resolution].squeeze(-1).to(model_input.device)
                        normal_cross_map, anomaly_cross_map = self.normalize_attention_maps(cross_map_final, attn_resolution, normal_token_num, anomaly_token_num)
                        # Cross attention map
                        da_loss[attn_resolution] = (((anomaly_cross_map.unsqueeze(1).float() - ground_truth.float())**2)).mean() 
                        normal_cross_map = normal_cross_map.unsqueeze(1)
                        da_loss_normal = F.mse_loss(normal_cross_map[ground_truth_mask].float(), (1-ground_truth)[ground_truth_mask].float(), reduction="mean")
                        da_loss[attn_resolution] += da_loss_normal
                    # Traning Loss of SD for anomaly images
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 
                    loss += sum(da_loss.values())

                    # Normal Image Alignent Loss
                    if self.with_Ni_Alignment: 
                        loss += na_loss
                    
                    accelerator.backward(loss)
                    optimizer_unet.step()
                    lr_scheduler_unet.step()
                    optimizer_unet.zero_grad(set_to_none=self.set_grads_to_none)
                    optimizer_text_encoder.step()
                    lr_scheduler_text_encoder.step()
                    optimizer_text_encoder.zero_grad(set_to_none=self.set_grads_to_none)
                    
                    logs = {"loss": loss.detach().item(),
                        "na_loss": na_loss.detach().item(),
                        "da_loss_32":da_loss["32"].detach().item(), 
                        "da_loss_16":da_loss["16"].detach().item(),
                        "lr": lr_scheduler_unet.get_last_lr()[0]}                
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # save models
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if accelerator.is_main_process:
                        if global_step % self.checkpointing_steps == 0:
                            save_path = os.path.join(self.output_dir, f"generation-checkpoint")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            if self.train_text_encoder:
                                sub_dir = "tokenizer"
                                tokenizer.save_pretrained(os.path.join(save_path, sub_dir))
                           
                            # save model_index.json of stable diffusion
                            model_index_path = os.path.join(self.pretrained_model_name_or_path, "model_index.json")
                            new_model_index_path = os.path.join(save_path, "model_index.json")
                            shutil.copy(model_index_path, new_model_index_path)

                            # save feature_extractor of stable diffusion
                            feature_extractor_path = os.path.join(self.pretrained_model_name_or_path, "feature_extractor")
                            new_feature_extractor_path = os.path.join(save_path, "feature_extractor")
                            shutil.copytree(feature_extractor_path, new_feature_extractor_path)

                            # save scheduler of stable diffusion
                            scheduler_path = os.path.join(self.pretrained_model_name_or_path, "scheduler")
                            new_scheduler_path = os.path.join(save_path, "scheduler")
                            shutil.copytree(scheduler_path, new_scheduler_path)

                if global_step >= self.max_train_steps:
                    break

        accelerator.wait_for_everyone()
        accelerator.end_training()