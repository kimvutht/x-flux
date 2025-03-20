import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import trange
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from image_datasets.textual_inversion_dataset import textual_inversion_dataset_loader
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)
from image_datasets.dataset import loader
from PIL import Image, ExifTags
from contextlib import nullcontext
from transformers.integrations import is_deepspeed_zero3_enabled

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

import deepspeed

class FluxModel(torch.nn.Module):
    def __init__(self, name: str, device, offload: bool, is_schnell: bool, train_t5: bool = False):
        super().__init__()
        
        self.device = torch.device(device)
        self.offload = offload
        
        self.t5 = load_t5(device, max_length=256 if is_schnell else 512)
        self.clip = load_clip(device)
        self.model = load_flow_model2(name, device="cpu")
        self.vae = load_ae(name, device="cpu" if offload else device)
        
        self.vae.requires_grad_(False)
        self.clip.requires_grad_(False)
        self.model.requires_grad_(False)
        
        self.clip.hf_module.requires_grad_(False)
        # self.clip.hf_module.text_model.embeddings.position_embedding.requires_grad_(False)
        self.clip.hf_module.get_input_embeddings().requires_grad_(True)
        
        self.t5.hf_module.requires_grad_(False)
        if train_t5:
            self.t5.hf_module.get_input_embeddings().requires_grad_(True)
        
    def forward(
        self,
        prompt,
        guidance,
        seed,
        num_steps=50,
        width=1024,
        height=1024,
        # controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        # control_weight = 0.9,
        neg_prompt="",
        # image_proj=None,
        # neg_image_proj=None,
        # ip_scale=1.0,
        # neg_ip_scale=1.0,
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            
            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                # image_proj=image_proj,
                # neg_image_proj=neg_image_proj,
                # ip_scale=ip_scale,
                # neg_ip_scale=neg_ip_scale,
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.vae.decoder.to(x.device)
            x = unpack(x.to(dtype=torch.bfloat16), height, width)
            x = self.vae.decode(x)
            self.offload_model_to_cpu(self.vae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()



def get_models(name: str, device, offload: bool, is_schnell: bool, train_t5: bool = False):
    model = FluxModel(name, device, offload, is_schnell, train_t5=train_t5)
    return model, model.model, model.vae, model.t5, model.clip
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def tokenizer_init(args, embdder_list: list):
    # Add the placeholder token in tokenizer_1
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens
    for embdder in embdder_list:
        tokenizer, text_encoder = embdder.tokenizer, embdder.hf_module
        
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != args.num_vectors:
            raise ValueError(
                f"The tokenizer ({tokenizer.__class__.__name__}) already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        
        
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # with deepspeed.zero.GatheredParameters(text_encoder.get_input_embeddings().weight, modifier_rank=0, enabled=is_deepspeed_zero3_enabled()):
         # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()
                # assert token_id != initializer_token_id, "The placeholder token should not be the same as the initializer token."
                # assert token_id == len(token_embeds), "The token_id should be equal to the length of the token embeddings."
                # token_embeds = torch.cat((token_embeds, token_embeds[initializer_token_id].clone().unsqueeze(0)), dim=0)
    
    return placeholder_tokens

def restore_origin_text_encoder_embeddings(accelerator, placeholder_tokens, embdder_list: list, orig_embeds_params: list):
    
    for orig_embeds_index, (tokenizer, text_encoder) in enumerate(zip([embdder.tokenizer for embdder in embdder_list], [embdder.hf_module for embdder in embdder_list])):
        # Let's make sure we don't update any embedding weights besides the newly added token
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        
        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(text_encoder.get_input_embeddings().weight, modifier_rank=0, enabled=is_deepspeed_zero3_enabled()):
                text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[orig_embeds_index][index_no_updates]

def main():

    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    model, dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell, train_t5=args.train_t5)

    train_embdders = [clip] + ([t5] if args.train_t5 else [])
    
    # Add token
    placeholder_tokens = tokenizer_init(args, train_embdders)
    
    model = model.to(weight_dtype)
    clip.train()
    if args.train_t5:
        t5.train()
    
    
    optimizer_cls = torch.optim.AdamW
    
    parameters =  [p for embdder in train_embdders for p in embdder.hf_module.get_input_embeddings().parameters()]
    
    logger.info(f'** Model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000000}M')
    logger.info(f"** Train parameters: {sum(p.numel() for p in parameters) / 1_000_000:.2f}M (CLIP{'+T5' if args.train_t5 else ''})")
    
    assert sum([p.numel() for p in model.parameters() if p.requires_grad]) == sum([p.numel() for p in parameters if p.requires_grad]), 'train parameters mismatch'
    
    optimizer = optimizer_cls(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = textual_inversion_dataset_loader(**args.data_config, placeholder_token=args.placeholder_token)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            dit_state = torch.load(os.path.join(args.output_dir, path, 'dit.bin'), map_location='cpu')
            dit_state2 = {}
            for k in dit_state.keys():
                dit_state2[k[len('module.'):]] = dit_state[k]
            dit.load_state_dict(dit_state2)
            optimizer_state = torch.load(os.path.join(args.output_dir, path, 'optimizer.bin'), map_location='cpu')['base_optimizer_state']
            optimizer.load_state_dict(optimizer_state)

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    
    model, optimizer, _, lr_scheduler = accelerator.prepare(
        model, optimizer, deepcopy(train_dataloader), lr_scheduler
    )
    
    # keep original embeddings as reference
    with deepspeed.zero.GatheredParameters([embdder.hf_module.get_input_embeddings().weight for embdder in train_embdders], modifier_rank=None, enabled=is_deepspeed_zero3_enabled()):
        orig_embeds_params = [embdder.hf_module.get_input_embeddings().weight.data.clone() for embdder in train_embdders]

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                img, prompts = batch
                # Encode the image, this don't need gradients because we are not training the VAE
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(weight_dtype))
                
                # Prepare the inputs for the model
                inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=accelerator.device))
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                # Predict the noise residual and compute loss
                model_pred = dit(img=x_t.to(weight_dtype),
                                img_ids=inp['img_ids'].to(weight_dtype),
                                txt=inp['txt'].to(weight_dtype),
                                txt_ids=inp['txt_ids'].to(weight_dtype),
                                y=inp['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),)

                #loss = F.mse_loss(model_pred.float(), (x_1 - x_0).float(), reduction="mean")
                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Restore the original embeddings, as we only train the new token's embeddings
                restore_origin_text_encoder_embeddings(
                    accelerator, placeholder_tokens, train_embdders, orig_embeds_params
                )
                
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        torch.save(dit.state_dict(), os.path.join(save_path, 'dit.bin'))
                        torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.bin'))
                        #accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    logger.info(f"Running validation: {args.validation_prompt}")
                    for i in trange(args.num_validation_images, desc="Validation", disable=not accelerator.is_local_main_process):
                        image = model(args.validation_prompt, guidance=args.guidance, seed=random.randint(0, 1000000))
                        if not os.path.exists(os.path.join(args.output_dir, "samples")):
                            os.mkdir(os.path.join(args.output_dir, "samples"))
                        image.save(os.path.join(args.output_dir, "samples", f"image_{global_step}_{i}.png"))


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
