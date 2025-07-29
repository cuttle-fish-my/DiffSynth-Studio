import torchvision.transforms.functional
from torchvision.transforms import InterpolationMode

from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
from einops import rearrange
from torchvision.utils import draw_bounding_boxes
from lightning.pytorch.utilities import grad_norm

import pytorch_lightning as pl

pl.seed_everything(42, workers=True)

os.environ["TOKENIZERS_PARALLELISM"] = "True"
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s")


def token_loss(attn_map, seg_mask):
    return (1 - (attn_map * seg_mask).sum(dim=(2, 3)) / (attn_map.sum(dim=(2, 3)) + 1e-5)).mean()



class LightningModel(LightningModelForT2ILoRA):
    def __init__(
            self,
            torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
            learning_rate=1e-4, use_gradient_checkpointing=True,
            lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="kaiming",
            pretrained_lora_path=None,
            state_dict_converter=None, quantize=None,
            mask_loss_weight=1.0, ce_loss_weight=1.0, token_loss_weight=1.0
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing,
                         state_dict_converter=state_dict_converter)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)
        if preset_lora_path is not None:
            preset_lora_path = preset_lora_path.split(",")
            for path in preset_lora_path:
                model_manager.load_lora(path)

        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.pipe.dit.eval()

        if quantize is not None:
            self.pipe.dit.quantize()

        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.freeze_parameters()
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
            state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format
        )

        self.mask_loss_weight = mask_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.token_loss_weight = token_loss_weight

    def training_step(self, batch, batch_idx):
        text, image = batch["text"], batch["image"]

        entity_prompts = []
        entity_masks = []
        entity_cateogries = []
        for o, iii in enumerate(batch["entity_prompt"]):
            if iii[0] != '':
                entity_prompts.append(iii[0])
                entity_masks.append(batch["entity_mask"][o])

        seg_mask = batch["masks"]
        seg_mask = torchvision.transforms.functional.resize(seg_mask, size=(64, 64),
                                                            interpolation=InterpolationMode.NEAREST)

        height, width = 1024, 1024

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        prompt_emb_nega = None  # self.pipe.encode_prompt( "", positive=False, t5_sequence_length=77)
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.pipe.prepare_eligen(prompt_emb_nega,
                                                                                            entity_prompts,
                                                                                            entity_masks, width, height,
                                                                                            512, False, False, 3.5, True
                                                                                            )

        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        noise_pred, joint_attn, single_attn = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **eligen_kwargs_posi,
            use_gradient_checkpointing=self.use_gradient_checkpointing, conditioning=3.5,
            return_attn_maps=True, mask_gt=seg_mask
        )

        loss_mse = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss_mse = loss_mse * self.pipe.scheduler.training_weight(timestep)

        # AM loss

        ce_loss = torch.nn.functional.cross_entropy
        loss_ce = ce_loss(joint_attn, seg_mask) + ce_loss(single_attn, seg_mask)
        loss_ce = loss_ce * self.pipe.scheduler.training_weight(timestep)
        loss_token = token_loss(joint_attn, seg_mask) + token_loss(single_attn, seg_mask)
        loss_token = loss_token * self.pipe.scheduler.training_weight(timestep)

        loss = loss_mse + self.mask_loss_weight * (self.ce_loss_weight * loss_ce + self.token_loss_weight * loss_token)

        # Record log
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train/lr", current_lr, prog_bar=True, logger=True)
        self.log("train/mse_loss", loss_mse, prog_bar=True, logger=True)
        self.log("train/ce_loss", loss_ce, prog_bar=True, logger=True)
        self.log("train/token_loss", loss_token, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text = batch["text"]
        bboxes = batch["bboxes"] * 1024
        entity_prompts = []
        entity_masks = []
        for o, iii in enumerate(batch["entity_prompt"]):
            if iii[0] != '':
                entity_prompts.append(iii[0])
                entity_masks.append(batch["entity_mask"][o])
        entity_categories = []
        for iii in batch["categories"]:
            entity_categories.append(iii[0])

        self.pipe.device = self.device
        height, width = 1024, 1024

        images = []
        joint_attns = []
        single_attns = []
        for i in range(3):
            image, joint_attn, single_attn = self.pipe(
                prompt=text,
                cfg_scale=3.0,
                num_inference_steps=50,
                embedded_guidance=3.5,
                height=height,
                width=width,
                eligen_entity_prompts=entity_prompts,
                eligen_entity_masks=entity_masks,
                seed=i,
                return_attn_maps=True,
                # progress_bar_cmd=lambda x: x  # disable progress bar in validation step
            )
            images.append(image)
            joint_attns.append(joint_attn)
            single_attns.append(single_attn)

        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Convert PIL images to numpy arrays
        # img_arrays = [np.array(img) for img in images]
        # combined_image = np.concatenate(img_arrays, axis=1)  # Shape: (H, 3*W, C)
        # combined_pil = Image.fromarray(combined_image)
        img_tensors = [torchvision.transforms.functional.to_tensor(img) for img in images]
        img_tensors = [draw_bounding_boxes(img, bboxes[0], width=3, labels=entity_categories) for img in img_tensors]
        combined_tensor = torch.cat(img_tensors, dim=2)  # Shape: (3, H, 3*W)
        combined_pil = torchvision.transforms.functional.to_pil_image(combined_tensor.float())

        joint_attns = torch.cat(joint_attns, dim=0)  # Shape: (3, n_obj, H, W)
        joint_attns = rearrange(joint_attns, 'b n h w -> (n h) (b w)')  # Shape: (n_obj * H, 3 * W)
        joint_attns = torchvision.transforms.functional.to_pil_image(joint_attns.float())

        single_attns = torch.cat(single_attns, dim=0)  # Shape: (3, n_obj, H, W)
        single_attns = rearrange(single_attns, 'b n h w -> (n h) (b w)')
        single_attns = torchvision.transforms.functional.to_pil_image(single_attns.float())

        self.logger.log_image(key=f"val/joint_attn_{batch_idx}", images=[joint_attns], step=self.global_step)
        self.logger.log_image(key=f"val/single_attn_{batch_idx}", images=[single_attns], step=self.global_step)
        self.logger.log_image(key=f"val/image_{batch_idx}", images=[combined_pil], step=self.global_step)

        # Save images
        return {
            "nothing": 0
        }

    def on_before_optimizer_step(self, optimizer):

        # Calculate and log gradient norm
        grad_norm_dict = grad_norm(self.pipe.denoising_model(), norm_type=2)
        avg_grad_norm = sum(grad_norm_dict.values()) / len(grad_norm_dict)
        # self.log_dict(grad_norm_dict, logger=True)
        self.log("train/grad_norm", avg_grad_norm, prog_bar=True, logger=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_text_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder/model.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_2_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained t5 text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder_2`.",
    )
    parser.add_argument(
        "--pretrained_dit_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained dit model. For example, `models/FLUX/FLUX.1-dev/flux1-dev.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vae model. For example, `models/FLUX/FLUX.1-dev/ae.safetensors`.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--align_to_opensource_format",
        default=False,
        action="store_true",
        help="Whether to export lora files aligned with other opensource format.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["float8_e4m3fn"],
        help="Whether to use quantization when training the model, and in which format.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser.add_argument(
        "--mask_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ce_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--token_loss_weight",
        type=float,
        default=1.0,
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype={"32": torch.float32, "bf16": torch.bfloat16}.get(args.precision, torch.float16),
        pretrained_weights=[args.pretrained_dit_path, args.pretrained_text_encoder_path,
                            args.pretrained_text_encoder_2_path, args.pretrained_vae_path],
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else None,
        quantize={"float8_e4m3fn": torch.float8_e4m3fn}.get(args.quantize, None),
        mask_loss_weight=args.mask_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        token_loss_weight=args.token_loss_weight,
    )
    launch_training_task(model, args)
