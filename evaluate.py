import os
import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from omegaconf import OmegaConf

from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
from torchvision.utils import save_image

from dataset import VWPDataset
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline
from evaluation.clip_sim import calc_clip_score
from evaluation.fid_utils import calculate_fid_given_features, inception_feature

# Move the inputs to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = get_logger(__name__)


def get_model(args: OmegaConf):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_path, subfolder="unet"
    )
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_path, subfolder="scheduler"
    )

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")

    vae.eval()
    text_encoder.eval()
    unet.eval()

    return accelerator, pipeline


def sample(args: OmegaConf) -> None:
    # assert args.test_model_file is not None, "test_model_file cannot be None"
    # assert (
    #     args.gpu_ids == 1 or len(args.gpu_ids) == 1
    # ), "Only one GPU is supported in test mode"
    test_dataset = VWPDataset("test", args)

    print(test_dataset.__len__())

    if args.start is not None:
        start = args.start
    else:
        start = 0

    if args.end is not None:
        end = args.end
    else:
        end = len(test_dataset)

    test_dataset = torch.utils.data.Subset(test_dataset, range(start, end))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1
    )

    log_dir = os.path.join(args.logdir, args.out_mode)
    save_dir_gold = os.path.join(log_dir, "gold/")
    os.makedirs(save_dir_gold, exist_ok=True)
    save_dir_pred = os.path.join(log_dir, "pred/")
    os.makedirs(save_dir_pred, exist_ok=True)

    sample_seeds = torch.randint(0, 100000, (args.num_sample_per_prompt,))
    sample_seeds = sorted(sample_seeds.numpy().tolist())

    accelerator, pipeline = get_model(args)

    generator = []
    for seed in sample_seeds:
        generator_temp = torch.Generator(device=accelerator.device)
        generator_temp.manual_seed(seed)
        generator.append(generator_temp)

    with torch.no_grad():
        predictions = []
        for batch_id, batch in enumerate(test_dataloader):
            if batch_id >= 1:
                break
            # a test sample consists of a complete story:
            # prompt for each frame
            # image for each frame
            prompts = batch["prompt"]  # list of prompts
            original_images = batch["image"]  # list of images
            sample_id = batch["sample_id"]  # story id

            assert len(original_images[0]) == len(
                prompts
            ), f"The number of prompts should be the same as the number of images, num prompts: {len(prompts)}, num images: {original_images[0].shape}"

            image_prompt = original_images  # Note: this is a dummy and won't be used for the first frames
            prev_prompt = ["" for _ in range(len(image_prompt[0]))]
            batch_generations = []

            for turn in range(len(prompts)):
                prompt = prompts[turn][0]
                if turn == 0:
                    stage = "no"  # first frame is generated without any guidance
                else:
                    stage = "auto-regressive"

                # this should be per sample, and not per batch. It is repeated autoregressively untill all frames in the story are generated
                # TODO: does the previous generated image need post-processing before being used as guidance?
                curr_gen = pipeline(
                    stage=stage,
                    prompt=prompt,
                    image_prompt=image_prompt,
                    prev_prompt=prev_prompt,
                    height=512,
                    width=512,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    num_images_per_prompt=args.num_sample_per_prompt,
                    output_type=None,
                ).images

                curr_gen = torch.from_numpy(curr_gen)  # (B,512,512,3)
                curr_gen = torch.transpose(curr_gen, 1, 3)  # (B,3,512,512)
                curr_gen = torch.transpose(curr_gen, 2, 3)  # (B,3,512,512)
                batch_generations.append(curr_gen[0])

                image_prompt = transforms.Normalize([0.5], [0.5])(
                    curr_gen.unsqueeze(1)
                )  # (B,1,3,512,512)
                prev_prompt = prompt

            original_images = [transforms.ToPILImage()(im) for im in original_images[0]]

            ori = inception_feature(
                original_images,
                device,
            )

            batch_generations = [
                transforms.ToPILImage(mode="RGB")(im) for im in batch_generations
            ]

            gen = inception_feature(
                batch_generations,
                device,
            )

            predictions.append(
                [original_images, batch_generations, ori, gen, sample_id]
            )

    for output in predictions:
        sample_id = output[4][0]
        for i, gold in enumerate(output[0]):
            gold.save(os.path.join(save_dir_gold, f"{sample_id}_{i}.jpg"))
        for i, pred in enumerate(output[1]):
            pred.save(os.path.join(save_dir_pred, f"{sample_id}_{i}.jpg"))

    all_gold_images = [elem for output in predictions for elem in output[0]]
    all_pred_images = [elem for output in predictions for elem in output[1]]
    clip_score = calc_clip_score(all_gold_images, all_pred_images)

    # if args.calculate_fid:
    ori = np.array([elem.tolist() for output in predictions for elem in output[2]])
    gen = np.array([elem.tolist() for output in predictions for elem in output[3]])
    fid = calculate_fid_given_features(ori, gen)

    with open(os.path.join(log_dir, "metrics.txt"), "w") as f:
        f.write("CLIP-I: {}".format(clip_score))
        f.write("\n")
        f.write("FID: {}".format(fid))


if __name__ == "__main__":

    config = "./config/inference_config.yml"
    args = OmegaConf.load(config)

    assert args.test_batch_size == 1, "Batch size should be 1 for inference"

    sample(args)
