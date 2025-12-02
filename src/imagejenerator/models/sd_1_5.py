from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
import time
import datetime 
from imagejenerator.models.registry import register_model

from imagejenerator.core.image_generator import ImageGenerator
from imagejenerator.models.sd_schedulers import schedulers


@register_model("stable-diffusion-v1-5")
class StableDiffusion_1_5(ImageGenerator):
    """
    Concrete implementation of ImageGenerator for Stable Diffusion v1.5.

    This class handles the loading and inference of the Stable Diffusion v1.5 model
    using the Hugging Face Diffusers library. It supports custom schedulers,
    attention slicing for memory optimization, and mixed-precision inference.
    """

    def __init__(self, config):
        """
        Initializes the Stable Diffusion 1.5 generator.

        This method expands the 'prompts' list in the config to match the total
        number of images to generate (prompts * images_per_prompt) to facilitate
        batch processing.

        Args:
            config (dict): Configuration dictionary. Must include standard ImageGenerator
                           keys plus model-specific keys.
        """
        super().__init__(config)
        self.pipe = None
        self.images = None
        self.prompts = config["prompts"] * config["images_to_generate"]


    def create_pipeline(self):
        """
        Loads the Stable Diffusion pipeline and applies configurations.

        Steps taken:
        1. Loads the pipeline using `StableDiffusionPipeline.from_pretrained`.
        2. Moves the pipeline to the specific device (CPU/CUDA).
        3. Enables attention slicing if `config['enable_attention_slicing']` is True.
        4. Swaps the default scheduler if `config['scheduler']` is specified.

        Raises:
            KeyError: If specific config keys (like 'model_path') are missing.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config["model_path"],
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)

        if self.config["enable_attention_slicing"]:
            self.pipe.enable_attention_slicing()

        if self.config["scheduler"]:
            scheduler = schedulers[self.config["scheduler"]]
            self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)


    def run_pipeline_impl(self):
        """
        Executes the Stable Diffusion inference.

        Runs the pipeline within a `torch.autocast` context to ensure the correct
        precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        with autocast(self.device):    
            self.images = self.pipe(
                self.prompts, 
                height = self.config["height"], 
                width = self.config["width"],
                num_inference_steps = self.config["num_inference_steps"],
                guidance_scale = self.config["guidance_scale"],
                generator=self.generators,
            ).images


    def complete_image_generation_record_impl(self):
        """
        Implementation hook for recording extra stats.

        Currently a no-op for SD 1.5 as the base class records all necessary metadata.
        """
        pass

