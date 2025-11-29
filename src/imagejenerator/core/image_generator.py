from abc import ABC, abstractmethod
import datetime
import time
import os
import random
import torch

from imagejenerator.core.image_generation_record import ImageGenerationRecord
from imagejenerator.config import config


DTYPES_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ImageGenerator(ABC):

    def __init__(self, config = config):
        self.config = config
        self.pipe = None
        self.images = None
        self.image_generation_record = ImageGenerationRecord()
        self.save_timestamp = None
        self.prompts = []
        self.dtype = None
        self.device = None
        self.seeds = config["seeds"]
        self.generators = []
        self.detect_device_and_dtype()
        self.create_generators()


    def detect_device_and_dtype(self):
        if self.config["device"] == "detect":
            self.set_device()
        else:
            self.device = self.config["device"]

        self.set_dtype()
        

    def set_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


    def set_dtype(self):
        if self.config["dtype"] == "detect":
            if self.device == "cuda":
                self.dtype = torch.bfloat16
                self.config["dtype"] = "bfloat16"
            else:
                self.dtype = torch.float32
                self.config["dtype"] = "float32"
            return
        
        self.dtype = DTYPES_MAP[self.config["dtype"]]


    def create_generators(self):
        if not self.seeds:
            batch_size = len(self.config["prompts"]) * self.config["images_to_generate"]
            self.seeds = [self.create_random_seed() for i in range(batch_size)]
                
        self.generators = [
            torch.Generator(device=self.device).manual_seed(seed)
            for seed in self.seeds
        ]


    @staticmethod
    def create_random_seed(size: int = 32) -> int:
        return random.randint(0, (2**size) - 1)


    @abstractmethod
    def create_pipeline(self):
        pass


    def run_pipeline(self):
        start_time = time.time()
        self.run_pipeline_impl()
        end_time = time.time()
        self.image_generation_record.total_generation_time = end_time - start_time
        self.image_generation_record.generation_time_per_image = (
            self.image_generation_record.total_generation_time / (self.config["images_to_generate"] * len(self.prompts))
        )
        print("Generation time: ", str(self.image_generation_record.total_generation_time))
        print("time per image: ", str(self.image_generation_record.generation_time_per_image))


    @abstractmethod
    def run_pipeline_impl(self):
        pass
    

    def generate_image(self):
        self.create_pipeline()
        self.run_pipeline()
        self.save_image()
        if self.config["save_image_gen_stats"]:
            self.save_image_gen_stats()


    def save_image(self):
        self.save_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for i, image in enumerate(self.images):
            file_name = f"{self.save_timestamp}_no{i}.png"
            save_path = os.path.join(self.config["image_save_folder"], file_name)
            image.save(save_path)


    def complete_image_generation_record(self, prompt, i):
        self.image_generation_record.image_gen_data_file_path = self.config["image_gen_data_file_path"]
        self.image_generation_record.filename = f"{self.save_timestamp}_no{i}.png"
        self.image_generation_record.timestamp = self.save_timestamp
        self.image_generation_record.model = self.config["model"]
        self.image_generation_record.device = self.device
        self.image_generation_record.dtype = self.config["dtype"]
        self.image_generation_record.prompt = prompt
        self.image_generation_record.seed = self.seeds[i]
        self.image_generation_record.height = self.config["height"]
        self.image_generation_record.width = self.config["width"]
        self.image_generation_record.inf_steps = self.config["num_inference_steps"]
        self.image_generation_record.guidance_scale = self.config["guidance_scale"]
        self.complete_image_generation_record_impl()


    @abstractmethod
    def complete_image_generation_record_impl(self):
        # Model classes can add stats unique to them with this method
        pass


    def save_image_gen_stats(self):
        all_prompts_used = self.config["prompts"] * self.config["images_to_generate"]
        for i, prompt in enumerate(all_prompts_used):
            self.complete_image_generation_record(prompt, i)
            self.image_generation_record.save_data()

