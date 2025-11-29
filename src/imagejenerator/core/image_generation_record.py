from dataclasses import dataclass, fields, field
import csv
import os
from typing import List

@dataclass
class ImageGenerationRecord():
    image_gen_data_file_path: str = ""
    filename: str = ""
    timestamp: str = ""
    model: str = ""
    device: str = ""
    dtype: str = ""
    prompt: str = ""
    seed: str = ""
    height: int = 0
    width: int = 0
    inf_steps: int = 0
    guidance_scale: float = 0
    total_generation_time: float = 0
    generation_time_per_image: float = 0
    image_rating: float = -1
    fields_to_not_save: list[str] = field(default_factory=list)


    def __post_init__(self):
        self.fields_to_not_save += ["image_gen_data_file_path", "fields_to_not_save", "path_to_csv"]


    def create_data_row(self):
        row = []
        for field in fields(self):
            if field.name in self.fields_to_not_save:
                continue
            field_value = getattr(self, field.name)

            if isinstance(field_value, float):
                row.append(f"{field_value:.2f}")
            else:
                row.append(field_value)
        return row


    def append_data_to_csv(self):
        with open(self.image_gen_data_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.create_data_row())


    def create_header_row(self):
        headers = []
        for header in fields(self):
            if header.name in self.fields_to_not_save:
                continue
            headers.append(header.name)
        return headers


    def create_new_stats_file(self):
        with open(self.image_gen_data_file_path, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.create_header_row())


    def save_data(self):
        if not os.path.exists(self.image_gen_data_file_path):
            self.create_new_stats_file()

        self.append_data_to_csv()
        
