# Image Jenerator

**A Flexible and Configurable Image Generation Suite**

`Image Jenerator` is an extensible Python framework designed to run various image generation models (like Stable Diffusion) through a unified, standardized pipeline. 

It takes care of the setup such as device management (CUDA/CPU), mixed-precision handling, and random seed generation. It also automatically generates performance logs including generation time, which you can use for benchmarking, and saves these to CSV.

## Requirements

* Python 3.10+
* GPU and CUDA drivers (or equivalent) are recommended. Image Jenerator supports CPU, but don't blame me if your system crashes or your images take half an hour to generate.

## Installation

Not on PyPI yet, so...

```sh
git clone https://github.com/WarrenDavies/image-jenerator.git
cd imagejenerator
python -m venv venv
(Linux/macOS) source venv/bin/activate
(Windows) .\venv\Scripts\activate
pip install -e .
```

## Config

Create a config dictionary with the following keys (default parameters are located at: `src/imagejenerator/config.py`):

| **Setting** | **Type** | **Description** |
| ----------- | -------- | --------------- |
| `model` | `str` | The registered model name to use (e.g., `"stable-diffusion-v1-5"`). |
| `model_path` | `str` | Local path or Hugging Face repository ID for the model weights. |
| `device` | `str` | Compute device (`"detect"`, `"cuda"`, or `"cpu"`). |
| `dtype` | `str` | Tensor precision (`"detect"`, `"bfloat16"`, `"float16"`, or `"float32"`). |
| `scheduler` | `str` | The name of the scheduler/sampler to use (must be a key in `sd_schedulers.py`). |
| `enable_attention_slicing` | `bool` | If True, enables attention slicing to reduce VRAM usage. |
| `height` | `int` | The height of the generated image in pixels. |
| `width` | `int` | The width of the generated image in pixels. |
| `num_inference_steps` | `int` | The number of diffusion steps to run. |
| `guidance_scale` | `float` | Classifier-free guidance scale (CFG). Higher values increase adherence to the prompt. |
| `prompts` | `list[str]` | The list of prompts to generate - more than one prompt will be batched so watch your VRAM! |
| `images_to_generate` | `int` | The number of images to generate per prompt. The batch size will be number of prompts * images_to_generate |
| `seeds` | `list[int]` | List of random seeds. Leave empty (`[]`) for automatic random generation. |
| `image_save_folder` | `str` | Output directory for image files. |
| `save_image_gen_stats` | `bool` | If True, saves detailed metadata to the statistics CSV file. |
| `image_gen_data_file_path` | `str` | Path to the output statistics CSV file. |

## Generating images

You can run the example script:

```sh
python src/imagejenerator/examples/generate_image.py
```

Which simply contains:

```py
from imagejenerator.models import registry
from imagejenerator.config import config

image_generator = registry.get_model_class(config)
image_generator.generate_image()
```

Your images and a statistics CSV file will be saved to the configured output folders.

## Architecture and Extensibility

The core of `Image Jenerator` is built around a base ImageGenerator class and a model registry, making it easy to add new models (like SDXL, Stable Cascade, or custom pipelines) without modifying the core logic.

### 1. Abstract Base Class (`ImageGenerator`)

All model pipelines inherit from `ImageGenerator`, which defines the standard workflow methods and parameters common to all models.

### 2. Model Registration

Each model (or group of models that require the same diffusers class) use their own model class, with the `@register_model` decorator applied to it. This automatically adds model classes to the registry, so that they can be specified in the config.

### 3. Data Logging (`ImageGenerationRecord`)

The `ImageGenerationRecord` class uses Python dataclass features to structure and save the generation parameters (prompt, seed, model, timings, etc.) to a specified CSV file. So you can track and analyse your experiments with different inference steps, prompt lengths, and so on.

