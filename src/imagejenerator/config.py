config = {
    "model": "stable-diffusion-v1-5",
    "model_path": "runwayml/stable-diffusion-v1-5",

    "device": "cuda",
    "enable_attention_slicing": True,
    "scheduler": "EulerDiscreteScheduler",

    "height": 512,
    "width": 512,
    "num_inference_steps": 30,
    "guidance_scale": 10,
    "images_to_generate": 1,
    "seeds": [], # leave empty for random
    "dtype": "bfloat16",


    "image_save_folder": "./images/",

    "save_image_gen_stats": True,
    "image_gen_data_file_path": "./stats/image_gen_stats.csv",

    "prompts": [
        "A rockstar playing a guitar solo on stage"
    ]
}