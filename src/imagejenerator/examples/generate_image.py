from imagejenerator.models import registry
from imagejenerator.config import config

image_generator = registry.get_model_class(config)
image_generator.generate_image()

