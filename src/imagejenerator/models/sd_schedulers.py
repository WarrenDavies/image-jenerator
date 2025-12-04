"""
Defines a mapping of human-readable names to specific scheduler classes
from the Hugging Face Diffusers library.

This allows configuration files to specify the desired sampler/scheduler
using simple strings, which the image generator pipeline can then use
to instantiate the correct class.
"""
from diffusers import EulerDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler

schedulers = {
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler
}
