from diffusers import EulerDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler

schedulers = {
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler
}
