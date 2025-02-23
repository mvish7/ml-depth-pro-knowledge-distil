import torch

training_configs = \
    {
        "num_epochs": 100,
        "warmup_steps": 0.1,
        "decay_start": 0.71,
        "lr": 1e-5,
        "device": "cuda",
        "precision": torch.float16
}