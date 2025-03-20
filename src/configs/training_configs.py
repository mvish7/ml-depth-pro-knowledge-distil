import torch

training_configs = \
    {
        "num_epochs": 100,
        "warmup_steps": 0.1,
        "decay_start": 0.71,
        "lr": 1e-5,
        "device": "cuda",
        "precision": torch.float16,
        "log_interval": 10,    # How often to log batch results
        "save_checkpoint": True,
        "checkpoint_interval": 2,  # Save every N epochs
        "save_best": True,  # saving the best ckpt as per metrics on validation
        "checkpoint_dir": "checkpoints/",
        "resume_from_checkpoint": None,
        "loss_config": {
            """
            idea is to use following schedule for adpting kd and direct supervision loss weights:
            initial phase (0-30% of total): kd - 0.7, direct - 0.3
            mid phase (31-80% of total): kd - 0.5, direct - 0.5
            final phase (81 - 100% of total): kd - 0.3, direct - 0.7
            """
            "kd_weighting": 0.7,  # initial scaling factors, will be adapted in trainer as training progresses
            "gt_weight": 0.3,  # initial scaling factors, will be adapted in trainer as training progresses
            "berhu_scaling": 1.0,
            "si_scaling": 0.5,
            "fov_l1_scaling": 0.3,
            "berhu_kd_scaling": 0.5,
            "cs_kd_scaling": 0.3,
            "grad_kd_scaling": 0.3
        }
}
