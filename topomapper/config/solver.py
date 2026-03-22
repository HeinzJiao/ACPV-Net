from yacs.config import CfgNode as CN

SOLVER = CN()

# === Optimizer ===
SOLVER.OPTIMIZER = "ADAM"
SOLVER.BASE_LR = 0.01
SOLVER.BETAS = (0.9, 0.999)
SOLVER.MOMENTUM = 0.9
SOLVER.WEIGHT_DECAY = 0.0002
SOLVER.AMSGRAD = False
SOLVER.BACKBONE_LR_MULT = 1.0  # Keep consistent with `make_optimizer`.

# === Learning Rate Scheduler ===
SOLVER.LR_SCHEDULER = 'linear_poly'  # Supported: `linear_poly`, `multistep`, `ADAMcos`.
SOLVER.WARMUP_ITERS = 1500
SOLVER.TOTAL_ITERS = 160000
SOLVER.POLY_POWER = 1.0
SOLVER.ETA_MIN = 0.0
SOLVER.STEPS = (25,)  # Used by MultiStepLR.
SOLVER.GAMMA = 0.1    # Learning rate decay factor for MultiStepLR.
SOLVER.STATIC_STEP = 25  # Used by ADAMcos.

# === Training Management ===
SOLVER.IMS_PER_BATCH = 6
SOLVER.MAX_EPOCH = 30
SOLVER.CHECKPOINT_PERIOD = 1

SOLVER.LDM_LR_ENABLE = False  # Whether to use a separate learning rate for the LDM branch.
SOLVER.LDM_LR = 2e-6  # Learning rate for the LDM branch when enabled.
SOLVER.LDM_WEIGHT_DECAY = 0.0  # Weight decay for the LDM branch.
SOLVER.LDM_PARAM_NAME_KEYWORDS = ["ldm", "diffusion_unet", "denoiser", "model"]  # Parameter name keywords for the LDM branch.
