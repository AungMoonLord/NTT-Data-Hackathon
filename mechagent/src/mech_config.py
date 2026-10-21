"""Global constants shared by simulation, training and inference."""
FS = 16_000                      # Hz – sample rate expected by the HF audio backbone
DURATION_S = 1.0
N_SAMPLES = int(FS * DURATION_S)
CLASSES = ["normal", "inner_race", "outer_race", "ball", "cage"]
FAULT_TYPES = CLASSES[1:]
RUL_CAP_H = 1000.0               # piece-wise RUL cap (~41.7 days)
RPM_RANGE = (900.0, 3600.0)
DEFAULT_BACKBONE = "ntu-spml/distilhubert"
