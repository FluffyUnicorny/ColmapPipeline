import numpy as np
from pathlib import Path

POSE_DIR = Path("output_pose")
OUT_FILE = Path("camera_poses.npy")

poses = {}

rvec_files = sorted(POSE_DIR.glob("*_rvec.npy"))

print(f"Found {len(rvec_files)} camera poses")

for rvec_file in rvec_files:
    stem = rvec_file.stem.replace("_rvec", "")
    tvec_file = POSE_DIR / f"{stem}_tvec.npy"

    if not tvec_file.exists():
        print(f"Missing tvec for {stem}, skip")
        continue

    rvec = np.load(rvec_file)
    tvec = np.load(tvec_file)

    poses[stem] = {
        "rvec": rvec,
        "tvec": tvec
    }

print(f"Collected {len(poses)} poses")

np.save(OUT_FILE, poses)
print(f"Saved to {OUT_FILE.resolve()}")
