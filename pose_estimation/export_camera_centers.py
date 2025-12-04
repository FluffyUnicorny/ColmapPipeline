import numpy as np
import cv2
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
POSE_FILE = Path("camera_poses.npy")   # ไฟล์ที่คุณใช้ check_camera_poses
OUT_FILE = Path("camera_centers.npy")

# -----------------------------
# Load camera poses
# camera_poses = {
#   img_name: {"rvec": (3,1), "tvec": (3,1)}
# }
# -----------------------------
camera_poses = np.load(POSE_FILE, allow_pickle=True).item()

camera_centers = {}

for name, pose in camera_poses.items():
    rvec = pose["rvec"]
    tvec = pose["tvec"]

    # rvec -> Rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Camera center: C = -R^T t
    C = -R.T @ tvec

    camera_centers[name] = C.reshape(3)

    print(f"{name} -> Camera center: {C.ravel()}")

# -----------------------------
# Save result
# -----------------------------
np.save(OUT_FILE, camera_centers)

print("✅ Saved camera centers to:", OUT_FILE)
print("Total cameras:", len(camera_centers))
