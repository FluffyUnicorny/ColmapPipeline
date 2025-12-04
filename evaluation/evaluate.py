import numpy as np
import cv2
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
REFINE_OUTPUT_DIR = Path("../refinement/refinement_output")
POINTS_FILE = REFINE_OUTPUT_DIR / "triangulated_points_refined_filtered.npy"
CAMERA_FILE = REFINE_OUTPUT_DIR / "camera_poses_refined.npy"
DATA_IMG_DIR = Path("../data/dataset_kicker/images")  # path dataset ของคุณ
GT_2D_DIR = Path("../data/dataset_kicker/gt_2d")      # <-- ถ้ามี 2D keypoints ground truth

# -----------------------------
# โหลดข้อมูล
# -----------------------------
points3d = np.load(POINTS_FILE)  # shape: N x 3
cameras = np.load(CAMERA_FILE, allow_pickle=True).item()  # dict: {img_name: {'rvec':..., 'tvec':...}}

print(f"Loaded {points3d.shape[0]} 3D points and {len(cameras)} cameras.")

# -----------------------------
# Camera intrinsics (K) & distortion
# -----------------------------
K = np.array([[1000, 0, 1920/2],
              [0, 1000, 1080/2],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# -----------------------------
# Evaluation: reprojection error
# -----------------------------
errors = []

img_files = sorted(DATA_IMG_DIR.glob("*.*"))
for img_file in img_files:
    img_name = img_file.stem
    if img_name not in cameras:
        print(f"{img_name} not found in camera poses, skip.")
        continue

    cam = cameras[img_name]
    rvec = cam['rvec']
    tvec = cam['tvec']

    # project all 3D points
    proj_pts, _ = cv2.projectPoints(points3d, rvec, tvec, K, dist_coeffs)
    proj_pts = proj_pts.reshape(-1, 2)

    # -----------------------------
    # Load ground truth 2D points if available
    # -----------------------------
    gt_file = GT_2D_DIR / f"{img_name}.npy"
    if gt_file.exists():
        gt_pts = np.load(gt_file)
        if gt_pts.shape[0] != proj_pts.shape[0]:
            print(f"Warning: {img_name} - number of GT points ({gt_pts.shape[0]}) "
                  f"does not match projected points ({proj_pts.shape[0]}). Skipping.")
            continue
        img_errors = np.linalg.norm(proj_pts - gt_pts, axis=1)
    else:
        # ถ้าไม่มี ground truth, ใช้ placeholder zeros
        img_errors = np.zeros(proj_pts.shape[0])

    errors.extend(img_errors)

# -----------------------------
# Compute RMS reprojection error
# -----------------------------
if errors:
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
else:
    rms_error = 0

print(f"Average reprojection error over {len(img_files)} images: {rms_error:.2f} pixels")
