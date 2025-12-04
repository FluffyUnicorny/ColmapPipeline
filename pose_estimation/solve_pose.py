import cv2
import numpy as np
from pathlib import Path
import sys

# เพิ่ม parent folder ให้ import shared ได้
sys.path.append(str(Path(__file__).parent.parent))

# -----------------------------
# CONFIG
# -----------------------------
DATA_IMG_DIR = Path("../data/dataset_kicker/images")       # ใส่ path dataset จริง
POINTS3D_FILE = Path("../triangulation/triangulated_points.npy")
DESCS3D_FILE = Path("../triangulation/triangulated_desc.npy")
OUT_POSE_DIR = Path("output_pose")
OUT_POSE_DIR.mkdir(exist_ok=True)

# -----------------------------
# โหลด 3D points + descriptors
# -----------------------------
points3d = np.load(POINTS3D_FILE)      # shape: N x 3
desc3d = np.load(DESCS3D_FILE)         # shape: N x 32 (ORB descriptor)

print(f"Loaded {points3d.shape[0]} 3D points with descriptors.")

# -----------------------------
# ORB detector + matcher
# -----------------------------
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# -----------------------------
# Loop ภาพทั้งหมด
# -----------------------------
for img_file in DATA_IMG_DIR.glob("*.*"):  # *.png หรือ *.jpg
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Cannot read {img_file.name}, skip.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect 2D keypoints + descriptors
    kps2d, des2d = orb.detectAndCompute(gray, None)
    if des2d is None or len(kps2d) == 0:
        print(f"No descriptors in {img_file.name}, skip.")
        continue

    # match 3D descriptors ↔ 2D descriptors
    matches = bf.match(desc3d, des2d)
    matches = sorted(matches, key=lambda m: m.distance)[:500]  # เลือก 500 คู่ที่ดีที่สุด
    if len(matches) < 4:
        print(f"Not enough matches for {img_file.name}, skip.")
        continue

    # สร้าง array สำหรับ solvePnP
    pts3d_matched = np.array([points3d[m.queryIdx] for m in matches], dtype=np.float32)
    pts2d_matched = np.array([kps2d[m.trainIdx].pt for m in matches], dtype=np.float32)

    # -----------------------------
    # Camera intrinsics (ใส่ K จริงถ้ามี)
    # -----------------------------
    K = np.array([[1000, 0, img.shape[1]/2],
                  [0, 1000, img.shape[0]/2],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    # -----------------------------
    # solvePnP
    # -----------------------------
    success, rvec, tvec = cv2.solvePnP(pts3d_matched, pts2d_matched, K, dist_coeffs)
    if not success:
        print(f"solvePnP failed for {img_file.name}")
        continue

    print(f"{img_file.name} --> rvec: {rvec.ravel()}, tvec: {tvec.ravel()}")

    # -----------------------------
    # save pose
    # -----------------------------
    np.save(OUT_POSE_DIR / f"{img_file.stem}_rvec.npy", rvec)
    np.save(OUT_POSE_DIR / f"{img_file.stem}_tvec.npy", tvec)

    # -----------------------------
    # visualize reprojected 3D points
    # -----------------------------
    
    img_vis = img.copy()
    proj_pts, _ = cv2.projectPoints(pts3d_matched, rvec, tvec, K, dist_coeffs)
    for uv in proj_pts:
        uv = uv.ravel()  # flatten เป็น [u,v]
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(img_vis, (u,v), 3, (0,255,0), -1)

    cv2.imwrite(OUT_POSE_DIR / f"{img_file.stem}_reproj.jpg", img_vis)

print("Done. Check output_pose folder for rvec/tvec and reprojected images.")
