import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# -----------------------------
# CONFIG
# -----------------------------
DATA_IMG_DIR = Path("../data/dataset_kicker/images")
POINTS3D_FILE = Path("../triangulation/triangulated_points.npy")
DESCS3D_FILE = Path("../triangulation/triangulated_desc.npy")
POSE_DIR = Path("output_refined_pose")  # โฟลเดอร์ rvec/tvec
OUT_VIS_DIR = Path("output_reproj_vis")
OUT_VIS_DIR.mkdir(exist_ok=True)

# -----------------------------
# โหลด 3D points + descriptors
# -----------------------------
points3d = np.load(POINTS3D_FILE)
desc3d = np.load(DESCS3D_FILE)

# -----------------------------
# ORB detector + matcher
# -----------------------------
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

errors = []

# -----------------------------
# Loop ภาพทั้งหมด
# -----------------------------
for img_file in DATA_IMG_DIR.glob("*.*"):
    img = cv2.imread(str(img_file))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect 2D keypoints + descriptors
    kps2d, des2d = orb.detectAndCompute(gray, None)
    if des2d is None or len(kps2d)==0:
        continue

    # match descriptors
    matches = bf.match(desc3d, des2d)
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    if len(matches)<4:
        continue

    pts3d_matched = np.array([points3d[m.queryIdx] for m in matches], dtype=np.float32)
    pts2d_matched = np.array([kps2d[m.trainIdx].pt for m in matches], dtype=np.float32)

    # โหลด refined pose
    rvec = np.load(POSE_DIR / f"{img_file.stem}_rvec.npy")
    tvec = np.load(POSE_DIR / f"{img_file.stem}_tvec.npy")

    K = np.array([[1000, 0, img.shape[1]/2],
                  [0, 1000, img.shape[0]/2],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    # project 3D -> 2D
    proj_pts, _ = cv2.projectPoints(pts3d_matched, rvec, tvec, K, dist_coeffs)
    proj_pts = proj_pts.reshape(-1,2)

    # compute reprojection error
    err = np.linalg.norm(pts2d_matched - proj_pts, axis=1)
    mean_err = err.mean()
    errors.append(mean_err)
    print(f"{img_file.name}: mean reprojection error = {mean_err:.2f} pixels")

    # -----------------------------
    # visualize keypoints vs projected points
    # -----------------------------
    img_vis = img.copy()
    # keypoints จริง: แดง
    for pt in pts2d_matched:
        cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)
    # projected points: เขียว
    for pt in proj_pts:
        cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)

    cv2.imwrite(OUT_VIS_DIR / f"{img_file.stem}_reproj_vis.jpg", img_vis)

# -----------------------------
# สรุปค่า reprojection error
# -----------------------------
if errors:
    print(f"\nAverage reprojection error over {len(errors)} images: {np.mean(errors):.2f} pixels")
else:
    print("No images processed.")
