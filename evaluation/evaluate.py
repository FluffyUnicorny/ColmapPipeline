import numpy as np
import cv2
from pathlib import Path


def evaluate_reprojection(
    points_file: Path,
    camera_file: Path,
    img_dir: Path,
    gt_2d_dir: Path,
    K=None,
    dist_coeffs=None,
):
    """
    Evaluate RMS reprojection error using COLMAP camera poses
    """

    # ---------- load data ----------
    points3d = np.load(points_file)                     # (N, 3)
    cameras = np.load(camera_file, allow_pickle=True).item()

    print(f"[INFO] Loaded {points3d.shape[0]} 3D points")
    print(f"[INFO] Loaded {len(cameras)} camera poses")

    if K is None:
        K = np.array(
            [[1000, 0, 1920 / 2],
             [0, 1000, 1080 / 2],
             [0,    0,       1]],
            dtype=np.float32,
        )

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    # ---------- evaluation ----------
    errors = []
    used_images = 0

    for img_path in sorted(img_dir.glob("*")):
        img_name = img_path.name   # IMPORTANT: COLMAP pose key = filename

        if img_name not in cameras:
            print(f"[SKIP] {img_name}: camera pose not found")
            continue

        gt_file = gt_2d_dir / f"{img_path.stem}.npy"
        if not gt_file.exists():
            print(f"[SKIP] {img_name}: no GT 2D points")
            continue

        cam = cameras[img_name]

        R = cam["R"]
        tvec = cam["t"].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)

        proj_pts, _ = cv2.projectPoints(
            points3d, rvec, tvec, K, dist_coeffs
        )
        proj_pts = proj_pts.reshape(-1, 2)

        gt_pts = np.load(gt_file)

        if len(gt_pts) != len(proj_pts):
            print(
                f"[SKIP] {img_name}: GT {len(gt_pts)} != proj {len(proj_pts)}"
            )
            continue

        img_errors = np.linalg.norm(proj_pts - gt_pts, axis=1)
        errors.extend(img_errors)
        used_images += 1

    rms_error = np.sqrt(np.mean(np.square(errors))) if errors else float("nan")

    print(
        f"[INFO] Reprojection evaluation finished | "
        f"images={used_images}, RMS={rms_error:.2f} px"
    )

    return rms_error, used_images


# ---------- standalone test ----------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    evaluate_reprojection(
        points_file = ROOT / "colmap/output/points3d.npy",
        camera_file = ROOT / "colmap/output/camera_poses.npy",
        img_dir     = ROOT / "data/dataset_kicker/images",
        gt_2d_dir   = ROOT / "data/dataset_kicker/gt_2d",
    )
