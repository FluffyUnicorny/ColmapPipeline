import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares
import time

# ----------------------------- CONFIG -----------------------------
# Adjust these paths if your layout differs
ROOT = Path("..")  # adjust if run from a different cwd; this assumes run_ba.py in bundle_adjustment/
DATA_IMG_DIR = ROOT / "data" / "dataset_kicker" / "images"   # <-- set your images folder
POINTS3D_FILE = ROOT / "triangulation" / "triangulated_points.npy"
DESCS3D_FILE = ROOT / "triangulation" / "triangulated_desc.npy"
CAMERA_POSES_FILE = ROOT / "pose_estimation" / "camera_poses.npy"

OUT_DIR = Path(".") / "ba_output"
OUT_DIR.mkdir(exist_ok=True)

OUT_POINTS = OUT_DIR / "triangulated_points_refined.npy"
OUT_CAMERA_POSES = OUT_DIR / "camera_poses_refined.npy"

# Camera intrinsics (change to real K if you have them)
FX = 1000.0
FY = 1000.0
# principal point will be set per-image using image size if images vary; otherwise you can set fixed cx,cy
USE_IMAGE_CENTER_PRINCIPAL_POINT = True

# BA optimizer settings
MAX_NFEV = 200  # increase if you have many params/observations
LOSS = 'huber'  # robust loss

# Match settings
MAX_MATCHES_PER_IMAGE = 1000  # limit per image for speed/robustness
MIN_MATCHES_FOR_IMAGE = 6

# ----------------------------- Helper functions -----------------------------
def pack_params(rvecs, tvecs, points3d):
    return np.hstack((rvecs.ravel(), tvecs.ravel(), points3d.ravel()))

def unpack_params(x, n_cameras, n_points):
    cam_r_vecs = x[: 3 * n_cameras].reshape((n_cameras, 3))
    cam_t_vecs = x[3 * n_cameras: 6 * n_cameras].reshape((n_cameras, 3))
    pts = x[6 * n_cameras:].reshape((n_points, 3))
    return cam_r_vecs, cam_t_vecs, pts

def project_point(rvec, tvec, point3d, K):
    R, _ = cv2.Rodrigues(rvec)
    Xc = R.dot(point3d) + tvec
    x = Xc[0] / Xc[2]
    y = Xc[1] / Xc[2]
    uv = K.dot(np.array([x, y, 1.0]))
    return uv[0], uv[1]

# vectorized residuals function for least_squares
def reprojection_residuals(x, n_cameras, n_points, camera_indices, point_indices, points_2d, Ks):
    rvecs, tvecs, pts3d = unpack_params(x, n_cameras, n_points)
    n_obs = camera_indices.shape[0]
    residuals = np.zeros((n_obs * 2,), dtype=float)

    for i in range(n_obs):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        u_obs, v_obs = points_2d[i]

        rvec = rvecs[cam_idx]
        tvec = tvecs[cam_idx]
        P3 = pts3d[pt_idx]

        K = Ks[cam_idx] if isinstance(Ks, (list, tuple)) else Ks
        try:
            u_proj, v_proj = project_point(rvec, tvec, P3, K)
        except Exception:
            u_proj, v_proj = 1e6, 1e6

        residuals[2 * i] = u_proj - u_obs
        residuals[2 * i + 1] = v_proj - v_obs

    return residuals

# ----------------------------- Main BA pipeline -----------------------------
def main():
    t_start = time.time()
    print("Loading data...")

    if not POINTS3D_FILE.exists():
        raise FileNotFoundError(f"Missing 3D points file: {POINTS3D_FILE}")
    if not DESCS3D_FILE.exists():
        raise FileNotFoundError(f"Missing 3D descriptors file: {DESCS3D_FILE}")
    if not CAMERA_POSES_FILE.exists():
        raise FileNotFoundError(f"Missing camera poses file: {CAMERA_POSES_FILE}")
    if not DATA_IMG_DIR.exists():
        raise FileNotFoundError(f"Missing image folder: {DATA_IMG_DIR}")

    points3d = np.load(POINTS3D_FILE)    # (N_pts, 3)
    desc3d = np.load(DESCS3D_FILE)       # (N_pts, 32) uint8
    camera_poses_dict = np.load(CAMERA_POSES_FILE, allow_pickle=True).item()
    image_files = sorted(list(DATA_IMG_DIR.glob("*.*")))

    print(f"Loaded {points3d.shape[0]} 3D points and {desc3d.shape[0]} descriptors")
    print(f"Found {len(camera_poses_dict)} camera poses and {len(image_files)} image files")

    # Build mapping from image basename -> index in camera list
    camera_names = sorted(list(camera_poses_dict.keys()))
    n_cameras = len(camera_names)
    camname_to_idx = {name: idx for idx, name in enumerate(camera_names)}

    # Initialize rvecs/tvecs arrays (n_cameras x 3)
    rvecs_init = np.zeros((n_cameras, 3), dtype=float)
    tvecs_init = np.zeros((n_cameras, 3), dtype=float)
    for i, name in enumerate(camera_names):
        r = camera_poses_dict[name]["rvec"].reshape(3,)
        t = camera_poses_dict[name]["tvec"].reshape(3,)
        rvecs_init[i] = r
        tvecs_init[i] = t

    n_points = points3d.shape[0]

    # Build observations by matching desc3d -> each image descriptors
    print("Building observations by matching 3D-desc -> image descriptors (this may take a while)...")
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    camera_indices_list = []
    point_indices_list = []
    points_2d_list = []
    Ks = []  # per-camera K (we'll build one per camera using image size)

    for name_idx, name in enumerate(camera_names):
        # find image file matching name if possible
        # use name + any extension
        candidates = [p for p in image_files if p.stem == name]
        if not candidates:
            # fallback: try first few images (or skip)
            print(f"Warning: no image file found for camera name {name}; skipping this camera for observations")
            Ks.append(np.array([[FX,0,0],[0,FY,0],[0,0,1]], dtype=float))
            continue
        img_path = candidates[0]
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps2d, des2d = orb.detectAndCompute(gray, None)
        if des2d is None or len(kps2d) == 0:
            print(f"Warning: no descriptors in image {img_path.name}, skip matches")
            Ks.append(np.array([[FX,0,img.shape[1]/2],[0,FY,img.shape[0]/2],[0,0,1]], dtype=float))
            continue

        # Create K for this camera (use image center)
        if USE_IMAGE_CENTER_PRINCIPAL_POINT:
            cx = img.shape[1] / 2.0
            cy = img.shape[0] / 2.0
            K_cam = np.array([[FX, 0, cx],
                              [0, FY, cy],
                              [0, 0, 1]], dtype=float)
        else:
            K_cam = np.array([[FX, 0, img.shape[1]/2.0],
                              [0, FY, img.shape[0]/2.0],
                              [0, 0, 1]], dtype=float)
        Ks.append(K_cam)

        # match desc3d -> des2d
        matches = bf.match(desc3d.astype(np.uint8), des2d)
        matches = sorted(matches, key=lambda m: m.distance)[:MAX_MATCHES_PER_IMAGE]
        if len(matches) < MIN_MATCHES_FOR_IMAGE:
            print(f"Note: only {len(matches)} matches for image {img_path.name} (camera {name}) - skipping if < {MIN_MATCHES_FOR_IMAGE}")
            # still add matches if some exist
        for m in matches:
            pt3_idx = m.queryIdx
            kp2 = kps2d[m.trainIdx].pt  # (u,v)
            camera_indices_list.append(name_idx)
            point_indices_list.append(pt3_idx)
            points_2d_list.append(kp2)

    camera_indices = np.array(camera_indices_list, dtype=int)
    point_indices = np.array(point_indices_list, dtype=int)
    points_2d = np.array(points_2d_list, dtype=float)

    print(f"Total observations: {camera_indices.shape[0]}")

    if camera_indices.shape[0] < 10:
        raise RuntimeError("Too few observations to run BA. Need more matches/observations.")

    # Compute initial reprojection error
    print("Computing initial reprojection error...")
    # Build single K per camera or a list Ks (we pass Ks to residuals)
    X0 = pack_params(rvecs_init, tvecs_init, points3d.astype(float))
    residuals0 = reprojection_residuals(X0, n_cameras, n_points, camera_indices, point_indices, points_2d, Ks)
    rms0 = np.sqrt(np.mean(residuals0**2))
    print(f"Initial RMS reprojection error (pixels): {rms0:.2f}")

    # ------------------ Run least_squares BA ------------------
    print("Running least squares BA ... this may take time")
    t0 = time.time()
    res = least_squares(fun=reprojection_residuals,
                        x0=X0,
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, Ks),
                        verbose=2,
                        xtol=1e-8,
                        ftol=1e-8,
                        gtol=1e-8,
                        max_nfev=MAX_NFEV,
                        loss=LOSS)
    t1 = time.time()
    print(f"BA finished in {(t1 - t0):.1f}s, success: {res.success}, message: {res.message}")
    # final residuals
    residuals_final = res.fun
    rms_final = np.sqrt(np.mean(residuals_final**2))
    print(f"Final RMS reprojection error (pixels): {rms_final:.2f}")

    # ------------------ Unpack optimized parameters ------------------
    rvecs_opt, tvecs_opt, pts3d_opt = unpack_params(res.x, n_cameras, n_points)

    # Save results: camera poses (dict) and points3d
    camera_poses_refined = {}
    for i, name in enumerate(camera_names):
        camera_poses_refined[name] = {
            "rvec": rvecs_opt[i].reshape(3,1),
            "tvec": tvecs_opt[i].reshape(3,1)
        }
    np.save(OUT_CAMERA_POSES, camera_poses_refined)
    np.save(OUT_POINTS, pts3d_opt)
    t_end = time.time()
    print(f"Saved refined camera poses to: {OUT_CAMERA_POSES.resolve()}")
    print(f"Saved refined 3D points to: {OUT_POINTS.resolve()}")
    print(f"Total time: {(t_end - t_start):.1f}s")
    print(f"Initial RMS: {rms0:.2f} px, Final RMS: {rms_final:.2f} px")

if __name__ == "__main__":
    main()
