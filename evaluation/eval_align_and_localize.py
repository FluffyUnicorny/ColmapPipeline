def evaluate_alignment(estimated_points_file, gt_ply_file, out_dir,
                       voxel_size=0.05, icp_threshold=0.2):
    """
    Evaluate alignment of estimated 3D points against GT point cloud using ICP.
    Supports:
      - .npy (Nx3)
      - COLMAP sparse folder (points3D.bin)
    """
    import numpy as np
    import open3d as o3d
    from pathlib import Path

    estimated_points_file = Path(estimated_points_file)
    gt_ply_file = Path(gt_ply_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load estimated 3D points
    # -----------------------------
    if estimated_points_file.is_dir():
        # 👉 COLMAP sparse directory
        from colmap_utils.read_write_model import read_model

        print(f"[INFO] Loading COLMAP sparse from {estimated_points_file}")
        _, _, points3D = read_model(str(estimated_points_file), ext=".bin")
        points_est = np.array([p.xyz for p in points3D.values()], dtype=np.float32)

    else:
        # 👉 .npy file
        print(f"[INFO] Loading estimated points from {estimated_points_file}")
        points_est = np.load(estimated_points_file)

    if points_est.shape[0] == 0:
        raise RuntimeError("No estimated 3D points loaded")

    pc_est = o3d.geometry.PointCloud()
    pc_est.points = o3d.utility.Vector3dVector(points_est)

    print(f"[INFO] Estimated points: {points_est.shape[0]}")

    # -----------------------------
    # Load GT point cloud
    # -----------------------------
    if not gt_ply_file.exists():
        raise FileNotFoundError(f"GT PLY not found: {gt_ply_file}")

    pc_gt = o3d.io.read_point_cloud(str(gt_ply_file))
    print(f"[INFO] GT points: {len(pc_gt.points)}")

    # -----------------------------
    # Downsample
    # -----------------------------
    pc_est_ds = pc_est.voxel_down_sample(voxel_size)
    pc_gt_ds = pc_gt.voxel_down_sample(voxel_size)

    # -----------------------------
    # ICP alignment
    # -----------------------------
    reg = o3d.pipelines.registration.registration_icp(
        pc_est_ds,
        pc_gt_ds,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T = reg.transformation
    pc_est.transform(T)

    # -----------------------------
    # Distance evaluation
    # -----------------------------
    distances = np.asarray(pc_est.compute_point_cloud_distance(pc_gt))
    rmse = float(np.sqrt(np.mean(distances ** 2)))
    mean_err = float(np.mean(distances))
    median_err = float(np.median(distances))

    # -----------------------------
    # Save results
    # -----------------------------
    np.save(out_dir / "distances.npy", distances)
    o3d.io.write_point_cloud(
        str(out_dir / "aligned_estimated_points.ply"), pc_est
    )

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Mean error (m): {mean_err:.4f}\n")
        f.write(f"Median error (m): {median_err:.4f}\n")
        f.write(f"RMSE (m): {rmse:.4f}\n")

    print(f"[OK] Alignment evaluation finished")
    print(f"     Mean   : {mean_err*100:.2f} cm")
    print(f"     Median : {median_err*100:.2f} cm")
    print(f"     RMSE   : {rmse*100:.2f} cm")

    return distances, mean_err, median_err, rmse
