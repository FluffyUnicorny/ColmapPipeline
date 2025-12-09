# shared/colmap_io.py
from pathlib import Path
import numpy as np

def load_colmap_sparse(sparse_dir: Path):
    from colmap_utils.read_write_model import read_model

    cameras, images, points3D = read_model(str(sparse_dir), ext=".bin")

    pts3d = np.array([p.xyz for p in points3D.values()])

    poses = {}
    for img in images.values():
        poses[img.name] = {
            "R": img.qvec2rotmat(),
            "t": img.tvec
        }

    return pts3d, poses
