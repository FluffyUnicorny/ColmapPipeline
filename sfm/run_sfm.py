# sfm/run_sfm.py
import subprocess
from pathlib import Path

def run_sfm(img_dir: Path, colmap_ws: Path):
    """
    Run COLMAP automatic reconstruction
    :param img_dir: Path to folder containing images
    :param colmap_ws: Path to output workspace
    """
    if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
        print(f"⚠ No images found in {img_dir}")
        return False

    colmap_ws.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "automatic_reconstructor",
        "--image_path", str(img_dir),
        "--workspace_path", str(colmap_ws),
        "--dense", "0"
    ]

    print("▶ Running COLMAP:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ COLMAP finished successfully")
    return True

# สำหรับรัน standalone (ทดสอบ)
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    IMG_DIR = ROOT / "data/dataset_kicker/images"
    COLMAP_WS = ROOT / "data/dataset_kicker/colmap"
    run_sfm(IMG_DIR, COLMAP_WS)
