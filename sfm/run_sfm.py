import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "data" / "dataset_kicker" / "images"
OUTPUT_PATH = Path(__file__).parent / "colmap_output"
DB_PATH = OUTPUT_PATH / "database.db"
SPARSE_PATH = OUTPUT_PATH / "sparse"

OUTPUT_PATH.mkdir(exist_ok=True)
SPARSE_PATH.mkdir(exist_ok=True)

COLMAP = r"C:\COLMAP\nocuda\COLMAP.bat"

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, shell=True, check=True)

print("=== STEP 1: Feature extraction ===")
run([
    COLMAP, "feature_extractor",
    "--database_path", str(DB_PATH),
    "--image_path", str(IMAGE_PATH),
    "--ImageReader.single_camera", "1"
])

print("=== STEP 2: Matching ===")
run([
    COLMAP, "exhaustive_matcher",
    "--database_path", str(DB_PATH)
])

print("=== STEP 3: Mapping (SfM) ===")
run([
    COLMAP, "mapper",
    "--database_path", str(DB_PATH),
    "--image_path", str(IMAGE_PATH),
    "--output_path", str(SPARSE_PATH)
])

print("\n✅ SfM pipeline finished")
