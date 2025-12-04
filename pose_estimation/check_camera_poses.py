import numpy as np

poses = np.load("camera_poses.npy", allow_pickle=True).item()

print("Number of cameras:", len(poses))
print("Example keys:", list(poses.keys())[:5])

first_key = list(poses.keys())[0]
print("Sample rvec shape:", poses[first_key]["rvec"].shape)
print("Sample tvec shape:", poses[first_key]["tvec"].shape)
