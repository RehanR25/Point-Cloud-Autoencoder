import os 
import glob
import numpy as np 
import trimesh 
import urllib.request
import zipfile

# create an image to point cloud pipeline

url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
zip_path = "ModelNet40.zip"
extract_dir = r"C:\Users\rehan\Documents\3D image\data\ModelNet40"

extract_parent = os.path.dirname(extract_dir)
os.makedirs(extract_parent, exist_ok=True)

if not os.path.exists(extract_dir):
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_parent)

# Sample points from each mesh
def sample_points_from_mesh(off_path, n_points=1024): 
    mesh = trimesh.load(off_path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = np.asarray(points, dtype=np.float32)
    points -= points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= (scale + 1e-8)
    return points 

output_dir = r"C:\Users\rehan\Documents\3D image\data\ModelNet40_pointclouds"
off_files = glob.glob(os.path.join(extract_dir, "**", "*.off"), recursive=True)
print(f"Found {len(off_files)} mesh files")

if not os.path.exists(output_dir): 
    for off_path in off_files: 
        try: 
            pts = sample_points_from_mesh(off_path, n_points=1024)
        except Exception as e: 
            print(f'Skipped {off_path}: {e}')
            continue

        rel_path = os.path.relpath(off_path, extract_dir)
        out_path = os.path.join(output_dir, rel_path).replace(".off", ".npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True) 
        np.save(out_path, pts) 

print("Preprocessing done")

# The preprocessing converts each mesh 
# into a point cloud (fixed list of 1024 x,y,z coordinates)