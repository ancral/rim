import os
from PIL import Image
import h5py
import numpy as np

from utils.generate_mask import rearrange_image_and_create_mask

def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

def loadFiles(path_dir, load_images=True, dim=32):
    collected = []
    files = [os.path.join(r, f)
             for r, _, fs in os.walk(path_dir)
             for f in fs if (f.endswith(".h5") and not load_images) or 
                           (f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) and load_images)]

    length = len(os.listdir(path_dir))
    print_progress_bar(0, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i, filepath in enumerate(sorted(files)):
        print_progress_bar(i + 1, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        try:
            if filepath.endswith(".h5") and not load_images:
                with h5py.File(filepath, "r") as f:
                    data = np.array(f["dataset_name"])
                    collected.append(data)
            elif load_images:
                img = Image.open(filepath).convert("RGB")
                if img.size[0] == img.size[1]:
                    img = img.resize((dim, dim), Image.Resampling.LANCZOS)
                else:
                    img = np.array(img)
                    img, _ = rearrange_image_and_create_mask(img, dim)
                    collected.append(img)
                    continue
                collected.append(np.array(img))
        except Exception as e:
            print(f"[WARNING] Could not process {filepath}: {e}")

    collected_np = np.array(collected)
    print(f"\nTotal loaded: {collected_np.shape}")
    return collected_np