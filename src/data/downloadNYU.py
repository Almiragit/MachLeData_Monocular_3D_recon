import os
import requests
from tqdm import tqdm

<<<<<<< HEAD
=======

>>>>>>> 83dab2a3a58fb0b30202b6ff52cba402c35217c8
def download_nyu_data(url, save_path):
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return

    print(f"Downloading NYU Depth V2 from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
<<<<<<< HEAD
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
=======

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

>>>>>>> 83dab2a3a58fb0b30202b6ff52cba402c35217c8
    with open(save_path, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

<<<<<<< HEAD
if __name__ == "__main__":
    NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    SAVE_DIR = "data/raw/nyu_depth_v2_labeled.mat"
    download_nyu_data(NYU_URL, SAVE_DIR)
=======

if __name__ == "__main__":
    NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    SAVE_DIR = "data/nyu/raw/nyu_depth_v2_labeled.mat"
    download_nyu_data(NYU_URL, SAVE_DIR)
>>>>>>> 83dab2a3a58fb0b30202b6ff52cba402c35217c8
