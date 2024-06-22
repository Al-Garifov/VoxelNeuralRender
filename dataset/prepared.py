import torch
import glob
import numpy as np

from dataset import DATA_ROOT


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str = DATA_ROOT):
        self._root = root
        self._volume_template = root + "converted_v001/volume_samples_*.npz"
        self._image_template = root + "converted_v001/image_samples_*.npz"
        self._volume_files = sorted(glob.glob(self._volume_template))
        self._image_files = sorted(glob.glob(self._image_template))

        if len(self._image_files) == 0:
            raise RuntimeError("Dataset.prepared (npz): image files can't be found.")

        if len(self._volume_files) == 0:
            raise RuntimeError("Dataset.prepared (npz): volume files can't be found.")

        if len(self._image_files) != len(self._volume_files):
            raise RuntimeError("Dataset.prepared (npz): image files count differs with volume files count.")

    def __len__(self):
        return len(self._volume_files)

    def __getitem__(self, idx):
        volumes_np = np.load(self._volume_files[idx])["arr_0"]  # FIXME: why do we use arr_0?
        volumes = torch.tensor(volumes_np, dtype=torch.float32)
        images_np = np.load(self._image_files[idx])["arr_0"]
        images = torch.tensor(images_np, dtype=torch.float32)
        return volumes, images
