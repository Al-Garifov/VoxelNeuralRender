from numpy import ndarray
import numpy as np
import imageio as iio
import pyopenvdb as vdb

from dataset import DATA_ROOT


def _get_images(count: int, start: int = 0) -> ndarray:
    """

    :return: ndarray
    """

    def rgb2gray(rgb: ndarray) -> ndarray:
        """

        :param rgb: ndarray
        :return: ndarray
        """
        return rgb[..., :1] / 256.0

    images = ndarray([count, 256, 256, 1])

    for i in range(count):
        idx = i + start + 1
        im = rgb2gray(iio.v2.imread(DATA_ROOT + "/render_v001/render.{0:05d}.png".format(idx)))  # 256 256 1
        images[i] = im

    return images


def _get_volumes(count: int, start: int = 0) -> ndarray:
    """

    :return: ndarray
    """
    volumes = ndarray([count, 256, 256, 256], np.float32)
    for i in range(count):
        idx = i + start + 1
        print(f"reading vdb: {idx}")
        grid = vdb.read(DATA_ROOT + "/vdb_v001/vdb.{0:05d}.vdb".format(idx), "density")
        grid.copyToArray(volumes[i], (0, 0, 0))
    return volumes.reshape([-1, 1, 256, 256, 256])


def save_volume(volume: ndarray, filename: str):
    vdb_volume = vdb.FloatGrid()
    vdb_volume.copyFromArray(volume)
    vdb_volume.name = "density"
    vdb_volume.transform = vdb.createLinearTransform(voxelSize=1 / 64)
    vdb_volume.transform.translate([-0.5, -0.5, -0.5])
    vdb.write(filename, vdb_volume)


def get_volume(path: str):
    volume = ndarray([1, 64, 64, 64], np.float32)
    grid = vdb.read(path, "density")
    grid.copyToArray(volume[0], (0, 0, 0))
    return volume


def convert_to_npz(samples: int, path: str, batch: int = 100):
    for i in range(0, samples // batch):
        volumes = _get_volumes(batch, start=i * batch)
        images = _get_images(batch, start=i * batch)
        np.savez_compressed(path + f"/volume_samples_{i:>03}.npz", volumes)
        np.savez_compressed(path + f"/image_samples_{i:>03}.npz", images)


if __name__ == "__main__":
    convert_to_npz(5000, DATA_ROOT + "/converted_v001/")
