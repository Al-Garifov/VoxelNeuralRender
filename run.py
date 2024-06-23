import torch

from dataset import prepared
from models.cnn import ConvDeconv
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


model = ConvDeconv().to(dev)
dataset = prepared.Dataset()

image = model(dataset[-1][0][56].to(dev))

plt.imsave("prediction.jpg", image.cpu().detach()[0][0], cmap='gray', vmin=0, vmax=1)
