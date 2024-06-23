import os.path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import time

from dataset import prepared
from models.cnn import ConvDeconv

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


model = ConvDeconv().to(dev)
dataset = prepared.Dataset()

loss_fn = nn.L1Loss()
opt = optim.Adam(model.parameters(), lr=3e-4)

epoch_init = 0
epochs = 10
batch_size = 2

min_loss = 1e9

if os.path.exists("checkpoint.pt"):
    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_init = checkpoint['epoch'] + 1

for epoch in range(epoch_init, epoch_init + epochs):
    start = time.time()
    running_loss = 0.0
    validation_loss = 0.0
    for chunk in range(len(dataset) - 1):
        volumes, images = dataset[chunk]
        for sample in range(len(volumes), step=batch_size):
            inputs = volumes[sample:sample+batch_size].to(dev)
            targets = images[sample:sample+batch_size].to(dev)

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape([-1]), targets.reshape([-1]))
            loss.backward()
            opt.step()

            running_loss += loss
    with torch.no_grad():
        volumes, images = dataset[-1]
        for sample in range(len(volumes), step=batch_size):
            inputs = volumes[sample:sample+batch_size].to(dev)
            targets = images[sample:sample+batch_size].to(dev)

            outputs = model(inputs)
            loss = loss_fn(outputs.reshape([-1]), targets.reshape([-1]))
            validation_loss += loss

    print(f"Epoch {epoch + 1} done in {time.time() - start:.2f} seconds.")

    print(f"Train      loss is {running_loss/(len(dataset) - 1)/100.0*batch_size:.2f}")
    print(f"Validation loss is {validation_loss/100.0*batch_size:.2f}")

    image = model(dataset[-1][0][56].to(dev))

    plt.imsave(f"./progress/epoch{epoch}.jpg", image.cpu().detach()[0][0], cmap='gray', vmin=0, vmax=1)

    if min_loss > validation_loss:
        min_loss = validation_loss
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': validation_loss,
                }, "checkpoint.pt")
