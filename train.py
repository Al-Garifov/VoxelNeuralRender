import numpy.dtypes
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
opt = optim.Adam(model.parameters(), lr=2e-3)

epochs = 10

min_loss = 1e9

for epoch in range(epochs):
    start = time.time()
    running_loss = 0.0
    validation_loss = 0.0
    for batch in range(len(dataset) - 1):
        volumes, images = dataset[batch]
        for sample in range(len(volumes)):
            inputs = volumes[sample].to(dev)
            targets = images[sample].to(dev)

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape([-1]), targets.reshape([-1]))
            loss.backward()
            opt.step()

            running_loss += loss
    with torch.no_grad():
        volumes, images = dataset[-1]
        for sample in range(len(volumes)):
            inputs = volumes[sample].to(dev)
            targets = images[sample].to(dev)

            outputs = model(inputs)
            loss = loss_fn(outputs.reshape([-1]), targets.reshape([-1]))
            validation_loss += loss

    print(f"Epoch {epoch + 1} done in {time.time() - start:.2f} seconds.")

    print(f"Train      loss is {running_loss/(len(dataset) - 1)/100.0:.2f}")
    print(f"Validation loss is {validation_loss/100.0:.2f}")

    if min_loss > validation_loss:
        min_loss = validation_loss
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': validation_loss,
                }, "checkpoint.pt")
