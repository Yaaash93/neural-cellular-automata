from model import CAModel
from train import to_rgb, make_seed

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = CAModel(n_channels=16, device=device)
model.load_state_dict(torch.load("nca_model.pth", map_location=device))
model.eval()

# create seed
x = make_seed(size=40, n_channels=16).to(device)

# evolve
frames = []

with torch.no_grad():
    for i in range(300):
        x = model(x)

        img = to_rgb(x[:, :4]).detach().cpu()
        frames.append(img)

import matplotlib.pyplot as plt

plt.imshow(frames[-1][0].permute(1,2,0))
plt.show()