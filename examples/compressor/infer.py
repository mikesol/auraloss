import os
import glob
import torchaudio
import sys
import torch
import lightning.pytorch as pl
from argparse import ArgumentParser

from tcn import TCNModel
# from data2 import SignalTrainSingerDataset
from data import SignalTrainLA2ADataset
# checkpoint = "./lightning_logs/version_23/checkpoints/epoch=19-step=29180.ckpt"
checkpoint = "./lightning_logs/version_5/checkpoints/epoch=19-step=29480.ckpt"
model = TCNModel.load_from_checkpoint(checkpoint)
device = "cuda"
dataset = torch.utils.data.DataLoader(SignalTrainLA2ADataset(root_dir="SignalTrain_LA2A_Dataset_1.1",                                                              half=False,

                                                             length=32768), shuffle=True,
                                      batch_size=1)
model.to(device)

input_file = torch.empty(1, 0)
pred_file = torch.empty(1, 0)
target_file = torch.empty(1, 0)
for x, (input, target, params) in enumerate(dataset):
    input = input.to(device)
    params = params.to(device)
    with torch.no_grad():
        out = model(input, params)
        input = input.to("cpu")
        target = target.to("cpu")
        out = out.to("cpu")
        print(input.shape, target.shape, out.shape)
        input_file = torch.cat(
            (input_file, torch.squeeze(input, dim=0)), 1)
        pred_file = torch.cat(
            (pred_file, torch.squeeze(out, dim=0)), 1)
        target_file = torch.cat(
            (target_file, torch.squeeze(target, dim=0)), 1)
        torchaudio.save(f"input5.wav", input_file, 44100)
        torchaudio.save(f"pred5.wav", pred_file, 44100)
        torchaudio.save(f"target5.wav", target_file, 44100)
        sys.exit(0)
