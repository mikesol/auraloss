import os
import glob
import torch
import lightning.pytorch as pl
from argparse import ArgumentParser

from tcn import TCNModel
from data import SignalTrainLA2ADataModule

from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience

def cli_main():
    cli = LightningCLI(TCNModel, SignalTrainLA2ADataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
