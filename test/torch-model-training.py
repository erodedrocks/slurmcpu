import argparse
import math
import os
import sys
import time

import torch
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import TensorDataset, Dataset
from torchmetrics import Accuracy
import lightning as pl
import numpy as np


class SDSCSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# %%


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train a simple Convolutional Neural Network to classify images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-c', '--classes', type=int, default=10, choices=[10, 100], help='number of classes in dataset')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32', 'fp64'], help='floating-point precision')
    parser.add_argument('-e', '--epochs', type=int, default=42, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-H', '--height', type=int, default=128, help='img height')
    parser.add_argument('-W', '--width', type=int, default=192, help='img width')
    parser.add_argument('-a', '--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'hpu', 'tpu'], help='accelerator')
    parser.add_argument('-w', '--num_workers', type=int, default=-1, help='number of workers | if num_workers is -1, it will be set as cpus * 2')
    parser.add_argument('-m', '--model_file', type=str, default="", help="pre-existing model file if needing to further train model")
    parser.add_argument('-P', '--savepytorch', type=bool, default=False, help="save model as keras model file")
    parser.add_argument('-O', '--saveonnx', type=bool, default=False, help="save model as ONNX model file")

    args = parser.parse_args()
    return args


def createdataset(root: str, dtype):
    dataset: ImageFolder = datasets.ImageFolder(root)
    dstuple = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(6059))
    trainds: SDSCSubset = SDSCSubset(dstuple[0], transforms.Compose([transforms.CenterCrop([128, 192]), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype)]))
    testds: SDSCSubset = SDSCSubset(dstuple[1], transforms.Compose([transforms.CenterCrop([128, 192]), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype)]))
    valds: SDSCSubset = SDSCSubset(dstuple[2], transforms.Compose([transforms.CenterCrop([128, 192]), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype)]))
    return trainds, testds, valds


# %%
def create_datasets(dtype):
    return createdataset("/expanse/lustre/projects/ddp324/akallu/sdsc10", dtype)

# %%


class CNN(pl.LightningModule):
    def __init__(self, classes, args):
        super(CNN, self).__init__()
        self.args = args

        self.train_acc = Accuracy(num_classes=classes, task='MULTICLASS')
        self.test_acc = Accuracy(num_classes=classes, task='MULTICLASS')
        self.val_acc = Accuracy(num_classes=classes, task='MULTICLASS')

        newheight = math.floor(0.5 * (math.floor(0.5 * (args.height - 2)) - 2)) - 2
        newwidth = math.floor(0.5 * (math.floor(0.5 * (args.width - 2)) - 2)) - 2

        self.cnn_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (newwidth) * (newheight), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, classes)
        )

    def forward(self, x):
        return self.cnn_block(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.log("train_acc_epoch", self.train_acc.compute(), prog_bar=True, on_epoch=True)

        self.train_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

# %%


def main():
    """ Train CNN on CIFAR """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()
    if args.num_workers == -1:
        args.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])

    # Set internal variables from input variables and command-line arguments
    classes = args.classes
    match args.precision:
        case 'bf16': tf_float = torch.bfloat16
        case 'fp16': tf_float = torch.float16
        case 'fp64': tf_float = torch.float64
        case 'fp32': tf_float = torch.float32
        case _: raise Exception(
            "Provided precision string: " +
            args.precision +
            " is not within the accepted set of values: ['bf16', 'fp16', 'fp64', 'fp32']"
        )

    epochs = args.epochs
    batch_size = args.batch_size

    # increase speed by optimizing fp32 matmul | TODO: MAKE THIS AN ARG
    if torch.cuda.device_count() > 0:
        torch.set_float32_matmul_precision('high')

    # Create training and test datasets
    train_dataset, test_dataset, val_dataset = create_datasets(tf_float)

    # Prepare the datasets for training and evaluation
    cifar_datamodule = pl.LightningDataModule.from_datasets(train_dataset=train_dataset, num_workers=args.num_workers, batch_size=batch_size, val_dataset=val_dataset, test_dataset=test_dataset)

    # Create model
    if args.model_file != "":
        model = torch.load(args.model_file)
    else:
        model = CNN(classes, args)

    # # Train the model on the dataset || TODO: make the accel option and devices / nodes an arg
    trainer = pl.Trainer(max_epochs=epochs, accelerator=args.accelerator, limit_train_batches=0.1)
    trainer.fit(model, datamodule=cifar_datamodule)
    trainer.test(model, dataloaders=cifar_datamodule, verbose=True)

    fake_input = torch.rand((batch_size, 3, 128, 192), dtype=tf_float)  # Fake input to emulate how actual input would be given

    modelDir = "model_exports/version_torch"  # Create str ref of model directory
    version = str(trainer.logger.version)

    os.makedirs(modelDir, exist_ok=True)

    # export ONNX and PyTorch models w/ builtin versions
    if args.saveonnx:
        torch.onnx.export(model.eval(), fake_input, f"{modelDir}/{version}_model.onnx", input_names=["input"], output_names=["output"])
    if args.savepytorch:
        torch.save(model.eval(), f"{modelDir}/{version}_model.pt")

    return 0


if __name__ == '__main__':
    timestart = time.time()
    i = main()
    output = open(f"benchmarks.log", "a")
    output.writelines(f"{os.environ['SLURM_CPUS_PER_TASK']},{time.time() - timestart}\n")
    sys.exit(i)