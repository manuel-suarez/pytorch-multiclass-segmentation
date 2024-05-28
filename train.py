import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model import UNET

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print("Running on the GPU")
else:
    DEVICE = "cpu"
    print("Running on the CPU")

MODEL_PATH = "YOUR-MODEL-PATH"
LOAD_MODEL = False
ROOT_DIR = "../datasets/cityscapes"
IMG_HEIGHT = 110
IMG_WIDTH = 220
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 5


def train_function(data, model, optimizer, loss_fn, device):
    print("Entering into train function")
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    global epoch
    epoch = 0
    LOSS_VALS = []

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
        ]
    )

    train_set = get_cityscapes_data(
        split="train",
        mode="fine",
        relabelled=True,
        root_dir=ROOT_DIR,
        transforms=transform,
        batch_size=BATCH_SIZE,
    )

    print("Data Loaded Successfully!")
