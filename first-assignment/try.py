import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
from enum import Enum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")