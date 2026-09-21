# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 08/01/2024
# ---------------------------------------------------------------------------------------------

import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile

from utils.architecture import Unet
from utils.diffusion import ElucidatedDiffusion

torch.manual_seed(23)
import pickle

DTYPE = torch.float32

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def error_metric(pred,true, Par):
    #re-normalize
    # true = true*Par['out_scale'] + Par['out_shift']
    # true = true*Par['out_scale'] + Par['out_shift']
    return torch.norm(true-pred, p=2)/torch.norm(true, p=2)

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample
    
def preprocess(x,y, Par):
    x = sliding_window_view(x[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])
    y = sliding_window_view(y[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])

    print('x: ', x.shape)
    print('y: ', y.shape)
    print()
    return x,y

unet_dir = (
    r"E:\AI_Project\NeuralOperator_DiffusionModel"
    r"\case_1_kolmogorov\no_dm\unet"
)
begin_time = time.time()
x_train = np.load(os.path.join(unet_dir, "TRAIN_PRED.npy"))
y_train = np.load(os.path.join(unet_dir, "TRAIN_TRUE.npy"))

x_val = np.load(os.path.join(unet_dir, "VAL_PRED.npy"))
y_val = np.load(os.path.join(unet_dir, "VAL_TRUE.npy"))

x_test = np.load(os.path.join(unet_dir, "TEST_PRED.npy"))
y_test = np.load(os.path.join(unet_dir, "TEST_TRUE.npy"))
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")

# 快速验证：只取50帧
val_num = min(50, len(x_val))
val_indices = np.linspace(0, len(x_val) - 1, val_num, dtype=int)
x_val = x_val[val_indices]
y_val = y_val[val_indices]

# 快速测试：也只取50帧
test_num = min(50, len(x_test))
test_indices = np.linspace(0, len(x_test) - 1, test_num, dtype=int)
x_test = x_test[test_indices]
y_test = y_test[test_indices]

print("Validation:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)

# Train-Val-Test Split
# idx1 = int(0.8 * inp.shape[0])
# idx2 = int(0.9 * inp.shape[0])

# x_train = inp[:idx1]
# x_val   = inp[idx1:idx2]
# x_test  = inp[idx2:]

# y_train = out[:idx1]
# y_val   = out[idx1:idx2]
# y_test  = out[idx2:]



inp_min = np.min(x_train)
inp_max = np.max(x_train)
out_min = np.min(y_train)
out_max = np.max(y_train)



Par = {"inp_shift" : torch.tensor(inp_min, dtype=DTYPE, device=device),
       "inp_scale" : torch.tensor(inp_max - inp_min, dtype=DTYPE, device=device),
       "out_shift" : torch.tensor(out_min, dtype=DTYPE, device=device),
       "out_scale" : torch.tensor(out_max - out_min, dtype=DTYPE, device=device),
       "nx"        : x_train.shape[2],
       "ny"        : x_train.shape[3],
       "nf"        : 1,
       "lb"        : 1,
       "lf"        : 1,
       "num_epochs": 10
       }

# Normalizing the data to [0,1]
shift = Par['inp_shift'].detach().cpu().numpy()
scale = Par['inp_scale'].detach().cpu().numpy()
x_train = (x_train - shift)/scale
x_val = (x_val - shift)/scale
x_test = (x_test - shift)/scale

shift = Par['out_shift'].detach().cpu().numpy()
scale = Par['out_scale'].detach().cpu().numpy()
y_train = (y_train - shift)/scale
y_val = (y_val - shift)/scale
y_test = (y_test - shift)/scale

Par["sigma_data"] = np.std(y_train)

# Traj splitting
begin_time = time.time()
print('\nTrain Dataset')
x_train, y_train = preprocess(x_train, y_train, Par)
print('\nValidation Dataset')
x_val, y_val = preprocess(x_val, y_val, Par)
print('\nTest Dataset')
x_test, y_test = preprocess(x_test, y_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

Par.update({"channels"       : x_train.shape[1],
            "self_condition" : True
            })

print("Par")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models_reduced")
os.makedirs(model_dir, exist_ok=True)

best_model_path = os.path.join(model_dir, "best_model.pt")
with open(os.path.join(script_dir, "Par_reduced.pkl"), "wb") as f:
    pickle.dump(Par, f)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = MyDataset(x_train_tensor, y_train_tensor)
val_dataset = MyDataset(x_val_tensor, y_val_tensor)
test_dataset = MyDataset(x_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 4 #16
val_batch_size   = 1
test_batch_size  = 1
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Define Network Architecture
net = Unet(
    dim = 16,
    dim_mults = (1, 2, 4, 8),
    channels = Par["channels"],
    self_condition = Par["self_condition"],
    flash_attn = False
).to(device).to(torch.float32)
summary(net, input_size=((1,)+x_train.shape[1:], (1,)) )

model = ElucidatedDiffusion(
    net,
    channels=Par["channels"],
    image_size_h=Par["nx"],
    image_size_w=Par["ny"],
    sigma_data=Par["sigma_data"]
).to(device)

# Adjust the dimensions as per your model's input size
dummy_x = torch.tensor(torch.randn(1, Par["channels"], Par["nx"], Par["ny"]),   dtype=DTYPE, device=device)
dummy_input = (dummy_x, dummy_x)

# Profile the model
flops = torchprofile.profile_macs(model, dummy_input)
print(f"FLOPs: {flops:.3e}")

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed

# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0


t0 = time.time()
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    train_time = time.time()
    for l_fidel, h_fidel  in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        with autocast():
            loss = model(h_fidel.to(device), l_fidel.to(device))
#随机生成噪声强度 sigma；
#给真实流场 h_fidel 加入高斯噪声；
#将带噪流场、sigma和NO预测 l_fidel输入DM内部U-Net；
#让网络估计干净流场；
#计算去噪训练损失。
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)
    train_time = time.time()-train_time

    # Validation
    if (epoch + 1) % 5 == 0:
        val_time = time.time()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for l_fidel, h_fidel in val_loader:
                with autocast():
                    pred = model.sample(l_fidel.to(device))
                    loss   = error_metric(pred, h_fidel.to(device), Par)
                val_loss += loss.item()

        val_loss /= len(val_loader)

            # Save the model if validation loss is the lowest so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_id = epoch+1
            torch.save(model.state_dict(), best_model_path)
        
        if epoch % 100 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"model_{epoch + 1}.pt")
            )

        val_time = time.time() - val_time
        
        time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
        elapsed_time = time.time() - begin_time
        print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}, val time: {val_time:.2f}')

    else:
        time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
        print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}')


print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")
model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)
# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for l_fidel, h_fidel in test_loader:
        with autocast():
            pred = model.sample(l_fidel.to(device))
            loss   = error_metric(pred, h_fidel.to(device), Par)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

