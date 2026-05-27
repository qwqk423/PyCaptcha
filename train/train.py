import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import string
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from dataset import CaptchaDataset
from model import CRNN
from utils import train, valid, decode, decode_target

torch.backends.cudnn.benchmark = True

characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)
n_input_length = 12

batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_set = CaptchaDataset(characters, 1000 * batch_size, width, height, n_input_length, n_len)
valid_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

model = CRNN(n_classes, input_shape=(3, height, width))
model = model.to(device)

if hasattr(torch, 'compile'):
    model = torch.compile(model)
    print('Model compiled with torch.compile()')

print(f'characters: {characters}')
print(f'classes: {n_classes}, image size: {width}x{height}')

scaler = GradScaler('cuda')

optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
epochs = 6
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, characters, device, scaler)
    valid(model, epoch, valid_loader, characters, device)

optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
epochs = 3
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, characters, device, scaler)
    valid(model, epoch, valid_loader, characters, device)

dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)
image, target, input_length, label_length = dataset[0]
print('true:', decode_target(target, characters))

model.eval()
output = model(image.unsqueeze(0).to(device))
output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
print('pred:', decode(output_argmax[0], characters))

save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'captcha_crnn.pth')
torch.save(model.state_dict(), save_path)
print(f'model saved to {save_path}')
