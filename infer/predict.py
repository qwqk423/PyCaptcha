import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import string
import random
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms.functional import to_tensor
import numpy as np

from model import CRNN
from utils import decode

characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = CRNN(n_classes, input_shape=(3, height, width))
model = model.to(device)

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'captcha_crnn.pth')
state_dict = torch.load(model_path, map_location=device, weights_only=True)
new_state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.eval()
print('Model loaded successfully!')


def add_noise(img):
    draw = ImageDraw.Draw(img)

    arr = np.array(img)
    bg_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    bg_color = [random.randint(200, 255) for _ in range(3)]
    arr[bg_mask] = bg_color
    img = ImageDraw.Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    noise_count = random.randint(100, 400)
    for _ in range(noise_count):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.point((x, y), fill=color)

    line_count = random.randint(3, 6)
    for _ in range(line_count):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

    curve_count = random.randint(1, 3)
    for _ in range(curve_count):
        points = []
        x = random.randint(0, width // 4)
        y = random.randint(0, height)
        points.append((x, y))
        for _ in range(3):
            x = random.randint(x, min(x + width // 3, width))
            y = random.randint(0, height)
            points.append((x, y))
        color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        draw.line(points, fill=color, width=1)

    blur_radius = random.uniform(0.3, 0.8)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def predict_single(image_path, use_noise=False):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    if use_noise:
        image = add_noise(image)
    image_tensor = to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output_argmax = output.permute(1, 0, 2).argmax(dim=-1)
        result = decode(output_argmax[0], characters)

    return result


def predict_batch(image_dir, use_noise=False):
    results = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(image_dir, filename)
            result = predict_single(filepath, use_noise)
            results.append((filename, result))
            print(f'{filename}: {result}')
    return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path or directory> [--noise]")
        print("Example: python predict.py ../../test_images/01_A3K9.png")
        print("         python predict.py ../../test_images/ --noise")
        sys.exit(1)

    path = sys.argv[1]
    use_noise = '--noise' in sys.argv

    if os.path.isdir(path):
        print(f'\nBatch predicting images in: {path}\n')
        predict_batch(path, use_noise)
    else:
        result = predict_single(path, use_noise)
        print(f'Predicted: {result}')
