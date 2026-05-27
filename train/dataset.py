import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from captcha.image import ImageCaptcha
from PIL import ImageDraw, ImageFilter


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        img = self.generator.generate_image(random_str)
        img = self._add_noise(img)
        image = to_tensor(img)
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length

    def _add_noise(self, img):
        draw = ImageDraw.Draw(img)

        arr = np.array(img)
        bg_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
        bg_color = [random.randint(200, 255) for _ in range(3)]
        arr[bg_mask] = bg_color
        img = ImageDraw.Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        noise_count = random.randint(100, 400)
        for _ in range(noise_count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            draw.point((x, y), fill=color)

        line_count = random.randint(3, 6)
        for _ in range(line_count):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

        curve_count = random.randint(1, 3)
        for _ in range(curve_count):
            points = []
            x = random.randint(0, self.width // 4)
            y = random.randint(0, self.height)
            points.append((x, y))
            for _ in range(3):
                x = random.randint(x, min(x + self.width // 3, self.width))
                y = random.randint(0, self.height)
                points.append((x, y))
            color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
            draw.line(points, fill=color, width=1)

        blur_radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return img
