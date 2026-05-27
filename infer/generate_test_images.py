import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import string
import random
from PIL import ImageDraw, ImageFilter
from captcha.image import ImageCaptcha
import numpy as np

characters = '-' + string.digits + string.ascii_uppercase
width, height = 192, 64
generator = ImageCaptcha(width=width, height=height)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
os.makedirs(output_dir, exist_ok=True)

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

print(f'Generating 10 test images to {output_dir}/')

for i in range(1, 11):
    random_str = ''.join([random.choice(characters[1:]) for _ in range(4)])
    img = generator.generate_image(random_str)
    img = add_noise(img)
    
    filename = f'{i:02d}_{random_str}.png'
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    print(f'Generated: {filename}')

print(f'\nDone! 10 test images saved to {output_dir}/')
