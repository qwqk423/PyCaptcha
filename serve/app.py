import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import string
import random
import base64

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from torchvision.transforms.functional import to_tensor
from captcha.image import ImageCaptcha

from model import CRNN
from utils import decode

app = FastAPI(title="验证码识别服务")

characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(n_classes, input_shape=(3, height, width))
model = model.to(device)

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'captcha_crnn.pth')
state_dict = torch.load(model_path, map_location=device, weights_only=True)
new_state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()
print(f'Model loaded on {device}')

generator = ImageCaptcha(width=width, height=height)


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


def predict_image(img: Image.Image) -> str:
    img = img.convert('RGB').resize((width, height))
    image_tensor = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output_argmax = output.permute(1, 0, 2).argmax(dim=-1)
        result = decode(output_argmax[0], characters)

    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    result = predict_image(img)
    return JSONResponse({"result": result})


@app.post("/predict-base64")
async def predict_base64(data: dict):
    image_data = data.get("image", "")
    if image_data.startswith("data:image"):
        image_data = image_data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    result = predict_image(img)
    return JSONResponse({"result": result})


@app.get("/generate")
async def generate():
    random_str = ''.join([random.choice(characters[1:]) for _ in range(4)])
    img = generator.generate_image(random_str)
    img = add_noise(img)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse({
        "image": f"data:image/png;base64,{img_base64}",
        "answer": random_str
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
