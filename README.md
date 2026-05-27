# 验证码识别 (CAPTCHA Recognition)

基于 CRNN (CNN + BiLSTM + CTC) 的验证码识别项目，支持数字和大写字母的 4 位验证码识别。

## 项目结构

```
word/
├── model.py                 # CRNN 模型定义
├── utils.py                 # 训练/验证工具函数 + CTC 解码
├── requirements.txt         # 依赖列表
├── captcha_crnn.pth         # 预训练模型权重
├── train/
│   ├── train.py             # 训练脚本
│   └── dataset.py           # 数据集（动态生成验证码 + 噪声增强）
├── infer/
│   ├── predict.py           # 单张/批量预测脚本
│   └── generate_test_images.py  # 生成测试图片
└── serve/
    ├── app.py               # FastAPI Web 服务
    └── index.html           # Web 前端页面
```

## 模型架构

- **CNN**: 5 个卷积块（共 10 层卷积），逐步提取图像特征
- **BiLSTM**: 2 层双向 LSTM，序列建模
- **FC**: 全连接层输出字符概率分布
- **CTC Loss**: 实现不定长文本的对齐与识别

## 训练

### 数据

- **字符集**: 数字 `0-9` + 大写字母 `A-Z`，共 36 种字符 + 1 个 CTC blank（`-`），总计 37 类
- **验证码长度**: 4 位
- **图片尺寸**: 192×64（RGB）
- **数据生成**: 使用 `captcha` 库动态生成，无需预先准备数据集
- **数据量**: 训练集每 epoch 128000 张（1000 batch × 128），验证集每 epoch 12800 张

### 数据增强

训练时对验证码图片施加多种噪声以提高模型鲁棒性：

| 增强方式 | 说明 |
|---------|------|
| 背景色抖动 | 将白色背景替换为随机浅色（200-255） |
| 噪点 | 随机 100-400 个彩色噪点 |
| 干扰线 | 3-6 条随机线段 |
| 干扰曲线 | 1-3 条贝塞尔曲线 |
| 高斯模糊 | 半径 0.3-0.8 的随机模糊 |

### 训练策略

1. **两阶段学习率**: 先以 `lr=1e-3` 训练 6 个 epoch，再以 `lr=1e-4` 微调 3 个 epoch
2. **优化器**: Adam（amsgrad=True）
3. **混合精度**: 使用 `GradScaler` + `autocast` 加速训练、减少显存占用
4. **模型编译**: 自动启用 `torch.compile()` 提升性能
5. **CTC 输入长度**: CNN 将 192px 宽度下采样 16 倍 → 序列长度 12
6. **DataLoader**: `num_workers=4`、`pin_memory=True`、`prefetch_factor=2` 加速数据加载

### 执行训练

```bash
cd train
python train.py
```

训练完成后模型权重自动保存到项目根目录 `captcha_crnn.pth`。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 训练

```bash
cd train
python train.py
```

### 预测

```bash
# 单张图片预测
cd infer
python predict.py ../infer/img/01_A3K9.png

# 批量预测（添加噪声测试鲁棒性）
python predict.py ../infer/img/ --noise

# 生成测试图片
python generate_test_images.py
```

### Web 服务

```bash
cd serve
python app.py
```

访问 `http://127.0.0.1:8000` 打开 Web 界面，支持：
- 粘贴图片识别（Ctrl+V）
- 拖拽/点击上传图片
- 随机生成验证码并自动识别
- REST API：`POST /predict`、`POST /predict-base64`、`GET /generate`

## 预训练模型

`captcha_crnn.pth` 为已训练好的模型权重，字符集为 `-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`（`-` 为 CTC blank），验证码长度 4，图片尺寸 192×64。

## 依赖

- PyTorch >= 2.7.0
- torchvision
- captcha
- fastapi + uvicorn
- Pillow
