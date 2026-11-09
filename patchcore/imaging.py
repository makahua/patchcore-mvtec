# -*- coding: utf-8 -*-
"""图像读入、几何预处理与扰动（从 patchcore_demo.py 拆出）。"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rel_path(p, data_root, cat):
    """把图片绝对路径转成 <类别>/<train|test>/... 的相对形式。

    落盘的 scores/*.json 要进公开仓库，写绝对路径会泄漏本机目录结构，
    所以这里统一按数据根目录做相对化，换台机器复算也不受影响。
    """
    try:
        rel = os.path.relpath(os.path.abspath(p), os.path.abspath(data_root))
        rel = rel.replace(os.sep, '/')
        if rel.startswith('..'):
            raise ValueError
        return rel
    except Exception:
        return '/'.join([cat] + os.path.abspath(p).replace('\\', '/').split('/')[-3:])


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_img(path, size, preprocess='stretch', aspect_crop=0.0, crop_zoom=1.0):
    """读图 → [1,3,S,S]。

    preprocess:
      stretch   —— 直接 resize 到 S×S（会改变长宽比；MVTec 全为方图，故与 letterbox 等价）
      letterbox —— 等比缩放到长边 = S，再用均值灰 (114) 居中补成 S×S（保住长宽比）
      crop      —— 将短边缩放到 S（长边≥S）后中心裁剪 S×S（上游 MVtec 几何：Resize(size)+CenterCrop(size)）
    aspect_crop:
      模拟非方形视野：先中心裁剪成 h/w = 该比例（如 0.75 ≈ 4:3 相机），再走上面的预处理。
      数据集本身全是方图，只有裁剪之后「拉伸畸变」才真正被引入，这就是本开关的用途。
    """
    im = Image.open(path).convert('RGB')
    if aspect_crop and aspect_crop > 0:
        w, h = im.size
        ch = min(h, max(1, int(round(w * aspect_crop))))
        top = (h - ch) // 2
        im = im.crop((0, top, w, top + ch))
    if preprocess == 'crop':
        w, h = im.size
        s = (size * max(1.0, crop_zoom)) / float(min(w, h))
        nw, nh = max(size, int(round(w * s))), max(size, int(round(h * s)))
        im = im.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - size) // 2, (nh - size) // 2
        im = im.crop((left, top, left + size, top + size))
    elif preprocess == 'letterbox':
        w, h = im.size
        s = size / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (size, size), (114, 114, 114))
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        im = canvas
    else:
        im = im.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x


def load_mask(path, size, preprocess='stretch', aspect_crop=0.0, crop_zoom=1.0):
    """与 load_img 完全相同的几何变换，但用 NEAREST、灰边填 0，返回 S×S 的 'L' 图。

    只有几何变换（裁剪 / letterbox）需要同步到 mask；亮度、噪声、模糊不动 mask。
    """
    im = Image.open(path).convert('L')
    if aspect_crop and aspect_crop > 0:
        w, h = im.size
        ch = min(h, max(1, int(round(w * aspect_crop))))
        top = (h - ch) // 2
        im = im.crop((0, top, w, top + ch))
    if preprocess == 'crop':
        w, h = im.size
        s = (size * max(1.0, crop_zoom)) / float(min(w, h))
        nw, nh = max(size, int(round(w * s))), max(size, int(round(h * s)))
        im = im.resize((nw, nh), Image.NEAREST)
        left, top = (nw - size) // 2, (nh - size) // 2
        im = im.crop((left, top, left + size, top + size))
    elif preprocess == 'letterbox':
        w, h = im.size
        s = size / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = im.resize((nw, nh), Image.NEAREST)
        canvas = Image.new('L', (size, size), 0)
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        im = canvas
    else:
        im = im.resize((size, size), Image.NEAREST)
    return im


def perturb_img(x, brightness=1.0, noise_std=0.0, blur_sigma=0.0, seed=0):
    """可复现的非几何采集扰动；x 为 [1,C,H,W]、范围 [0,1]。"""
    x = (x * brightness).clamp(0, 1)
    if blur_sigma > 0:
        radius = max(1, int(round(3 * blur_sigma)))
        ax = torch.arange(-radius, radius + 1, dtype=x.dtype)
        kernel = torch.exp(-(ax ** 2) / (2 * blur_sigma ** 2))
        kernel = kernel / kernel.sum()
        channels = x.shape[1]
        x = F.conv2d(x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                      padding=(0, radius), groups=channels)
        x = F.conv2d(x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                      padding=(radius, 0), groups=channels)
    if noise_std > 0:
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        x = (x + noise * noise_std).clamp(0, 1)
    return x


