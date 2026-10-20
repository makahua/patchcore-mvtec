# -*- coding: utf-8 -*-
"""生成 MVTec 风格的合成小数据集，仅用于离线冒烟测试脚本流程（精度无意义）。"""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 256
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syn_data', 'syn')

def base_texture(seed):
    r = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:S, 0:S]
    base = 150 + r.randn(S, S) * 6 + 18 * np.sin(2 * np.pi * xx / 32.0)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert('L')
    d = ImageDraw.Draw(img)
    for _ in range(3):
        d.line([(r.randint(0, S), r.randint(0, S)), (r.randint(0, S), r.randint(0, S))],
               fill=int(150 + r.randint(-8, 8)), width=2)
    return img

def make_defect(img, kind, seed):
    r = np.random.RandomState(seed)
    img, mask = img.copy(), Image.new('L', (S, S), 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    if kind == 'blob':
        x, y, rr = r.randint(60, S - 60), r.randint(60, S - 60), r.randint(18, 32)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=int(60 + r.randint(0, 20)))
        dm.ellipse([x - rr, y - rr, x + rr, y + rr], fill=255)
    else:
        x0, y0 = r.randint(40, S - 80), r.randint(40, S - 80)
        x1, y1 = x0 + r.randint(40, 80), y0 + r.randint(20, 60)
        w = r.randint(3, 6)
        d.line([(x0, y0), (x1, y1)], fill=int(55 + r.randint(0, 25)), width=w)
        dm.line([(x0, y0), (x1, y1)], fill=255, width=w + 4)
    return img.filter(ImageFilter.GaussianBlur(0.6)), mask.filter(ImageFilter.GaussianBlur(1.2))

for split, n in (('train', 16), ('test', 10)):
    d = os.path.join(root, split, 'good'); os.makedirs(d, exist_ok=True)
    for i in range(n):
        base_texture(1000 + i).save(os.path.join(d, f'{i:03d}.png'))

for kind in ('blob', 'scratch'):
    d = os.path.join(root, 'test', kind); os.makedirs(d, exist_ok=True)
    gm = os.path.join(root, 'ground_truth', kind); os.makedirs(gm, exist_ok=True)
    for i in range(8):
        img, mask = make_defect(base_texture(1000 + i), kind, 5000 + i)
        img.save(os.path.join(d, f'{i:03d}.png'))
        mask.save(os.path.join(gm, f'{i:03d}_mask.png'))
print('synthetic dataset ->', root)
