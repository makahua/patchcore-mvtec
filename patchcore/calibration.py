# -*- coding: utf-8 -*-
"""标定/评测辅助：进程峰值内存、掩码、像素 dump（从 patchcore_demo.py 拆出）。"""
import os

import numpy as np
from PIL import Image

from .imaging import load_img, load_mask, perturb_img, rel_path
from .scoring import patch_scores

def peak_ws_mb():
    """本进程峰值工作集（MB）。Windows 用 K32GetProcessMemoryInfo；其它平台返回 None。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]

        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.PeakWorkingSetSize / 1024 ** 2, 1)
    except Exception:
        return None



SUITE_CONDITIONS = (('clean', 1.0, 0.0, 0.0), ('bright08', 0.8, 0.0, 0.0),
                    ('noise005', 1.0, 0.05, 0.0), ('noise010', 1.0, 0.10, 0.0),
                    ('blur1', 1.0, 0.0, 1.0))


def mask_for(path, label, test_dir, gt_dir, hw, args):
    """取 GT 掩码并降采样到 patch 网格（几何变换与图像保持一致）。"""
    mask = None
    if label == 1:
        rel = os.path.relpath(path, test_dir)
        m = os.path.join(gt_dir, os.path.splitext(rel)[0] + '_mask.png')
        if not os.path.exists(m):
            m = os.path.join(gt_dir, os.path.dirname(rel),
                             os.path.splitext(os.path.basename(rel))[0] + '.png')
        if os.path.exists(m):
            if args.aspect_crop > 0 or args.preprocess != 'stretch':
                mimg = load_mask(m, args.size, args.preprocess, args.aspect_crop, args.crop_zoom)
                mask = np.asarray(mimg.resize((hw, hw), Image.NEAREST)) > 128
            else:
                mask = np.asarray(Image.open(m).convert('L').resize((hw, hw), Image.NEAREST)) > 128
    if mask is None:
        mask = np.zeros((hw, hw), bool)
    return mask


def _patch_map(model, mem, x, args, device, dtype):
    patch, _ = patch_scores(model, mem, x, device, args.neighbors, args.score_agg,
                            args.blur_sigma, args.neighbor_pad, fp16=args.fp16)
    return patch.detach().cpu().numpy().astype(dtype)


def dump_suite(args, model, mem, device, cat, score_src, test_imgs, test_dir, gt_dir, dtype):
    """同一记忆库、五组条件各存一份 calib/test 分数图，供离线比较聚合。"""
    os.makedirs(args.dump_suite, exist_ok=True)
    dname = 'float32' if dtype is np.float32 else 'float16'
    labels = np.array([0 if os.path.basename(os.path.dirname(p)) == 'good' else 1 for p in test_imgs], np.int8)
    rels = np.array([rel_path(p, args.data, cat) for p in test_imgs])
    masks = None
    for ci, (name, bright, noise, blur) in enumerate(SUITE_CONDITIONS):
        base_seed = args.seed * 1000 + ci * 777
        cals, tests = [], []
        for i, p in enumerate(score_src):
            x = load_img(p, args.size, args.preprocess, args.aspect_crop, args.crop_zoom)
            if bright != 1.0 or noise or blur:
                x = perturb_img(x, bright, noise, blur, base_seed + i)
            cals.append(_patch_map(model, mem, x, args, device, dtype))
        for i, p in enumerate(test_imgs):
            x = load_img(p, args.size, args.preprocess, args.aspect_crop, args.crop_zoom)
            if bright != 1.0 or noise or blur:
                x = perturb_img(x, bright, noise, blur, base_seed + 100000 + i)
            tests.append(_patch_map(model, mem, x, args, device, dtype))
        if masks is None:
            hw = cals[0].shape[-1]
            masks = np.stack([mask_for(p, int(l), test_dir, gt_dir, hw, args)
                              for p, l in zip(test_imgs, labels)])
        common = dict(thr_pct=np.float32(args.thr_pct), size=np.int32(args.size),
                      dtype=np.array(dname), condition=np.array(name))
        np.savez_compressed(os.path.join(args.dump_suite, f'{cat}_{name}_calib.npz'),
                            scores=np.stack(cals), labels=np.zeros(len(cals), np.int8),
                            paths=np.array([rel_path(p, args.data, cat) for p in score_src]),
                            **common)
        np.savez_compressed(os.path.join(args.dump_suite, f'{cat}_{name}_test.npz'),
                            scores=np.stack(tests), masks=masks, labels=labels, paths=rels,
                            **common)
        print(f'[suite] {cat} {name}: calib {len(cals)} / test {len(tests)} 张 → {args.dump_suite}')


