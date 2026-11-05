# -*- coding: utf-8 -*-
"""记忆库元数据兼容性：旧 build 格式（bank+meta）与 --save-bank 新格式必须读成同一份口径。"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import classify as C  # noqa: E402


class _Z:
    """最小 npz 门面：只需要 .files 与 __getitem__。"""

    def __init__(self, data):
        self._d = data
        self.files = list(data)

    def __getitem__(self, key):
        return self._d[key]


def _load(path):
    """Windows 上 npz 会一直占着文件句柄，临时目录删不掉 —— 读完就关。"""
    with np.load(path, allow_pickle=True) as z:
        return _Z(dict((k, z[k]) for k in z.files))


def _write_legacy(path, dim=1536, neighbors=False, agg='max'):
    meta = dict(category='screw', size=320, neighbors=bool(neighbors), greedy=2000,
                score_agg=agg, calib_ratio=0.3, fpr=0.005, ng_ratio=1.2, seed=0,
                imagenet_norm=False, dim=dim * (9 if neighbors else 1),
                thr_ok=38.2, thr_ng=45.8, weights='torch_cache/wide_resnet50_2_timm.pth')
    np.savez_compressed(path, bank=np.zeros((3, meta['dim']), dtype=np.float32),
                        meta=np.array(json.dumps(meta), dtype=object))


def _write_save_bank(path, layers='2+3', neighbors=False, thr=38.6):
    base = C.LAYER_DIMS[layers]
    dim = base * (9 if neighbors else 1)
    np.savez_compressed(
        path, mem=np.zeros((2000, dim), dtype=np.float32),
        layers=np.array(layers), feat_dim=np.int32(base), size=np.int32(320),
        preprocess=np.array('stretch'), aspect_crop=np.float32(0.0),
        neighbors=np.array(bool(neighbors)), neighbor_pad=np.array('zero'),
        score_agg=np.array('max'), blur_sigma=np.float32(2.0), thr=np.float32(thr),
        thr_pct=np.float32(99.5), weights=np.array('torch_cache/wide_resnet50_2_timm.pth'),
        grid=np.int32(40), pool=np.int64(358400))


def test_legacy_meta_defaults_to_2plus3():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'screw.npz')
        _write_legacy(p)
        z = _load(p)
        meta = C._meta_from_npz(z, p)
    assert meta['layers'] == '2+3'
    assert meta['feat_dim'] == 1536
    assert meta['bank_format'] == 'classify-build'
    assert C.expected_dim(meta) == 1536


def test_save_bank_meta_reads_all_fields():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'screw.npz')
        _write_save_bank(p, layers='2+3', thr=38.6)
        meta = C._meta_from_npz(_load(p), p)
    assert meta['layers'] == '2+3'
    assert meta['feat_dim'] == 1536
    assert meta['grid'] == 40
    assert meta['size'] == 320
    assert meta['neighbors'] is False
    assert abs(meta['thr_ok'] - 38.6) < 1e-3
    assert abs(meta['thr_ng'] - 38.6 * 1.2) < 1e-3
    assert meta['category'] == 'screw'
    assert meta['bank_format'] == 'save-bank'
    assert C.expected_dim(meta) == 1536


def test_save_bank_1plus2_dim_and_neighbors():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'pill.npz')
        _write_save_bank(p, layers='1+2', neighbors=True)
        meta = C._meta_from_npz(_load(p), p)
    assert meta['layers'] == '1+2'
    assert C.expected_dim(meta) == 768 * 9


def test_describe_bank_mentions_layers_and_agg():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'screw.npz')
        _write_save_bank(p, layers='2+3')
        meta = C._meta_from_npz(_load(p), p)
    line = C.describe_bank(meta, 'screw.npz')
    assert '2+3' in line and '1536' in line and 'max' in line


def test_unknown_npz_raises():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'bad.npz')
        np.savez_compressed(p, foo=np.zeros(3, dtype=np.float32))
        try:
            C._meta_from_npz(_load(p), p)
        except SystemExit as e:
            assert '无法识别' in str(e)
        else:
            raise AssertionError('应当对无法识别的 npz 报错')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
