"""极简 safetensors -> torch state_dict 转换（纯 numpy，不依赖 safetensors 包）。"""
import json, struct, numpy as np, torch, sys

src = sys.argv[1]
dst = sys.argv[2]

DT = {'F32': np.float32, 'F16': np.float16, 'F64': np.float64,
      'I64': np.int64, 'I32': np.int32, 'I16': np.int16, 'I8': np.int8,
      'U8': np.uint8, 'BOOL': np.bool_}

with open(src, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
    blob = f.read()

sd = {}
for name, info in header.items():
    if name == '__metadata__':
        continue
    off0, off1 = info['data_offsets']
    dtype = info['dtype']
    shape = info['shape']
    raw = blob[off0:off1]
    if dtype == 'BF16':
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        t = torch.from_numpy(arr.view(np.float32).copy()).view(tuple(shape))
    else:
        arr = np.frombuffer(raw, dtype=DT[dtype]).reshape(tuple(shape))
        t = torch.from_numpy(arr.copy())
    sd[name] = t

print('tensors:', len(sd))
print('sample keys:', list(sd)[:5])

from torchvision.models import wide_resnet50_2
m = wide_resnet50_2(weights=None)
clean = {}
for k, v in sd.items():
    kk = k
    for pre in ('model.', 'module.', '_orig_mod.'):
        if kk.startswith(pre):
            kk = kk[len(pre):]
    clean[kk] = v
miss, unexp = m.load_state_dict(clean, strict=False)
hit = len(m.state_dict()) - len(miss)
print('hit', hit, '/', len(m.state_dict()), '| unexpected', len(unexp), '| sample missing', miss[:4])
if hit >= len(m.state_dict()) * 0.9:
    torch.save({k: v for k, v in clean.items() if not k.startswith('fc.')}, dst)
    print('SAVED', dst)
else:
    print('MISMATCH')
