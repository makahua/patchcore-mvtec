# -*- coding: utf-8 -*-
"""断点续传下载器：走系统代理环境变量（urllib 自动读取 HTTPS_PROXY），失败自动重试续传。"""
import os, sys, time, urllib.request, urllib.error

URL = sys.argv[1]
OUT = sys.argv[2]
RETRIES = int(sys.argv[3]) if len(sys.argv) > 3 else 80

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)

def total_size():
    req = urllib.request.Request(URL, method='HEAD')
    with urllib.request.urlopen(req, timeout=40) as r:
        return int(r.headers.get('Content-Length') or 0)

TOTAL = total_size()
print(f'[dl] total = {TOTAL/1e9:.2f} GB -> {OUT}', flush=True)

attempt = 0
while attempt < RETRIES:
    attempt += 1
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if TOTAL and have >= TOTAL:
        break
    try:
        req = urllib.request.Request(URL)
        if have:
            req.add_header('Range', f'bytes={have}-')
        with urllib.request.urlopen(req, timeout=60) as r:
            mode = 'ab' if have and r.status == 206 else 'wb'
            if mode == 'wb':
                have = 0
            t0 = time.time()
            base = have
            with open(OUT, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    dt = time.time() - t0
                    if dt > 4:
                        spd = (have - base) / dt / 1e6
                        pct = have * 100.0 / TOTAL if TOTAL else 0
                        print(f'[dl] {have/1e9:.2f}/{TOTAL/1e9:.2f} GB  {pct:5.1f}%  {spd:5.1f} MB/s  try{attempt}', flush=True)
                        t0, base = time.time(), have
    except Exception as e:
        print(f'[dl] 中断({type(e).__name__}: {e})，3 秒后续传，第 {attempt} 次', flush=True)
        time.sleep(3)

have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
print(f'[dl] DONE {have}/{TOTAL} bytes', flush=True)
