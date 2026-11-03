# -*- coding: utf-8 -*-
"""检查仓库内 markdown 的相对链接是否都指向真实文件。

CI 里跑一遍，防止 README 链接到不存在的 docs（这类断链只有点进去才发现）。
只检查相对路径；http/https、mailto、纯锚点 #xxx 跳过。

用法：
    python tools/check_links.py          # 检查当前仓库
    python tools/check_links.py --root . # 指定根目录
"""
import argparse
import os
import re
import sys

SKIP_DIRS = {'.git', '.venv', 'MVTec', 'VisA_raw', 'VisA_pytorch', 'torch_cache', 'node_modules', '__pycache__'}
MD_LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    bad = []
    n_md = 0
    n_link = 0
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith('.md'):
                continue
            n_md += 1
            p = os.path.join(dirpath, name)
            with open(p, encoding='utf-8') as f:
                txt = f.read()
            for m in MD_LINK.finditer(txt):
                target = m.group(1).strip()
                if target.startswith(('http://', 'https://', 'mailto:', '#', '<')):
                    continue
                target = target.split()[0].rstrip('"')
                n_link += 1
                if not os.path.exists(os.path.join(dirpath, target)):
                    bad.append('%s -> %s' % (os.path.relpath(p, root), target))
    print('%d 篇 md，%d 条相对链接' % (n_md, n_link))
    if bad:
        print('\n'.join(bad))
        sys.exit('%d 条断链' % len(bad))
    print('全部相对链接有效')


if __name__ == '__main__':
    main()
