# -*- coding: utf-8 -*-
"""
通用面板分割器（无 AI，配合 label_detect.py 使用）

流程:
  1. 读图 + labels.json(标签坐标 {字母:[x,y]})
  2. 内容连通域 → 自动网格定面板粗边界(行/列空白带)
  3. bbox 左上角归属 → 面板 = 色块集合并集
  4. 验证: L形 + 贴字母 + 不重叠
  5. 裁剪子图 + 色块归属图 + 分割线图

用法:
  python split_panels.py <img> <labels.json> --out outdir [--panels '手工面板框json']
  --panels 可选: 若自动网格不准, 传手工面板框 {"A":[x0,y0,x1,y1],...}
"""
import sys, os, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def load_arr(src):
    return np.array(Image.open(src).convert('RGB'))


def cbands(proj, minv, minl, tot):
    """内容连续带 [(start,end)]"""
    b = []; inb = False
    for i, v in enumerate(proj):
        if v > minv:
            if not inb:
                s = i; inb = True
        else:
            if inb and i - s >= minl:
                b.append((s, i - 1))
            inb = False
    if inb and tot - s >= minl:
        b.append((s, tot - 1))
    return b


def auto_panels(labels, content, H, W):
    """从标签 + 内容空白带自动定面板粗边界"""
    # 标签按 y 分组成行
    ls = sorted(labels.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    rows = []
    for name, (x, y) in ls:
        if rows and abs(y - rows[-1][0][1][1]) < 45:
            rows[-1].append((name, (x, y)))
        else:
            rows.append([(name, (x, y))])
    # 行边界: 行顶=标签y, 行底=下行标签y前的内容空白带中心
    row_d = content.sum(axis=1)
    panels = {}
    for ri, row in enumerate(rows):
        y_top = min(p[1] for _, p in row)
        y_bot = H
        if ri + 1 < len(rows):
            y_nxt = min(p[1] for _, p in rows[ri + 1])
            # 两行标签之间找最空水平带
            seg = content[y_top:y_nxt]
            rc = seg.sum(axis=1)
            best = y_top; bmin = 1e9
            for y in range(0, len(rc) - 5):
                v = rc[y:y + 5].mean()
                if v < bmin:
                    bmin = v; best = y_top + y + 2
            y_bot = best
        row_sorted = sorted(row, key=lambda t: t[1][0])
        for j, (name, (x, y)) in enumerate(row_sorted):
            x_l = 0 if j == 0 else _col_gap(row_sorted[j - 1][1][0], x, y_top, y_bot, content)
            x_r = W if j == len(row_sorted) - 1 else _col_gap(x, row_sorted[j + 1][1][0], y_top, y_bot, content)
            panels[name] = (x_l, y_top, x_r, y_bot)
    return panels


def _col_gap(xa, xb, yt, yb, content):
    """两标签间列空白带中心"""
    seg = content[yt:yb]
    cc = seg.sum(axis=0)
    lo = max(0, xa + 5); hi = min(len(cc) - 1, xb - 5)
    best = (lo + hi) // 2; bmin = 1e9
    for x in range(lo, max(lo, hi - 8)):
        v = cc[x:x + 8].mean()
        if v < bmin:
            bmin = v; best = x + 4
    return best


def split(src, labels, panels):
    arr = load_arr(src)
    H, W, _ = arr.shape
    content = ~((arr[:, :, 0] >= 235) & (arr[:, :, 1] >= 235) & (arr[:, :, 2] >= 235))
    for n, (x, y) in labels.items():
        content[max(0, y - 5):y + 16, max(0, x - 4):x + 13] = False
    lab, nc = ndimage.label(content, structure=np.ones((3, 3)))
    sizes = ndimage.sum(content, lab, range(1, nc + 1))

    def assign(i):
        ys, xs = np.where(lab == i); bx0, by0 = int(xs.min()), int(ys.min())
        for n, (x0, y0, x1, y1) in panels.items():
            if x0 <= bx0 < x1 and y0 <= by0 < y1:
                return n
        return None
    owner = {i: assign(i) for i in range(1, nc + 1) if sizes[i - 1] >= 12}
    # 面板 = 色块集合并集
    crop = {}
    for n, (x0, y0, x1, y1) in panels.items():
        lx, ly = labels[n]; allx = [lx]; ally = [ly]
        for i, o in owner.items():
            if o == n:
                ys_, xs_ = np.where(lab == i); allx.extend(xs_.tolist()); ally.extend(ys_.tolist())
        crop[n] = (int(max(0, min(allx) - 3)), int(max(0, min(ally) - 3)),
                   int(min(W, max(allx) + 3)), int(min(H, max(ally) + 3)))
    # 验证
    result = {'crop': crop, 'owner_count': {}, 'L形': 'ALL OK', 'overlap': []}
    for n, p in panels.items():
        ids = [i for i, o in owner.items() if o == n]
        result['owner_count'][n] = len(ids)
        lx, ly = labels[n]
        if not ids:
            result['L形'] = 'FAIL(无色块)'
            continue
        minx = min([int(np.where(lab == i)[1].min()) for i in ids])
        miny = min([int(np.where(lab == i)[0].min()) for i in ids])
        if not (minx >= lx and miny >= ly):
            result['L形'] = 'FAIL'
    for a in crop:
        for b in crop:
            if a >= b:
                continue
            A, B = crop[a], crop[b]
            x = min(A[2], B[2]) - max(A[0], B[0]); y = min(A[3], B[3]) - max(A[1], B[1])
            if x > 0 and y > 0:
                result['overlap'].append(f'{a}∩{b}: x{x} y{y}')
    return arr, content, lab, sizes, owner, crop, result


def main():
    src = sys.argv[1]; labels_src = sys.argv[2]
    outdir = 'split_out'
    panels_json = None
    if '--out' in sys.argv:
        outdir = sys.argv[sys.argv.index('--out') + 1]
    if '--panels' in sys.argv:
        panels_json = json.load(open(sys.argv[sys.argv.index('--panels') + 1]))
    os.makedirs(outdir, exist_ok=True)
    labels = json.load(open(labels_src, encoding='utf-8'))
    if panels_json is None:
        arr0 = load_arr(src)
        content0 = ~((arr0[:, :, 0] >= 235) & (arr0[:, :, 1] >= 235) & (arr0[:, :, 2] >= 235))
        panels_json = auto_panels(labels, content0, content0.shape[0], content0.shape[1])
        print('自动面板边界:')
        for n, p in panels_json.items(): print(f'  {n}: {p}')
    arr, content, lab, sizes, owner, crop, result = split(src, labels, panels_json)
    # 子图
    for n, box in crop.items():
        Image.open(src).convert('RGB').crop(box).save(f'{outdir}/panel_{n}.png')
    # 色块归属图
    names = sorted(labels.keys()); pc = {}
    for i, n in enumerate(names):
        h = i / len(names); r, g, b = __import__('colorsys').hsv_to_rgb(h, 0.75, 0.95)
        pc[n] = (int(r * 255), int(g * 255), int(b * 255))
    outarr = np.full(arr.shape, 255, dtype=np.uint8)
    for i, o in owner.items():
        if o is None: continue
        ys_, xs_ = np.where(lab == i); outarr[ys_, xs_] = np.array(pc[o], dtype=np.uint8)
    img = Image.fromarray(outarr); d = ImageDraw.Draw(img)
    for n, (x, y) in labels.items(): d.text((x + 2, y + 2), n, fill=(0, 0, 0))
    img.save(f'{outdir}/色块归属.png')
    # 分割线图
    img2 = Image.open(src).convert('RGB'); d2 = ImageDraw.Draw(img2)
    for n, (x0, y0, x1, y1) in crop.items():
        d2.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        d2.text((x0 + 2, y0 + 2), n, fill=(0, 0, 255))
    img2.save(f'{outdir}/分割线.png')
    json.dump(crop, open(f'{outdir}/crop_boxes.json', 'w'), indent=1)

    print(f'裁剪框:')
    for n, b in crop.items(): print(f'  {n}: {b} {b[2]-b[0]}x{b[3]-b[1]}')
    print('L形:', result['L形'])
    print('重叠:', result['overlap'] if result['overlap'] else '无 ✓')
    print(f'→ 子图/色块归属.png/分割线.png/crop_boxes.json 存于 {outdir}')


if __name__ == '__main__':
    main()
