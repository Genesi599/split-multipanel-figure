#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify.py — 面板归属验证 + 面板边界裁剪

核心规则:
  * 归属 = 色块 bbox 左上角所在面板（不是质心）
  * L 形规则: 每面板色块集合左上角必须 ≥ 字母锚点（色块不能超字母横线上面/竖线左边）
  * 面板 = 归它的所有色块集合（并集）+ 边距，不是固定矩形框

用法:
  python verify.py <image> <labels.json> [--init init.json]
输出: 每个面板的裁剪框 JSON + L 形验证结果 + 面板色块数
"""
import sys, os, io, json
import numpy as np
from PIL import Image
from scipy import ndimage


def load_content(src):
    arr = np.array(Image.open(src).convert('RGB'))
    content = ~((arr[:, :, 0] >= 250) & (arr[:, :, 1] >= 250) & (arr[:, :, 2] >= 250))
    return arr, content


def build_panels(labels, W, H, row_gap_min=15):
    """根据标签坐标 + 行/列空白带，推断面板粗边界"""
    labs = sorted(labels.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    rows = []
    for name, (x, y) in labs:
        if rows and abs(y - rows[-1][0][1][1]) < 40:
            rows[-1].append((name, (x, y)))
        else:
            rows.append([(name, (x, y))])
    # 按标签 y 直接给水平分区（顶=标签y，底=下一行标签上沿+边距）
    panels = {}
    for ri, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda t: t[1][0])
        top = row_sorted[0][1][1]            # 面板顶 = 标签 y（色块不能超字母横线）
        bot = H
        if ri + 1 < len(rows):
            bot = min(p[1][1] for p in rows[ri + 1]) + 5
        for j, (name, (x, y)) in enumerate(row_sorted):
            xl = 0 if j == 0 else (row_sorted[j - 1][1][0] + x) // 2
            xr = W if j == len(row_sorted) - 1 else (x + row_sorted[j + 1][1][0]) // 2
            panels[name] = (xl, top, xr, bot)
    return panels


def assign_components(content, labels, panels, min_area=15):
    """连通域 + bbox 左上角归属"""
    labeled, n = ndimage.label(content, structure=np.ones((3, 3)))
    sizes = ndimage.sum(content, labeled, range(1, n + 1))
    owner = {}
    for i in range(1, n + 1):
        if sizes[i - 1] < min_area:
            continue
        ys, xs = np.where(labeled == i)
        bx0, by0 = int(xs.min()), int(ys.min())
        for name, (x0, y0, x1, y1) in panels.items():
            if x0 <= bx0 < x1 and y0 <= by0 < y1:
                owner[i] = name
                break
        else:
            owner[i] = None
    return labeled, sizes, owner


def panel_crop(labeled, sizes, owner, labels, name, W, H, margin=3):
    """面板 = 归它的所有色块集合并集 bbox + 边距（含字母）"""
    lx, ly = labels[name]
    allxs, allys = [lx], [ly]
    for i, o in owner.items():
        if o == name:
            ys, xs = np.where(labeled == i)
            allxs.extend(xs.tolist())
            allys.extend(ys.tolist())
    b0 = (max(0, min(allxs) - margin), max(0, min(allys) - margin),
          min(W, max(allxs) + margin), min(H, max(allys) + margin))
    return b0


def verify_lshape(labeled, sizes, owner, labels, name):
    """L 形: 面板色块集合左上角 ≥ 字母锚点"""
    lx, ly = labels[name]
    minx, miny = lx, ly
    for i, o in owner.items():
        if o == name:
            ys, xs = np.where(labeled == i)
            minx = min(minx, int(xs.min()))
            miny = min(miny, int(ys.min()))
    return (minx >= lx, miny >= ly), (minx, miny)


def main():
    src, lp = sys.argv[1], sys.argv[2]
    arr, content = load_content(src)
    H, W, _ = arr.shape
    with io.open(lp, encoding='utf-8') as f:
        labels = {k: tuple(v) for k, v in json.load(f).items()}
    panels = build_panels(labels, W, H)
    labeled, sizes, owner = assign_components(content, labels, panels)
    all_ok = True
    crops = {}
    for name in sorted(labels):
        lshape, (mx, my) = verify_lshape(labeled, sizes, owner, labels, name)
        b = panel_crop(labeled, sizes, owner, labels, name, W, H)
        crops[name] = b
        n_ids = sum(1 for o in owner.values() if o == name)
        ok = all(lshape)
        if not ok: all_ok = False
        print(f'{name}: 裁剪{b} 色块{n_ids} 集合左上({mx},{my}) L形{"OK" if ok else "FAIL"}')
    print(f'L形验证: {"ALL OK" if all_ok else "NEED FIX"}')
    with io.open('crop_boxes.json', 'w', encoding='utf-8') as f:
        json.dump({k: list(v) for k, v in crops.items()}, f, ensure_ascii=False, indent=1)
    print('裁剪框 → crop_boxes.json')


if __name__ == '__main__':
    main()
