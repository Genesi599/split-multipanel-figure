#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""find_labels.py — 自动定位面板标签字母（像素方法，不依赖视觉模型坐标）

两种策略自动切换：
  1. 高度直方图法（简单图）：全图文字高度统计，若存在明显高于正文的"标签高度峰"，
     用 [峰±2px] + 单字母宽阈过滤，一次命中全部标签。
  2. 先找 A → 按 A 高度模板匹配（复杂图）：高度直方图无单一标签峰时，
     左上角(x<120,y<100)最高孤立字母 = A，以 A 高度做全图模板匹配找其余同名高标签。

用法:
  python find_labels.py <image> [--method auto|histogram|anchor] [--out labels.json]
输出: 标签坐标 JSON {"A":[x,y], ...}（按布局行序 A,B,C..）

原理: 面板标签是"孤立大写字母"——周围空白的连通域，且同图内各面板标签字号统一。
"""
import sys, os, io, json, time
import numpy as np
from PIL import Image
from scipy import ndimage


def load(arr=None, src=None):
    if src is not None:
        arr = np.array(Image.open(src).convert('RGB'))
    dark = (arr[:, :, 0] < 200) & (arr[:, :, 1] < 200) & (arr[:, :, 2] < 200)
    return arr, dark


def label_components(dark, min_area=15, max_area=4000):
    """8 邻域连通域标记，返回 (labeled, sizes, 候选元组列表)"""
    struct = np.ones((3, 3), dtype=bool)
    labeled, n = ndimage.label(dark, structure=struct)
    sizes = ndimage.sum(dark, labeled, range(1, n + 1))
    return labeled, sizes


def glyph_str(dark, i, labeled, pad=1):
    """打印连通域字形（用于人工验证字母身份，A/B/C...）"""
    ys, xs = np.where(labeled == i)
    sub = dark[max(0, ys.min()-pad):ys.max()+pad+1,
               max(0, xs.min()-pad):xs.max()+pad+1]
    return '\n'.join(''.join('#' if v else '.' for v in r) for r in sub)


def match_by_height(dark, labeled, sizes, h_min, h_max, w_min=6, w_max=14,
                    fill_lo=0.3, fill_hi=0.9, area_min=20, area_max=400,
                    iso_thr=0.30, hard_iso=False):
    """按高度+宽度+填充率+孤立度找模板匹配的标签字母

    孤立度 = 候选 bbox 外扩 pad 区域内的非背景像素比例。
    面板标签是"孤立大字母"（周围空白），而正文/图内文字与其他字粘连 → 孤立度低。
    hard_iso=True 时要求外扩 1px 无任何相邻暗像素（完全孤立，最接近标签特征）。
    """
    out = []
    for i in range(1, sizes.size + 1):
        s = int(sizes[i - 1])
        if s < area_min or s > area_max:
            continue
        ys, xs = np.where(labeled == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if not (h_min <= h <= h_max and w_min <= w <= w_max):
            continue
        fill = s / (w * h)
        if not (fill_lo <= fill <= fill_hi):
            continue
        # 1px 完全孤立检查：外扩一圈无任何不属于本块的暗像素
        x0, x1 = max(0, xs.min() - 1), xs.max() + 2
        y0, y1 = max(0, ys.min() - 1), ys.max() + 2
        ring = dark[y0:y1, x0:x1].copy()
        ring[ys.min()-y0:ys.max()-y0+1, xs.min()-x0:xs.max()-x0+1] = False
        touched = ring.any()
        if hard_iso and touched:
            continue
        # 3px 孤立度（非硬约束）
        x0b, x1b = max(0, xs.min() - 3), xs.max() + 4
        y0b, y1b = max(0, ys.min() - 3), ys.max() + 4
        region = dark[y0b:y1b, x0b:x1b]
        mask_self = (labeled[y0b:y1b, x0b:x1b] == i)
        outside = region & ~mask_self
        iso = outside.sum() / (region.size - mask_self.sum()) if region.size > mask_self.sum() else 1.0
        if iso <= iso_thr:
            out.append((int(xs.min()), int(ys.min()), w, h, s, i))
    return out


def height_histogram(dark, labeled, sizes):
    """全图字符块高度直方图（排除细线/超宽图形）"""
    from collections import Counter
    counts = Counter()
    for i in range(1, sizes.size + 1):
        s = int(sizes[i - 1])
        if s < 8 or s > 5000:
            continue
        ys, xs = np.where(labeled == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h > 60 or w > 200:
            continue
        counts[h] += 1
    return counts


def pick_label_height(hist):
    """从高度直方图挑标签高度峰：正文峰附近略高的峰；无则返回 None"""
    if not hist:
        return None
    body_peak = max(hist, key=hist.get)  # 最常见块高 = 正文文字
    # 找 body_peak 上方 2-8px 范围内、数量显著次于正文峰的高度
    best_h, best_score = None, 0
    for h in range(body_peak + 2, body_peak + 9):
        if h in hist and hist[h] > 0:
            score = hist[h]
            # 标签峰应比正文少很多但非零；挑"比相邻明显突出"的
            if score > best_score:
                best_score = score
                best_h = h
    return best_h if best_score > 0 else None


def find_anchor_A(dark, labeled, sizes, max_x=120, max_y=100):
    """找左上角最高孤立字母 = A 标签"""
    best, bh = None, 0
    for i in range(1, sizes.size + 1):
        s = int(sizes[i - 1])
        if s < 25 or s > 2000:
            continue
        ys, xs = np.where(labeled == i)
        if xs.min() > max_x or ys.min() > max_y:
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if w > 60:
            continue
        fill = s / (w * h)
        if fill < 0.2:
            continue
        if h > bh:
            bh = h
            best = i
    if best is None:
        return None
    ys, xs = np.where(labeled == best)
    return (int(xs.min()), int(ys.min())), bh


def find_labels(src, method='auto'):
    arr, dark = load(src=src)
    labeled, sizes = label_components(dark)

    if method in ('auto', 'anchor'):
        # 先找 A
        A_pos, A_h = find_anchor_A(dark, labeled, sizes)
        if A_pos:
            h_min, h_max = A_h - 2, A_h + 2
            cands = match_by_height(dark, labeled, sizes, h_min, h_max, hard_iso=True)
            if cands:
                # 全部同高孤立字母候选 = 标签集；去重按位置排序
                return [c[:2] for c in sorted(cands, key=lambda c: (c[1], c[0]))]
    if method in ('auto', 'histogram'):
        hist = height_histogram(dark, labeled, sizes)
        H = pick_label_height(hist)
        if H:
            cands = match_by_height(dark, labeled, sizes, H - 2, H + 2, hard_iso=True)
            if candidate_count_is_reasonable(cands, len(hist)):
                return [c[:2] for c in sorted(cands, key=lambda c: (c[1], c[0]))]
    return []  # 需要视觉或用更宽松阈值


def candidate_count_is_reasonable(cands, n_hist):
    """候选数应在合理标签范围（2-30 个），避免全是噪声"""
    return 2 <= len(cands) <= 30


def main():
    src = sys.argv[1]
    method = 'auto'
    out = 'labels.json'
    if '--method' in sys.argv:
        method = sys.argv[sys.argv.index('--method') + 1]
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    t0 = time.time()
    positions = find_labels(src, method)
    if not positions:
        print('[!] 未自动定位到标签。可能: 标签字号不统一 / 图形元素多。'
              '请用 vision 人工认字母 + 写 labels.json，或检查高度直方图。')
        sys.exit(1)
    # 按行分组命名 A,B,C...
    positions.sort(key=lambda p: (p[1], p[0]))
    rows = []
    for p in positions:
        if rows and abs(p[1] - rows[-1][-1][1]) < 40:
            rows[-1].append(p)
        else:
            rows.append([p])
    labels = {}
    idx = 0
    for row in rows:
        row.sort(key=lambda p: p[0])
        for p in row:
            labels[chr(65 + idx)] = list(p)
            idx += 1
    with io.open(out, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    print(f'[ok] 定位 {len(labels)} 个标签 → {out} ({time.time()-t0:.1f}s)')
    print('标签:', labels)
    print('提示: 用字形 print 验证身份(A尖顶+B鼓包+C开口+D竖+圆+E三横+F双横+G圆底口)，'
          '错命中的在 labels.json 手工修正。')


if __name__ == '__main__':
    main()
