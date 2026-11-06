#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""find_labels.py — 自动定位面板标签字母（像素方法，不依赖视觉模型坐标）

两种策略自动切换：
  1. 高度直方图法（简单图）：全图文字高度统计，若存在明显高于正文的"标签高度峰"，
     用 [峰±2px] + 单字母宽阈过滤，一次命中全部标签。
  2. 先找 A → 按 A 高度模板匹配（复杂图）：高度直方图无单一标签峰时，
     左上角(x<120,y<100)最高孤立字母 = A，以 A 高度做全图模板匹配找其余同名高标签。
   3. 括号小写标签法（2026-11-06 新增）：标签是 (a)/(b)/(c) 时，找 3 个相邻窄连通域
     （左括号+字母+右括号），按行优先排序。适用整页渲染图/小写带括号标签。

用法:
  python find_labels.py <image> [--method auto|histogram|anchor|paren] [--out labels.json]
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
    """找左上角最高隔离大写字母 = A 标签

    排除细长竖条/横条（轴、图例条、分隔线）——它们高度大但高宽比极端（非字母）。
    字母高宽比通常 0.8-1.6。
    """
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
        # 排除细长条（非字母）：高宽比 > 2.2 或 宽高比 > 3
        if h / w > 2.2 or w / h > 3:
            continue
        fill = s / (w * h)
        if fill < 0.2 or fill > 0.9:
            continue
        if h > bh:
            bh = h
            best = i
    if best is None:
        return None
    ys, xs = np.where(labeled == best)
    return (int(xs.min()), int(ys.min())), bh


def find_paren_labels(dark, labeled, sizes, max_y=200):
    """检测 (a)/(b)/(c) 带括号小写标签。

    形态：3 个相邻窄连通域 = 左括号 + 小写字母 + 右括号（多数期刊图的标准写法）。
    Figure 1/3 实测：每个字符约 4-10px 宽、12-20px 高，三者 x 间距 <14px、y 中心差 <4px。

    返回 [(x, y), ...]，按行优先排序（行 = y//20）；找不到返回 []。
    """
    comps = []
    for i in range(1, sizes.size + 1):
        s = int(sizes[i - 1])
        if s < 6 or s > 500:
            continue
        ys, xs = np.where(labeled == i)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        if not (2 <= w <= 12 and 9 <= h <= 26):
            continue
        comps.append({"x": int(xs.min()), "y": int(ys.min()), "w": w, "h": h,
                      "cy": float(ys.mean()), "cx": float(xs.mean())})
    if not comps:
        return []

    # 按 y 带 + x 顺序聚类：同一行、x 相邻（gap <= 14px）的归一组
    comps.sort(key=lambda c: (c["cy"], c["x"]))
    groups = []
    for c in comps:
        placed = False
        for g in groups:
            if abs(g["cy"] - c["cy"]) <= 4.0 and 0 <= (c["x"] - g["x1"]) <= 14:
                g["items"].append(c)
                g["x1"] = max(g["x1"], c["x"] + c["w"])
                g["cy"] = sum(it["cy"] for it in g["items"]) / len(g["items"])
                placed = True
                break
        if not placed:
            groups.append({"items": [c], "x0": c["x"], "x1": c["x"] + c["w"], "cy": c["cy"]})

    out = []
    for g in groups:
        if len(g["items"]) != 3:      # 必须是「左括号+字母+右括号」三连
            continue
        width = g["x1"] - g["x0"]
        if not (12 <= width <= 48):   # (a) 整体宽度
            continue
        heights = [it["h"] for it in g["items"]]
        if max(heights) - min(heights) > 8:   # 同组字符高度应接近
            continue
        out.append((g["x0"], int(min(it["y"] for it in g["items"]))))
    out.sort(key=lambda p: (p[1] // 20, p[0]))
    return out


def find_labels_by_anchor(dark, labeled, sizes, anchor_region, tol=2, iso_gap=12):
    """AI 找左上角 A → 模板匹配全图（2026-11-06 新策略，已修两 bug）。

    与 find_anchor_A 的区别：
    - find_anchor_A 靠"左上角最高孤立**大写**字母"（对小写带括号 (a) 失效）
    - 本函数由 **AI 给 anchor 区域框**（如 [90,10,180,60]），在框内取标签作模板
      （尺寸 + 形态），再全图匹配同尺寸的连通域 → 合并括号组 → 得到所有标签位置。

    修的两个 bug（首版实测失败原因）：
    1. 模板选最小面积 → 选到噪点。现改为：面积 40-200 且宽高比 0.2-3 优先
    2. 孤立性检查把「(a) 的括号和字母」当成邻居排除。现改为：同括号组（x 间距<15、
       y 中心差<8）不算邻居

    返回：[(x, y), ...] 按行优先排序；找不到返回 []。
    """
    # 1) 收集所有合格小连通域
    comps = []
    for i in range(1, sizes.size + 1):
        s = int(sizes[i - 1])
        if s < 20 or s > 900:
            continue
        ys, xs = np.where(labeled == i)
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if not (3 <= w <= 30 and 8 <= h <= 34):
            continue
        comps.append({"i": i, "x": int(xs.min()), "y": int(ys.min()), "w": w, "h": h,
                      "cx": float(xs.mean()), "cy": float(ys.mean())})
    if not comps:
        return []

    # 2) anchor 区域内取模板：优先"面积 40-200 且宽高比合理"的（标签特征）
    ax0, ay0, ax1, ay1 = anchor_region
    inside = [c for c in comps
              if ax0 <= c["x"] and c["x"] + c["w"] <= ax1 and ay0 <= c["y"] and c["y"] + c["h"] <= ay1]
    if not inside:
        return []
    good = [c for c in inside if 40 <= c["w"] * c["h"] <= 200 and 0.2 <= c["w"] / c["h"] <= 3]
    tpl = good[0] if good else min(inside, key=lambda c: c["w"] * c["h"])
    tw, th = tpl["w"], tpl["h"]

    # 3) 全图匹配同尺寸（同图字号一致）
    matches = [c for c in comps if abs(c["w"] - tw) <= tol and abs(c["h"] - th) <= tol]
    if not matches:
        return []

    # 4) 过滤明显干扰：页眉/页脚（x 贴右边缘 或 y 贴底）
    W = dark.shape[1]
    Hh = dark.shape[0]
    matches = [c for c in matches if c["x"] < W - 40 and c["y"] < Hh - 60]

    # 5) 合并同一括号组（(a) 的 「(」「a」「)」是 3 个连通域）
    def same_group(a, b):
        # 用中心 y 判断同行（括号和小写字母高度不同，左上角 y 可能差 2-4px）
        return abs(a["cy"] - b["cy"]) <= 10 and 0 <= (b["x"] - (a["x"] + a["w"])) <= 20

    groups = []
    for c in sorted(matches, key=lambda c: (c["cy"], c["x"])):
        placed = False
        for g in groups:
            if same_group(g["last"], c):
                g["items"].append(c)
                g["last"] = c
                g["x1"] = max(g["x1"], c["x"] + c["w"])
                placed = True
                break
        if not placed:
            groups.append({"items": [c], "last": c, "x0": c["x"], "x1": c["x"] + c["w"]})

    # 6) 每组取最左字符的左上角；要求组宽 12-48（完整括号组）或单字符宽 8-16（大写字母）
    labels = []
    for g in groups:
        first = min(g["items"], key=lambda c: c["x"])
        gw = g["x1"] - g["x0"]
        if not (12 <= gw <= 48 or 8 <= gw <= 16):
            continue
        labels.append((first["x"], first["y"]))
    labels.sort(key=lambda p: (p[1] // 20, p[0]))
    return labels


def find_labels(src, method='auto', anchor_region=None):
    arr, dark = load(src=src)
    labeled, sizes = label_components(dark)

    if anchor_region is not None:
        # AI 给了左上角 A 的区域框 → 模板匹配（新策略，优先）
        cands = find_labels_by_anchor(dark, labeled, sizes, anchor_region)
        if cands:
            return cands
    if method in ('auto', 'anchor'):
        # 先找 A（find_anchor_A 找不到时返回 None，必须先判空再解包）
        res = find_anchor_A(dark, labeled, sizes)
        if res:
            A_pos, A_h = res
            h_min, h_max = A_h - 2, A_h + 2
            cands = match_by_height(dark, labeled, sizes, h_min, h_max, hard_iso=True)
            if cands:
                # 全部同高孤立字母候选 = 标签集；去重按位置排序
                return [c[:2] for c in sorted(cands, key=lambda c: (c[1], c[0]))]
    if method in ('auto', 'paren'):
        # (a)/(b) 带括号小写标签（2026-11-06 新增）：3 个相邻窄连通域
        paren = find_paren_labels(dark, labeled, sizes)
        if paren:
            return paren
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
    anchor_region = None
    if '--anchor-region' in sys.argv:
        anchor_region = json.loads(sys.argv[sys.argv.index('--anchor-region') + 1])
        print('  AI anchor 区域: %s' % (anchor_region,))
    positions = find_labels(src, method, anchor_region=anchor_region)
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
