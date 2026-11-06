#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""panel_ranges.py — 由标签位置推导/校验每个 panel 的范围

核心规则（2026-11-06 定稿，来源：19-panel 图实战）
------------------------------------------------
1. **标签在子图左上角** → 范围左上角 = 标签左上角 − 边距。
2. **行带取全宽横向白沟**：整行无墨迹的连续行段（宽 ≥ MIN_GUTTER）即行界候选，
   取离标签行最近的那个白沟中点作为行界。
3. **列界吸附竖向白沟**：同一行内相邻两个 panel，边界 = 两者之间最近的竖向白沟中点
   （在该行 y 跨度内整列无墨迹）。
4. **边界线必须零墨迹**：每条边那一行/列不允许有任何墨迹像素（否则就是切到内容）。
5. **标签必须完整落在自己的范围内**：标签墨迹 bbox 四边都严格在范围内部。
6. **互不重叠 + 覆盖全图**：面积合计 = W×H。

用法
----
    # 由「标签左上角」自动推导范围并校验
    python panel_ranges.py 图.png labels.json --out ranges.json --preview ranges.png

    # 只校验已有的范围（不推导）
    python panel_ranges.py 图.png labels.json --panels ranges.json --preview ranges.png

    # 行带/列界也可手工给（针对 L/M 上下叠这类特殊版式）
    python panel_ranges.py 图.png labels.json --rows 276,536,754,944 --stack L:M --out ranges.json

labels.json: {"A": [[x0,y0,x1,y1], ...], ...}  或  {"A": [x,y], ...}
             （前者=标签墨迹 bbox，更准；后者=标签点，脚本用 INK_THR 量出 bbox）
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

INK_THR = 235        # 任一通道 < 该值 = 墨迹（含抗锯齿）
MIN_GUTTER = 2       # 白沟最小宽度
MARGIN = 4           # 范围相对标签的外扩边距


def load_ink(path, thr=INK_THR):
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr, (arr < thr).any(axis=2)


def measure_label_boxes(ink, labels):
    """把 [x,y] 或 bbox 统一成 bbox；[x,y] 时在附近取最近的连通墨迹块。"""
    H, W = ink.shape
    lab, _ = ndimage.label(ink, structure=np.ones((3, 3), bool))
    objs = ndimage.find_objects(lab)
    out = {}
    for n, v in labels.items():
        if isinstance(v, (list, tuple)) and len(v) == 4 and isinstance(v[0], (list, tuple)) is False:
            out[n] = [int(a) for a in v]
            continue
        x, y = int(v[0]), int(v[1])
        best = None
        for i, sl in enumerate(objs):
            if sl is None:
                continue
            a = int((lab[sl] == i + 1).sum())
            if a < 25:
                continue
            bx0, bx1 = sl[1].start, sl[1].stop - 1
            by0, by1 = sl[0].start, sl[0].stop - 1
            if not (x - 6 <= bx0 <= x + 26 and y - 6 <= by0 <= y + 26):
                continue
            d = abs(bx0 - x) + abs(by0 - y)
            if best is None or d < best[0]:
                best = (d, bx0, by0, bx1, by1)
        out[n] = list(best[1:]) if best else [x, y, x + 10, y + 12]
    return out


def h_gutters(ink, min_w=MIN_GUTTER):
    """整行无墨迹的连续行段 → [(y0, y1)]"""
    rows = ink.any(axis=1)
    out, s = [], None
    for y in range(len(rows) + 1):
        v = rows[y] if y < len(rows) else True
        if not v:
            if s is None:
                s = y
        else:
            if s is not None:
                if y - s >= min_w:
                    out.append((s, y - 1))
                s = None
    return out


def v_gutters(ink, y0, y1, x0, x1, min_w=MIN_GUTTER):
    """在 [y0,y1) × [x0,x1) 内，整列无墨迹的连续列段 → [(x0, x1)]"""
    cols = ink[y0:y1, x0:x1].any(axis=0)
    out, s = [], None
    for i in range(len(cols) + 1):
        v = cols[i] if i < len(cols) else True
        if not v:
            if s is None:
                s = i
        else:
            if s is not None:
                if i - s >= min_w:
                    out.append((s + x0, i - 1 + x0))
                s = None
    return out


def snap(val, gutters, tol=40):
    cand = [(a + b) // 2 for a, b in gutters if abs((a + b) // 2 - val) <= tol]
    return min(cand, key=lambda m: abs(m - val)) if cand else val


def derive(ink, lb, stack=(), row_tol=45):
    """由标签 bbox 推导范围。

    规则：
      * 行聚类（y 差 ≤ row_tol 视为同行）
      * 行分界 = 全宽横向白沟中点（取离“下一行标签顶”最近的）；首行顶=0，末行底=H
      * 列分界 = 该行 y 跨度内的竖向白沟中点（取离“右邻标签左沿”最近的）；
        每行首列左=0，末列右=W
      * 分界只算一次，作为相邻两块的公共边 → 天然无重叠、无空隙
      * stack 里的组合（如 L/M 同列上下叠）按 y 再切一刀
    """
    H, W = ink.shape
    names = sorted(lb)
    order = sorted(names, key=lambda n: (lb[n][1], lb[n][0]))
    rows = [[order[0]]]
    for n in order[1:]:
        if lb[n][1] - min(lb[m][1] for m in rows[-1]) <= row_tol:
            rows[-1].append(n)
        else:
            rows.append([n])
    # 行分界
    hg = h_gutters(ink)
    rb = [0]
    for i in range(len(rows) - 1):
        want = min(lb[n][1] for n in rows[i + 1]) - MARGIN
        rb.append(int(snap(want, hg)))
    rb.append(H)
    rb = [max(rb[i], 0) for i in range(len(rb))]
    for i in range(len(rb) - 1):        # 保序、保正宽
        rb[i + 1] = max(rb[i + 1], rb[i] + 1)
    rb[-1] = H
    stack_map = {}
    for grp in stack:
        for n in grp:
            stack_map[n] = grp[0]
    P = {}
    for i, r in enumerate(rows):
        y0, y1 = rb[i], rb[i + 1]
        chains, seen = [], set()
        for n in sorted(r, key=lambda m: lb[m][0]):
            if n in seen:
                continue
            grp = stack_map.get(n)
            if grp and grp in r:
                ch = [m for m in r if stack_map.get(m) == grp]
                chains.append(ch); seen.update(ch)
            else:
                chains.append([n]); seen.add(n)
        # 列分界
        cb = [0]
        for k in range(len(chains) - 1):
            want = min(lb[m][0] for m in chains[k + 1]) - MARGIN
            cb.append(int(snap(want, v_gutters(ink, y0, y1, max(0, want - 60), min(W, want + 60)))))
        cb.append(W)
        for k in range(len(cb) - 1):
            cb[k + 1] = max(cb[k + 1], cb[k] + 1)
        cb[-1] = W
        for k, ch in enumerate(chains):
            if len(ch) > 1:                      # 同列上下叠：再按标签 y 切
                ch_sorted = sorted(ch, key=lambda m: lb[m][1])
                yb = [y0]
                for j in range(len(ch_sorted) - 1):
                    want = lb[ch_sorted[j + 1]][1] - MARGIN
                    yb.append(int(snap(want, h_gutters(ink[max(0, want - 60):min(H, want + 60)]) and
                                       [(a + max(0, want - 60), b + max(0, want - 60))
                                        for a, b in h_gutters(ink[max(0, want - 60):min(H, want + 60)])])))
                yb.append(y1)
                for j in range(len(yb) - 1):
                    yb[j + 1] = max(yb[j + 1], yb[j] + 1)
                yb[-1] = y1
                for j, m in enumerate(ch_sorted):
                    P[m] = [cb[k], yb[j], cb[k + 1], yb[j + 1]]
            else:
                P[ch[0]] = [cb[k], y0, cb[k + 1], y1]
    return P, rows


def as_boxes(v):
    if isinstance(v, (list, tuple)) and len(v) == 4 and not isinstance(v[0], (list, tuple)):
        return [tuple(int(a) for a in v)]
    return [tuple(int(a) for a in b) for b in v]


def verify(ink, P, lb):
    """返回 [(name, 问题列表)]；空列表 = 通过。支持一个 panel 多个矩形。"""
    H, W = ink.shape
    res = []
    for n in sorted(P):
        prob = []
        inside_any = False
        for (x0, y0, x1, y1) in as_boxes(P[n]):
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                prob.append("范围越界/宽高非法")
                continue
            lx = int(ink[y0:y1, x0].sum()); rx = int(ink[y0:y1, min(W - 1, x1)].sum())
            ty = int(ink[y0, x0:x1].sum()); by = int(ink[min(H - 1, y1), x0:x1].sum())
            if lx + rx + ty + by:
                prob.append("边界压墨迹(x%d-%d y%d-%d 左%d右%d上%d下%d)" % (x0, x1, y0, y1, lx, rx, ty, by))
            a = lb[n]
            if x0 <= a[0] and a[2] < x1 and y0 <= a[1] and a[3] < y1:
                inside_any = True
        if not inside_any:
            prob.append("标签不在范围内")
        if prob:
            res.append((n, prob))
    # 覆盖 / 重叠（按 JSON 顺序优先级）
    own = np.full((H, W), -1, dtype=np.int16)
    order = list(P.keys())
    for k, n in enumerate(order):
        for (x0, y0, x1, y1) in as_boxes(P[n]):
            sub = own[y0:y1, x0:x1]
            sub[sub == -1] = k
    un = int((own == -1).sum())
    if un:
        res.append(("__global__", ["未覆盖像素 %d（范围没铺满全图）" % un]))
    return res


def render(arr, P, lb, out):
    H, W = arr.shape[:2]
    if any(lb[n][3] >= H - 60 for n in lb):
        canvas = arr.copy()
    else:
        canvas = np.full((H + 26, W, 3), 255, dtype=np.uint8)
        canvas[:H] = arr
    im = Image.fromarray(canvas)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d0 = ImageDraw.Draw(ov)
    PAL = [(228, 26, 28), (255, 127, 0), (255, 191, 0), (60, 150, 60), (0, 170, 190),
           (40, 90, 200), (150, 60, 200), (230, 60, 160), (140, 90, 30), (96, 0, 160),
           (0, 160, 96), (200, 0, 80), (0, 120, 255), (180, 120, 0), (120, 0, 200),
           (0, 150, 150), (200, 120, 0), (100, 100, 0), (0, 80, 160), (160, 0, 0)]
    names = sorted(P)
    for i, n in enumerate(names):
        for (x0, y0, x1, y1) in as_boxes(P[n]):
            d0.rectangle([x0, y0, x1 - 1, y1 - 1], fill=PAL[i % len(PAL)] + (30,))
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        f = ImageFont.load_default()
    for i, n in enumerate(names):
        for (x0, y0, x1, y1) in as_boxes(P[n]):
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=PAL[i % len(PAL)], width=2)
        a = lb[n]
        d.rectangle([a[0] - 1, a[1] - 1, a[2] + 1, a[3] + 1], outline=(0, 255, 255), width=1)
    im.save(out)
    return im.size


def main():
    ap = argparse.ArgumentParser(description="由标签推导/校验 panel 范围")
    ap.add_argument("img")
    ap.add_argument("labels")
    ap.add_argument("--panels", help="已有范围 json（只校验，不推导）")
    ap.add_argument("--out", help="输出推导出的范围 json")
    ap.add_argument("--preview", help="输出范围预览图")
    ap.add_argument("--stack", action="append", default=[],
                    help="同列上下叠的组合，如 --stack L:M（可多次）")
    ap.add_argument("--thr", type=int, default=INK_THR)
    ap.add_argument("--row-tol", type=int, default=45, help="同一行标签 y 容差，默认 45")
    args = ap.parse_args()

    arr, ink = load_ink(args.img, args.thr)
    H, W = ink.shape
    labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
    lb = measure_label_boxes(ink, labels)
    print("  图 %dx%d ；标签 %d 个" % (W, H, len(lb)))

    if args.panels:
        _pj = json.loads(pathlib.Path(args.panels).read_text(encoding="utf-8"))
        P = {k.upper(): (list(v) if isinstance(v[0], (list, tuple)) else [int(a) for a in v]) for k, v in _pj.items()}
    else:
        stack = [tuple(s.split(":")) for s in args.stack]
        P, rows = derive(ink, lb, stack, args.row_tol)
        P = {k.upper(): P[k] for k in P}
        print("  推断 %d 行：%s" % (len(rows), " | ".join(" ".join(n.lower() for n in r) for r in rows)))
        for n in sorted(P):
            for b in as_boxes(P[n]):
                print("     (%s) x%d-%d y%d-%d %dx%d" % (n.lower(), b[0], b[2], b[1], b[3], b[2] - b[0], b[3] - b[1]))
        # 自动修复：标签不在自己范围内（标签落在邻 panel 的空档里）→ 加「标签条」矩形并前置
        strips = [n for n in P if not any(
            b[0] <= lb[n][0] and lb[n][2] < b[2] and b[1] <= lb[n][1] and lb[n][3] < b[3]
            for b in as_boxes(P[n]))]
        if strips:
            print("")
            print("  自动修复：%s 的标签落在邻 panel 的 x 范围内，加标签条矩形并前置优先级"
                  % " ".join(n.lower() for n in sorted(strips)))
            for n in strips:
                a = lb[n]
                strip = [a[0] - 2, a[1] - 2, a[2] + 2, a[3] + 2]
                P[n] = [strip] + as_boxes(P[n])
            P = {n: P[n] for n in sorted(strips)} | {n: P[n] for n in P if n not in strips}
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(P, ensure_ascii=False, indent=1), encoding="utf-8")
            print("  → %s" % args.out)

    print("")
    print("  校验（边界零墨迹 / 标签在范围内 / 无重叠 / 面积守恒）:")
    bad = 0
    for n, prob in verify(ink, P, lb):
        if prob:
            bad += 1
            print("     %-10s !! %s" % (n.lower(), "；".join(prob)))
    if not bad:
        print("     全部通过 ✅")
    if args.preview:
        s = render(arr, P, lb, args.preview)
        print("  → %s (%dx%d)  半透明=范围  青色小框=标签墨迹" % (args.preview, s[0], s[1]))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())