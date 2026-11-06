#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""panel_split.py — 文献多 panel 组图 → 独立子图

职责划分（别搞反）
------------------
* **视觉模型看图给区间**：每个 panel 在图上占哪块，由你看图写进 panels.json。
  允许一个 panel 写**多个矩形**（阶梯版式必需），也允许矩形之间重叠 —— 按 JSON 里的
  **先后顺序**首匹配，先写的优先。
* **脚本只做机械活**：切墨迹连通块 → 按区间归属 → 双向自查 → 实心化 → 面积守恒 → 切子图。

脚本**不猜版式、不吸附、不推断**。版式判断永远是视觉的活。

用法
----
    python panel_split.py 图.png labels.json --out out --panels panels.json [--check checks.json]

panels.json  —— 看图为每个 panel 给的区间，顺序即优先级：
    {
      "A": [0, 0, 412, 340],
      "B": [412, 0, 773, 345],
      "E": [[0, 872, 300, 1350], [300, 895, 415, 1350], [415, 912, 512, 1350]],
      ...
    }
labels.json  —— label_refine.py 的输出 {"A": [x, y], ...}
checks.json  —— 可选，逐部件覆盖率自查 {"A": {"字母": [x0,y0,x1,y1], ...}, ...}

输出（<out>/）
    01_标签框.png  02_色块归属.png  03_实心区域.png  04_区域叠原图.png
    panel_<字母>.png  report.txt
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

CONTENT_THR = 225
MIN_BLOCK_AREA = 5


def load(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr, (arr < CONTENT_THR).any(axis=2)


def safe_palette(orig, n):
    """挑 n 个原图里根本不存在的颜色 —— 否则自查会与原图像素撞色。"""
    cands = [(255, 0, 128), (255, 128, 0), (255, 208, 0), (0, 160, 0),
             (0, 192, 192), (0, 128, 255), (128, 0, 255), (192, 0, 192),
             (160, 96, 0), (96, 0, 160), (0, 255, 0), (255, 0, 255),
             (0, 255, 128), (255, 255, 0), (128, 255, 0), (0, 96, 160),
             (255, 96, 0), (0, 160, 96), (0, 0, 160), (160, 0, 0),
             (200, 120, 255), (120, 200, 255), (255, 120, 200), (120, 255, 200)]
    out = []
    for c in cands:
        if int(np.all(orig == np.array(c, dtype=orig.dtype), axis=2).sum()) == 0:
            out.append(c)
        if len(out) == n:
            return out
    raise RuntimeError("安全色不足，请扩充候选表")


def as_boxes(val):
    """值可以是 box=[x0,y0,x1,y1]，也可以是 box 列表（阶梯版式）。"""
    if isinstance(val, (list, tuple)) and len(val) == 4 and all(isinstance(v, (int, float)) for v in val):
        return [tuple(int(v) for v in val)]
    return [tuple(int(v) for v in b) for b in val]


def main():
    ap = argparse.ArgumentParser(description="多 panel 组图拆分（视觉给区间，脚本做归属与自查）")
    ap.add_argument("img")
    ap.add_argument("labels")
    ap.add_argument("--out", default="panel_out")
    ap.add_argument("--panels", required=True, help="看图为每个 panel 给的区间（可多矩形，顺序即优先级）")
    ap.add_argument("--check", help="逐部件覆盖率自查 json（可选）")
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.img).convert("RGB")
    orig, content = load(args.img)
    H, W = content.shape
    labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
    panels = json.loads(pathlib.Path(args.panels).read_text(encoding="utf-8"))
    order = list(panels.keys())
    boxes = {n: as_boxes(panels[n]) for n in order}

    report = []
    def say(s=""):
        print(s); report.append(s)

    say("== panel_split ==")
    say("图 %dx%d ；标签 %d ；区间里的 panel %d（顺序即优先级）: %s"
        % (W, H, len(labels), len(order), " ".join(n.lower() for n in order)))
    missing = [n for n in labels if n not in panels]
    extra = [n for n in panels if n not in labels]
    if missing: say("!! 这些标签没有对应区间: %s" % " ".join(missing))
    if extra:   say("!! 这些区间没有对应标签: %s" % " ".join(extra))
    if missing:
        (outdir / "report.txt").write_text("\n".join(report), encoding="utf-8")
        return 2

    # 01 标签框
    pv = img.copy(); d = ImageDraw.Draw(pv)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for n in labels:
        x, y = labels[n]
        d.rectangle([max(0, x - 4), max(0, y - 4), x + 25, y + 20], outline=(220, 30, 30), width=2)
        d.text((max(0, x - 4), y + 22), "%s(%d,%d)" % (n.lower(), x, y), fill=(220, 30, 30), font=font)
    pv.save(outdir / "01_标签框.png")
    say("→ 01_标签框.png（先人眼核对每个框套住字母）")

    say("")
    say("区间（视觉给定）:")
    for n in order:
        for bi, b in enumerate(boxes[n]):
            say("   (%s)%s x%d-%d y%d-%d %s" % (n.lower(), "" if len(boxes[n]) == 1 else "#%d" % (bi + 1), *b,
                                                "" if len(boxes[n]) == 1 else "（多矩形，阶梯版式）"))

    # 归属
    def owner_of(x, y):
        for n in order:
            for (x0, y0, x1, y1) in boxes[n]:
                if x0 <= x < x1 and y0 <= y < y1:
                    return n
        return None

    lab, _ = ndimage.label(content, structure=np.ones((3, 3), bool))
    objs = ndimage.find_objects(lab)
    owner = {}
    unmapped = 0
    for i, sl in enumerate(objs):
        if sl is None: continue
        y0, x0 = sl[0].start, sl[1].start
        if int((lab[sl] == i + 1).sum()) < MIN_BLOCK_AREA: continue
        n = owner_of(x0, y0)
        if n is None: unmapped += 1; continue
        owner[i + 1] = n

    counts = {n: 0 for n in order}
    bbox = {n: [10 ** 9, 10 ** 9, -1, -1] for n in order}
    for bid, n in owner.items():
        counts[n] += 1
        sl = objs[bid - 1]; b = bbox[n]
        b[0] = min(b[0], sl[1].start); b[1] = min(b[1], sl[0].start)
        b[2] = max(b[2], sl[1].stop - 1); b[3] = max(b[3], sl[0].stop - 1)

    say("")
    say("色块归属: " + "  ".join("(%s)%d" % (n.lower(), counts[n]) for n in order)
        + "  合计 %d ; 落在所有区间之外的块 %d" % (sum(counts.values()), unmapped))
    empty = [n for n in order if counts[n] == 0]
    if empty:
        say("!! 这些 panel 没分到块 → 区间给错了: %s" % " ".join(n.lower() for n in empty))

    pal = safe_palette(orig, len(order)); color = dict(zip(order, pal))
    mask_by = {}
    out = orig.copy()
    for n in order:
        m = np.zeros((H, W), bool)
        for bid, ow in owner.items():
            if ow == n: m |= (lab == bid)
        mask_by[n] = m
        out[m] = color[n]
    Image.fromarray(out).save(outdir / "02_色块归属.png")
    say("→ 02_色块归属.png（人眼确认：一个词没有两种颜色、没有邻panel串进来）")

    # 区间交错报告（不是错误：阶梯版式靠优先级裁决）
    say("")
    say("区间交错（正常现象；由 JSON 顺序裁决，先写的优先）:")
    inter = 0
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            ov = 0
            for (ax0, ay0, ax1, ay1) in boxes[a]:
                for (bx0, by0, bx1, by1) in boxes[b]:
                    w = min(ax1, bx1) - max(ax0, bx0)
                    h = min(ay1, by1) - max(ay0, by0)
                    if w > 0 and h > 0: ov += w * h
            if ov:
                inter += 1
                say("   %s ↔ %s: 重叠 %d px²（%s 优先）" % (a.lower(), b.lower(), ov, a.lower()))
    if not inter:
        say("   无重叠")

    # 假阴（可选）
    if args.check:
        checks = json.loads(pathlib.Path(args.check).read_text(encoding="utf-8"))
        say("")
        say("假阴自查（逐部件覆盖率，须 >97%）:")
        low = 0
        for n, parts in checks.items():
            if n not in mask_by: continue
            for tag, (x0, y0, x1, y1) in parts.items():
                tot = int(content[y0:y1, x0:x1].sum())
                if tot == 0:
                    say("   (%s) %-14s 框内无墨迹（采样框要校正）" % (n.lower(), tag)); continue
                got = int((content[y0:y1, x0:x1] & mask_by[n][y0:y1, x0:x1]).sum())
                pct = 100.0 * got / tot
                if pct <= 97: low += 1
                say("   (%s) %-14s %5.1f%% %s" % (n.lower(), tag, pct, "OK" if pct > 97 else "!! 漏标"))
        say("   不达标部件数: %d" % low)

    # 实心化 + 互斥划分
    used_r = {}
    for n, m in mask_by.items():
        cur = m
        for R in (4, 8, 12, 16, 20, 26, 32, 40):
            c = ndimage.binary_closing(cur, structure=np.ones((2 * R + 1, 2 * R + 1), bool))
            cur, used_r[n] = c, R
            if ndimage.label(c, structure=np.ones((3, 3), bool))[1] == 1: break
    bmin = np.full((H, W), 1e18); best = np.full((H, W), -1, dtype=np.int16)
    for k, n in enumerate(order):
        dd = ndimage.distance_transform_edt(~mask_by[n])
        m = dd < bmin
        bmin[m] = dd[m]; best[m] = k

    solid = orig.copy()
    for k, n in enumerate(order):
        solid[best == k] = color[n]
    Image.fromarray(solid).save(outdir / "03_实心区域.png")
    Image.blend(img, Image.fromarray(solid), 0.42).save(outdir / "04_区域叠原图.png")

    say("")
    say("实心区域 + 面积守恒终检:")
    total = 0; bad_conn = 0
    for k, n in enumerate(order):
        m = (best == k); a = int(m.sum()); total += a
        ncc = ndimage.label(m, structure=np.ones((3, 3), bool))[1]
        if ncc != 1: bad_conn += 1
        say("   (%s) 闭运算R=%-2d 实心连通块=%d 面积=%d %s" % (n.lower(), used_r[n], ncc, a,
                                                             "OK" if ncc == 1 else "!! 不连通"))
    say("   面积合计 %d ; 全图 %d ; 差 %d %s" % (total, H * W, H * W - total,
                                                "OK 完全铺满" if total == H * W else "!! 漏格/叠格"))
    say("→ 03_实心区域.png / 04_区域叠原图.png（后者最适合人眼验收）")

    # 切子图
    if not args.no_crop:
        say("")
        say("子图（本 panel 墨迹 bbox +2px，只保留本 panel 像素）:")
        for k, n in enumerate(order):
            if counts[n] == 0: continue
            m = (best == k); b = bbox[n]; M = 2
            x0 = max(0, b[0] - M); y0 = max(0, b[1] - M)
            x1 = min(W, b[2] + 1 + M); y1 = min(H, b[3] + 1 + M)
            tile = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
            mm = m[y0:y1, x0:x1]
            tile[mm] = orig[y0:y1, x0:x1][mm]
            Image.fromarray(tile).save(outdir / ("panel_%s.png" % n))
            say("   (%s) panel_%s.png %dx%d bbox=(%d,%d,%d,%d)"
                % (n.lower(), n, x1 - x0, y1 - y0, x0, y0, x1, y1))

    (outdir / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n报告: %s" % (outdir / "report.txt"))
    ok = (unmapped == 0) and (total == H * W) and (bad_conn == 0) and not empty
    print("全部检查通过" if ok else "!! 有检查项未通过，见 report.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())