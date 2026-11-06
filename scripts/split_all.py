#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split_all.py — 组图拆分「一条命令」入口

设计目标：把「找标签 → 定范围 → 校验 → 归属 → 出子图 → 出联排」压成一次调用，
减少多轮往返。所有判据与 panel_ranges.py / panel_split.py 完全相同，不降低准确性。

两种用法
--------
1) 我不知道标签在哪 → 先列候选（一次看图选号）
       python split_all.py 图.png --candidates --outdir out
   会输出 out/candidates.png（每个候选画编号框）+ 打印编号表；
   记下哪些号是面板字母，然后用 --pick 跑正式流程。

2) 已知标签 → 一把跑完
       python split_all.py 图.png --pick 3,9,14,20,25,31,38 --outdir out
       python split_all.py 图.png --labels labels.json --outdir out
       python split_all.py 图.png --pick ... --stack L:M --outdir out

输出（outdir/）
    candidates.png      候选编号图（仅 --candidates）
    ranges.png          范围预览（半透明块 + 标签青框）
    ranges.json         panel 范围
    02_色块归属.png / 03_实心区域.png / 04_区域叠原图.png
    panel_<X>.png       子图
    all_panels.png      子图联排（一张图看全部）
    report.txt          全部校验数字
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

INK_THR = 235
HERE = pathlib.Path(__file__).resolve().parent


def font(sz=14):
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def load(path, thr=INK_THR):
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr, (arr < thr).any(axis=2)


def letter_candidates(ink, dark=None, max_iso=2, min_fill=0.35, min_area=40,
                      max_w=34, max_h=34, isolated_pad=3):
    """面板标签候选：**孤立 + 深色 + 粗体**的小块，尺寸按全图众数自适应。

    不同期刊/图的标签字号差别很大（实测 h=12、17、23 都有），所以**不写死尺寸范围**：
    先收所有孤立小块，再取「尺寸众数那一类」——同一幅图的面板字母必然同字号。
    """
    H, W = ink.shape
    lab, _ = ndimage.label(ink, structure=np.ones((3, 3), bool))
    objs = ndimage.find_objects(lab)
    raw = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        a = int((lab[sl] == i + 1).sum())
        w = sl[1].stop - sl[1].start
        h = sl[0].stop - sl[0].start
        if not (4 <= w <= max_w and 8 <= h <= max_h and a >= min_area):
            continue
        if a < min_fill * w * h:
            continue
        if dark is not None:
            comp = (lab[sl] == i + 1)
            if not (dark[sl][comp].any() and dark[sl][comp].mean() >= 0.35):
                continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        ya, yb = max(0, y0 - isolated_pad), min(H, y1 + isolated_pad)
        xa, xb = max(0, x0 - isolated_pad), min(W, x1 + isolated_pad)
        box = ink[ya:yb, xa:xb].copy()
        box[y0 - ya:y1 - ya, x0 - xa:x1 - xa] = False
        if int(box.sum()) > max_iso:
            continue
        raw.append((x0, y0, x1 - 1, y1 - 1, a))
    if not raw:
        return []
    # 尺寸众数（按 4px 分桶，避免抗锯齿造成 ±1 抖动）
    from collections import Counter
    buckets = Counter(((c[2] - c[0]) // 4, (c[3] - c[1]) // 4) for c in raw)
    (bw, bh), _ = buckets.most_common(1)[0]
    out = []
    for c in raw:
        w = c[2] - c[0] + 1
        h = c[3] - c[1] + 1
        if abs(w // 4 - bw) <= 1 and abs(h // 4 - bh) <= 1:
            iso = 0
            out.append((c[0], c[1], c[2], c[3], c[4], iso))
    out.sort(key=lambda c: (c[1] // 40, c[0]))
    return out


def draw_candidates(img, cands, out):
    W, H = img.size
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    f = font(13)
    for k, (x0, y0, x1, y1, a, iso) in enumerate(cands, 1):
        d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(220, 0, 0), width=1)
        d.text((x0 - 1, max(0, y0 - 15)), str(k), fill=(0, 0, 200), font=f)
    im.save(out)
    return im.size


def build_contact(outdir, names, cols=4, cell=(430, 320)):
    tiles = []
    for n in names:
        fp = outdir / ("panel_%s.png" % n)
        if not fp.exists():
            continue
        im = Image.open(fp).convert("RGB")
        im.thumbnail(cell, Image.LANCZOS)
        tiles.append((n, im))
    if not tiles:
        return None
    TW, TH = cell[0] + 10, cell[1] + 26
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TW, rows * TH), "white")
    d = ImageDraw.Draw(sheet)
    f = font(16)
    for i, (n, im) in enumerate(tiles):
        px, py = (i % cols) * TW + 5, (i // cols) * TH + 22
        d.text((px, py - 18), "(%s)" % n.lower(), fill="black", font=f)
        sheet.paste(im, (px, py))
        d.rectangle([px, py, px + im.size[0], py + im.size[1]], outline=(190, 190, 190), width=1)
    f2 = outdir / "all_panels.png"
    sheet.save(f2)
    return f2


def main():
    ap = argparse.ArgumentParser(description="组图拆分一条命令入口")
    ap.add_argument("img")
    ap.add_argument("--outdir", default="split_out")
    ap.add_argument("--candidates", action="store_true", help="只列标签候选（编号图 + 表）")
    ap.add_argument("--pick", help="候选编号，逗号分隔（顺序即 a,b,c…）")
    ap.add_argument("--labels", help="labels.json（{字母: [x,y]} 或 bbox）")
    ap.add_argument("--stack", action="append", default=[], help="同列上下叠，如 --stack L:M")
    ap.add_argument("--row-tol", type=int, default=45)
    ap.add_argument("--thr", type=int, default=INK_THR)
    ap.add_argument("--cols", type=int, default=4, help="联排列数")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.img).convert("RGB")
    arr, ink = load(args.img, args.thr)
    dark = arr.max(axis=2) < 130
    H, W = ink.shape
    print("  图 %dx%d" % (W, H))

    # ── 模式 1：列候选 ──
    if args.candidates or (not args.pick and not args.labels):
        cands = letter_candidates(ink, dark)
        p = outdir / "candidates.png"
        draw_candidates(img, cands, p)
        print("  字母候选 %d 个 → %s" % (len(cands), p))
        print("  编号表（按行从上到下）：")
        for k, (x0, y0, x1, y1, a, iso) in enumerate(cands, 1):
            print("     #%-3d x%-4d y%-4d  %dx%d  area=%-4d 周边墨迹=%d" % (k, x0, y0, x1 - x0 + 1, y1 - y0 + 1, a, iso))
        print("\n  下一步：挑出面板字母的编号 → python split_all.py 图.png --pick 1,5,9,...")
        if not args.pick and not args.labels:
            return 0

    # ── 模式 2：一把跑完 ──
    if args.labels:
        labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
    else:
        cands = letter_candidates(ink, dark)
        idx = [int(s) for s in args.pick.split(",") if s.strip()]
        labels = {}
        for k, i in enumerate(idx):
            if not (1 <= i <= len(cands)):
                print("  !! 编号 %d 超范围（共 %d 个候选）" % (i, len(cands)))
                return 2
            x0, y0, x1, y1, a, iso = cands[i - 1]
            labels[chr(ord("A") + k)] = [x0, y0, x1, y1]
        print("  由编号得到 %d 个标签: %s" % (len(labels), " ".join(
            "%s(%d,%d)" % (n.lower(), v[0], v[1]) for n, v in labels.items())))

    lab_file = outdir / "labels.json"
    lab_file.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    py = sys.executable
    # ① 范围推导 + 校验
    cmd = [py, str(HERE / "panel_ranges.py"), args.img, str(lab_file),
           "--out", str(outdir / "ranges.json"), "--preview", str(outdir / "ranges.png"),
           "--row-tol", str(args.row_tol), "--thr", str(args.thr)]
    for s in args.stack:
        cmd += ["--stack", s]
    print("\n  ── ① 定范围 ──")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print((r.stdout or "").rstrip())
    if r.returncode != 0:
        print("  !! 范围校验未通过：按上面提示改（手工改 ranges.json 后加 --panels 重跑 panel_ranges）")
        (outdir / "report.txt").write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
        return r.returncode

    # ② 归属 + 切子图
    print("\n  ── ② 归属 + 切子图 ──")
    cmd2 = [py, str(HERE / "panel_split.py"), args.img, str(lab_file),
            "--out", str(outdir), "--panels", str(outdir / "ranges.json")]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8")
    if r2.returncode not in (0, 1) and (r2.stderr or "").strip():
        print("  !! panel_split 报错:\n" + (r2.stderr or "").strip()[:1500])
    keep = [l for l in (r2.stdout or "").splitlines()
            if any(k in l for k in ("标签框不合格", "分组", "色块（", "无主", "越界合计",
                                    "面积合计", "不合格部件", "panel_", "全部检查", "!!"))]
    print("\n".join(keep))

    # ③ 联排
    names = sorted(labels)
    f = build_contact(outdir, names, cols=args.cols)
    if f:
        print("\n  → %s（子图联排，一张图看全部）" % f)

    (outdir / "report.txt").write_text((r.stdout or "") + "\n" + (r2.stdout or ""), encoding="utf-8")
    ok = (r.returncode == 0 and r2.returncode == 0)
    print("\n  %s" % ("✅ 全部通过" if ok else "!! 有未通过项，见 report.txt"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())