# -*- coding: utf-8 -*-
"""标签位置精修：视觉模型认身份 + 像素检测定坐标。

适用场景（2026-11-06 实测补充）：
- 标签是 **小写带括号** `(a)/(b)/(c)`（Journal Club / 多数期刊图的标准写法）
- 标签字号接近正文（ratio < 1.3），`label_detect.py --ocr` 的高度筛会漏掉它们，
  easyocr 对「带括号小写字母」识别也差（实测只认出页眉的 I/L/O）
- 图是**整页 PDF 渲染图**（含正文/页眉/图注），全图 OCR 噪声极大

原理（与 SKILL.md §0 核心原则一致）：
- **视觉模型只认身份**（哪个位置大概是 a、b、c）—— 它给的坐标有 1-3px 偏差，不可靠
- **像素检测定坐标**：在估计点 ±R 内找**暗色小连通域**（标签墨迹），取最接近的那个
- 两者结合 = 身份可靠 + 坐标精确

用法：
    # 1. 先由视觉模型（读图）给出估计位置 {"A": [x, y], ...}
    python label_refine.py <img> --estimate '{"A":[112,105],"B":[345,105]}' --out labels.json
    # 或从文件读
    python label_refine.py <img> --estimate-file est.json --out labels.json

    # 2. 再跑 split_panels.py（自动模式，不要 --panels）
    python split_panels.py <img> labels.json --out outdir

输出 labels.json 格式与 split_panels.py 兼容：{"A": [x, y], ...}（像素，标签左上角）。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def refine(arr, est, radius=30, dark_thr=120, min_w=3, max_w=45, min_h=3, max_h=32,
           min_area=8, max_area=600):
    """对每个估计点，在 radius 内找最接近的暗色小连通域。

    返回 (precise, report)；report 记录每个标签的匹配距离与尺寸，便于判断可靠性。
    """
    try:
        from scipy import ndimage
    except ImportError:
        raise SystemExit("需要 scipy：pip install scipy")

    H, W = arr.shape[:2]
    dark = arr.max(axis=2) < dark_thr
    lab, n = ndimage.label(dark)
    objs = ndimage.find_objects(lab)

    precise, report = {}, {}
    for name, (ex, ey) in est.items():
        best = None
        for sl in objs:
            if sl is None:
                continue
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            w, h = x1 - x0, y1 - y0
            if not (min_w <= w <= max_w and min_h <= h <= max_h and min_area <= w * h <= max_area):
                continue
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            d = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            if d > radius:
                continue
            if best is None or d < best[0]:
                best = (d, x0, y0, w, h)
        if best:
            d, x0, y0, w, h = best
            precise[name] = [int(x0), int(y0)]
            report[name] = {"from": [ex, ey], "to": [int(x0), int(y0)],
                            "size": [w, h], "dist_px": round(d, 1), "matched": True}
        else:
            precise[name] = [int(ex), int(ey)]
            report[name] = {"from": [ex, ey], "to": [int(ex), int(ey)],
                            "size": None, "dist_px": None, "matched": False}
    return precise, report


def main():
    ap = argparse.ArgumentParser(description="视觉估计 + 像素精修标签坐标")
    ap.add_argument("img", help="图区 PNG（已裁掉正文/图注）")
    ap.add_argument("--estimate", help='视觉模型给的估计位置 JSON 字符串，如 \'{"A":[112,105]}\'')
    ap.add_argument("--estimate-file", help="或从文件读估计位置")
    ap.add_argument("--out", default="labels.json", help="输出 labels.json")
    ap.add_argument("--radius", type=int, default=30, help="搜索半径 px（默认 30）")
    ap.add_argument("--dark", type=int, default=120, help="暗色阈值（默认 120）")
    ap.add_argument("--report", action="store_true", help="打印每个标签的匹配报告")
    args = ap.parse_args()

    if args.estimate_file:
        est = json.loads(pathlib.Path(args.estimate_file).read_text(encoding="utf-8"))
    elif args.estimate:
        est = json.loads(args.estimate)
    else:
        raise SystemExit("需要 --estimate 或 --estimate-file（视觉模型给出的估计位置）")

    arr = load(args.img)
    print("  图区: %dx%d, 估计标签 %d 个" % (arr.shape[1], arr.shape[0], len(est)))
    precise, report = refine(arr, est, radius=args.radius, dark_thr=args.dark)

    ok = sum(1 for r in report.values() if r["matched"])
    print("  精修结果: %d/%d 命中" % (ok, len(est)))
    if args.report:
        for name, r in report.items():
            if r["matched"]:
                print("    %s: %s → %s  尺寸%s  距离 %.1fpx" % (
                    name, r["from"], r["to"], r["size"], r["dist_px"]))
            else:
                print("    %s: %s → ⚠ 未找到（沿用估计）" % (name, r["from"]))

    # 一致性检查：命中的标签尺寸应相近（同一张图的标签字号一致）
    sizes = [r["size"] for r in report.values() if r["matched"] and r["size"]]
    if len(sizes) >= 3:
        hs = [s[1] for s in sizes]
        if max(hs) - min(hs) > 6:
            print("  ⚠ 标签高度差异较大（%d~%d px），可能有误匹配，请 --report 核对" % (min(hs), max(hs)))

    pathlib.Path(args.out).write_text(json.dumps(precise, ensure_ascii=False), encoding="utf-8")
    print("  ✓ 已写 %s" % args.out)
    print("  下一步: python split_panels.py <img> %s --out <outdir>" % args.out)


if __name__ == "__main__":
    main()
