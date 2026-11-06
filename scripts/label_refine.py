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

用法（方案 A 区域框模式，推荐）：
    python label_refine.py <img> --regions '{"A":[80,10,260,100],...}' --out labels.json --report

用法（点模式，旧）：
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

    两轮策略（2026-11-06 改进）：
    1. 第一轮：每个估计点取最近的几个候选（不同尺寸）
    2. 用「出现最多的尺寸」作为模板（同一张图的标签字号一致）
    3. 第二轮：只接受尺寸在模板 ±tol 内的候选；不符合的标未命中（不硬塞错误对象）

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

    # 收集所有合格连通域（一次性，避免每标签重复扫描）
    comps = []
    for sl in objs:
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if min_w <= w <= max_w and min_h <= h <= max_h and min_area <= w * h <= max_area:
            comps.append((x0, y0, w, h, x0 + w / 2.0, y0 + h / 2.0))

    # 第一轮：每个估计点最近的 3 个候选
    first = {}
    for name, (ex, ey) in est.items():
        cands = []
        for c in comps:
            d = ((c[4] - ex) ** 2 + (c[5] - ey) ** 2) ** 0.5
            if d <= radius:
                cands.append((d, c))
        cands.sort(key=lambda t: t[0])
        first[name] = cands[:3]

    # 统计「最一致尺寸」作为模板：取每个标签最近候选的尺寸，找众数
    from collections import Counter
    size_counter = Counter()
    for name, cands in first.items():
        if cands:
            c = cands[0][1]
            size_counter[(c[2], c[3])] += 1
    template = size_counter.most_common(1)[0][0] if size_counter else None
    tol_w = max(2, int(round(template[0] * 0.5))) if template else 2
    tol_h = max(2, int(round(template[1] * 0.35))) if template else 2

    # 第二轮：只接受尺寸接近模板的候选
    precise, report = {}, {}
    for name, (ex, ey) in est.items():
        best = None
        if template:
            for d, c in first[name]:
                w, h = c[2], c[3]
                if abs(w - template[0]) <= tol_w and abs(h - template[1]) <= tol_h:
                    best = (d, c)
                    break
        else:
            if first[name]:
                best = first[name][0]
        if best:
            d, c = best
            precise[name] = [int(c[0]), int(c[1])]
            report[name] = {"from": [ex, ey], "to": [int(c[0]), int(c[1])],
                            "size": [c[2], c[3]], "dist_px": round(d, 1), "matched": True,
                            "template": template}
        else:
            precise[name] = [int(ex), int(ey)]
            report[name] = {"from": [ex, ey], "to": [int(ex), int(ey)],
                            "size": None, "dist_px": None, "matched": False,
                            "template": template}
    return precise, report


def refine_regions(arr, regions, dark_thr=120, min_w=3, max_w=45, min_h=8, max_h=40,
                   min_area=20, max_area=900):
    """方案 A（2026-11-06）：视觉模型给**区域框**而不是点，像素检测在框内找标签。

    为什么更鲁棒：框的容错性远大于点（±80px 的框 vs ±10px 的点）；
    视觉模型只需指出「标签大概在这个区域」，不需要精确坐标。

    区域格式：{"A": [x0, y0, x1, y1], ...}（像素，框可以宽松）

    流程：
    1. 收集全图合格的小连通域
    2. 对每个区域：筛出落在框内的候选
    3. 用「全图最常见尺寸」作模板（同图标签字号一致）
    4. 每个区域选：尺寸最接近模板、且离框中心最近的候选
    """
    try:
        from scipy import ndimage
    except ImportError:
        raise SystemExit("需要 scipy：pip install scipy")
    from collections import Counter

    H, W = arr.shape[:2]
    dark = arr.max(axis=2) < dark_thr
    lab, n = ndimage.label(dark)
    objs = ndimage.find_objects(lab)

    # 全图合格小连通域
    comps = []
    for sl in objs:
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if min_w <= w <= max_w and min_h <= h <= max_h and min_area <= w * h <= max_area:
            comps.append((x0, y0, w, h))

    # 模板：每个区域最近候选的尺寸众数
    per_region = {}
    for name, (rx0, ry0, rx1, ry1) in regions.items():
        inside = [c for c in comps
                  if rx0 <= c[0] and c[0] + c[2] <= rx1 and ry0 <= c[1] and c[1] + c[3] <= ry1]
        per_region[name] = inside
    size_counter = Counter()
    for name, inside in per_region.items():
        for c in inside:
            size_counter[(c[2], c[3])] += 1
    # 模板 = 出现最多的尺寸（同图标签字号一致，且比噪点更常见）
    template = size_counter.most_common(1)[0][0] if size_counter else None
    tol_w = max(2, int(round(template[0] * 0.5))) if template else 2
    tol_h = max(2, int(round(template[1] * 0.35))) if template else 2

    precise, report = {}, {}
    for name, (rx0, ry0, rx1, ry1) in regions.items():
        cx, cy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
        best = None
        for c in per_region[name]:
            w, h = c[2], c[3]
            if template and not (abs(w - template[0]) <= tol_w and abs(h - template[1]) <= tol_h):
                continue
            d = ((c[0] + w / 2.0 - cx) ** 2 + (c[1] + h / 2.0 - cy) ** 2) ** 0.5
            if best is None or d < best[0]:
                best = (d, c)
        if best:
            d, c = best
            # ⚠️ 标签 (a) 由「(」「a」「)」3 个连通域组成，检测到的常是**其中单个字符**。
            # 2026-11-06 实测：直接返回该字符坐标会让框整体偏右（落在字母或右括号上）。
            # 正解：在同一行（|cy 差|<6）、同尺寸（模板）、x 间距 <10px 的相邻字符里，
            #       向左一路收集，取最左字符的 x 作为整个括号组的起点。
            # 高度窗口必须**相对模板**：
            #   小写标签 (a)：模板 h≈10，「(」「)」h≈16 → 8..20 够用
            #   大写标签 (A)：模板 h≈22，「(」「)」h≈23 → 旧的固定 8..20 会把括号挡掉，
            #                 导致向左收集停在字母上，整框偏右约 8px（2026-11-06 实测）
            ht = max(1, c[3])
            h_lo, h_hi = max(6, int(ht * 0.55)), max(10, int(ht * 1.7))
            group = [c]
            for direction in (-1, 1):                # 先向左、再向右收集整组
                cur = c
                for _ in range(4):
                    cand = None
                    for o in comps:
                        if o is c or o in group:
                            continue
                        if not (h_lo <= o[3] <= h_hi):
                            continue
                        if direction < 0:
                            gap = cur[0] - (o[0] + o[2])
                        else:
                            gap = o[0] - (cur[0] + cur[2])
                        oy = (o[1] + o[3] / 2.0) - (c[1] + c[3] / 2.0)
                        if 0 < gap <= 10 and abs(oy) <= 6:
                            if cand is None or (o[0] < cand[0] if direction < 0 else o[0] > cand[0]):
                                cand = o
                    if cand is None:
                        break
                    group.append(cand)
                    cur = cand
            gx0 = min(g[0] for g in group)
            gy0 = min(g[1] for g in group)
            gx1 = max(g[0] + g[2] for g in group)
            gy1 = max(g[1] + g[3] for g in group)
            precise[name] = [int(gx0), int(gy0)]
            report[name] = {"region": [rx0, ry0, rx1, ry1], "to": [int(gx0), int(gy0)],
                            "box": [int(gx0), int(gy0), int(gx1), int(gy1)],
                            "group_chars": len(group),
                            "size": [c[2], c[3]], "dist_to_center": round(d, 1),
                            "matched": True, "template": template}
        else:
            precise[name] = [int(cx), int(cy)]
            report[name] = {"region": [rx0, ry0, rx1, ry1], "to": [int(cx), int(cy)],
                            "size": None, "dist_to_center": None,
                            "matched": False, "template": template}
    return precise, report


def render_preview(img_path, precise, pad, out_path, report=None):
    """把标签框画回原图：框四边各外扩 pad 像素、坐标文字放图外底部白条。

    这是标签验收的唯一可靠方式——只看坐标数值无法判断框是否套住字母。
    """
    from PIL import Image, ImageDraw, ImageFont
    fig = Image.open(img_path).convert("RGB")
    BAR_H = 46
    canvas = Image.new("RGB", (fig.size[0], fig.size[1] + BAR_H), "white")
    canvas.paste(fig, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 19)
    except Exception:
        font = ImageFont.load_default()
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
              "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]
    # 用像素测到的**真实组 bbox**画框（写死 21x16 会切掉括号，2026-11-06 实测）
    names = sorted(precise)
    for i, name in enumerate(names):
        x, y = precise[name]
        c = colors[i % len(colors)]
        bx = None
        if report and name in report and report[name].get("box"):
            bx = report[name]["box"]
        if bx:
            draw.rectangle([max(0, bx[0] - pad), max(0, bx[1] - pad),
                            bx[2] + pad, bx[3] + pad], outline=c, width=2)
        else:
            draw.rectangle([max(0, x - pad), max(0, y - pad),
                            x + 21 + pad, y + 16 + pad], outline=c, width=2)
    xpos, ybase = 16, fig.size[1] + 12
    for i, name in enumerate(names):
        c = colors[i % len(colors)]
        draw.rectangle([xpos, ybase - 2, xpos + 16, ybase + 10], outline=c, width=2)
        txt = "(%s) %d,%d" % (name.lower(), precise[name][0], precise[name][1])
        draw.text((xpos + 22, ybase - 4), txt, fill="black", font=font)
        xpos += 30 + len(txt) * 10
    canvas.save(out_path)
    return canvas.size


def main():
    ap = argparse.ArgumentParser(description="视觉估计 + 像素精修标签坐标")
    ap.add_argument("img", help="图区 PNG（已裁掉正文/图注）")
    ap.add_argument("--estimate", help='视觉模型给的估计位置 JSON 字符串，如 \'{"A":[112,105]}\'')
    ap.add_argument("--estimate-file", help="或从文件读估计位置")
    ap.add_argument("--regions", help='方案A：视觉模型给区域框 JSON，如 \'{"A":[80,10,260,100]}\'')
    ap.add_argument("--regions-file", help="或从文件读区域框")
    ap.add_argument("--out", default="labels.json", help="输出 labels.json")
    ap.add_argument("--radius", type=int, default=30, help="搜索半径 px（默认 30）")
    ap.add_argument("--dark", type=int, default=120, help="暗色阈值（默认 120）")
    ap.add_argument("--report", action="store_true", help="打印每个标签的匹配报告")
    ap.add_argument("--preview", help="输出标注图路径（框外扩 --pad 像素 + 图外坐标条）")
    ap.add_argument("--pad", type=int, default=4, help="标注框外扩像素，默认 4（不与墨迹重叠）")
    args = ap.parse_args()

    arr = load(args.img)
    print("  图区: %dx%d" % (arr.shape[1], arr.shape[0]))

    # 方案 A：区域框模式（更鲁棒）
    if args.regions_file or args.regions:
        regions = json.loads(pathlib.Path(args.regions_file).read_text(encoding="utf-8")) \
            if args.regions_file else json.loads(args.regions)
        print("  区域框模式: %d 个区域" % len(regions))
        precise, report = refine_regions(arr, regions, dark_thr=args.dark)
        ok = sum(1 for r in report.values() if r["matched"])
        print("  精修结果: %d/%d 命中 (模板尺寸 %s)" % (ok, len(regions), report[list(report)[0]]["template"] if report else None))
        if args.report:
            for name, r in report.items():
                if r["matched"]:
                    print("    %s: 区域%s → %s  尺寸%s" % (name, r["region"], r["to"], r["size"]))
                else:
                    print("    %s: 区域%s → ⚠ 未找到" % (name, r["region"]))
        pathlib.Path(args.out).write_text(json.dumps(precise, ensure_ascii=False), encoding="utf-8")
        print("  ✓ 已写 %s" % args.out)
        if args.preview:
            size = render_preview(args.img, precise, args.pad, args.preview, report)
            print("  ✓ 标注图 %s (%dx%d)，框外扩 %dpx" % (args.preview, size[0], size[1], args.pad))
            print("  ⚠ 必须看这张图核对每个框是否套住字母，才算完成")
        print("  下一步: python split_panels.py <img> %s --out <outdir>" % args.out)
        return

    # 点模式
    if args.estimate_file:
        est = json.loads(pathlib.Path(args.estimate_file).read_text(encoding="utf-8"))
    elif args.estimate:
        est = json.loads(args.estimate)
    else:
        raise SystemExit("需要 --estimate / --estimate-file（点）或 --regions / --regions-file（区域框）")

    print("  估计标签 %d 个" % len(est))
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
