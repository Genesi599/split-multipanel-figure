# -*- coding: utf-8 -*-
"""贯穿整图的弹性分割线（seam）——完全铺满切分，折线避让墨迹。

与 split_panels.py 的 crop 框区别：
- crop 框：每个 panel 各画 4 条边（相邻面板之间有两条不重合的线、留白）
- seam 线：**相邻面板共享一条贯穿线**，从图的一边走到另一边，用 DP（seam carving）
  在整个区间内选代价最小的路径 —— 有空白就走空白，无空白走最窄处。

代价设计：
- 落在内容像素上：`penalty`（默认 1000，极力避开）
- 偏离理想边界的水平/垂直位移：`dev_weight`（默认 1.0，第二优先级）

用法：
    python seam_lines.py <图.png> --panels panels.json --out outdir [--penalty 1000]

panels.json 格式（完全铺满、共边、左闭右开）：
    {"A":[0,0,346,294],"B":[346,0,627,294],...,"E":[0,294,1180,1380]}

输出：
    seam_分割线.png    原图 + 贯穿折线
    seam_路径.json     每条线的逐像素路径
    seam_报告.txt      每条线穿过内容像素的数量（0 = 全程走空白）
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

CONTENT_THRESHOLD = 225   # 任一 RGB 通道 < 225 判为内容


def load_content(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    content = (arr < CONTENT_THRESHOLD).any(axis=2)
    return arr, content


def load_labels(content):
    """连通域标签（用于「切块惩罚」）。"""
    try:
        from scipy import ndimage
    except ImportError:
        return None
    lab, _ = ndimage.label(content, structure=np.ones((3, 3), bool))
    return lab


def seam_vertical(content, x_ideal, y0, y1, half_width, penalty=1000.0, dev_weight=1.0, step=3,
                  lab=None, cut_penalty=3000.0):
    """竖向 seam：从 (x,y0) 到 (x',y1) 逐行下降，每行可左/中/右移动一格。

    返回 [(x, y), ...] 路径；代价 = 内容像素惩罚 + 偏离理想 x 的位移。
    """
    H, W = content.shape
    xs = list(range(max(0, x_ideal - half_width), min(W, x_ideal + half_width + 1)))
    if not xs:
        return [(x_ideal, y) for y in range(y0, y1 + 1)]

    INF = 1e18
    prev, back = {}, []
    for x in xs:
        c = abs(x - x_ideal) * dev_weight
        if content[y0, x]:
            c += penalty
        prev[x] = c

    for y in range(y0 + 1, y1 + 1):
        cur, bp = {}, {}
        row = content[y]
        for x in xs:
            best, bx = INF, x
            for dx in range(-step, step + 1):
                px = x + dx
                if px in prev and prev[px] < best:
                    best, bx = prev[px], px
            if best >= INF:
                continue
            c = best + abs(x - x_ideal) * dev_weight * 0.02
            if row[x]:
                c += penalty
            # 切块惩罚：左右两侧同属一个连通域 → 这条线会把该色块切成两半
            if lab is not None and 0 < x < W - 1:
                l_, r_ = lab[y, x - 1], lab[y, x + 1]
                if l_ != 0 and l_ == r_:
                    c += cut_penalty
            cur[x] = c
            bp[x] = bx
        if not cur:
            break
        back.append(bp)
        prev = cur

    if not prev:
        return [(x_ideal, y) for y in range(y0, y1 + 1)]
    x = min(prev, key=prev.get)
    path = [x]
    for bp in reversed(back):
        x = bp.get(x, x)
        path.append(x)
    path.reverse()
    return [(path[i], y0 + i) for i in range(len(path))]


def seam_horizontal(content, y_ideal, x0, x1, half_width, penalty=1000.0, dev_weight=1.0, step=3,
                    lab=None, cut_penalty=3000.0):
    """横向 seam：从 (x0,y) 到 (x1,y') 逐列右移，每列可上/中/下移动一格。"""
    H, W = content.shape
    ys = list(range(max(0, y_ideal - half_width), min(H, y_ideal + half_width + 1)))
    if not ys:
        return [(x, y_ideal) for x in range(x0, x1 + 1)]

    INF = 1e18
    prev, back = {}, []
    for y in ys:
        c = abs(y - y_ideal) * dev_weight
        if content[y, x0]:
            c += penalty
        prev[y] = c

    for x in range(x0 + 1, x1 + 1):
        cur, bp = {}, {}
        col = content[:, x]
        for y in ys:
            best, by = INF, y
            for dy in range(-step, step + 1):
                py = y + dy
                if py in prev and prev[py] < best:
                    best, by = prev[py], py
            if best >= INF:
                continue
            c = best + abs(y - y_ideal) * dev_weight * 0.02
            if col[y]:
                c += penalty
            if lab is not None and 0 < y < H - 1:
                u_, d_ = lab[y - 1, x], lab[y + 1, x]
                if u_ != 0 and u_ == d_:
                    c += cut_penalty
            cur[y] = c
            bp[y] = by
        if not cur:
            break
        back.append(bp)
        prev = cur

    if not prev:
        return [(x, y_ideal) for x in range(x0, x1 + 1)]
    y = min(prev, key=prev.get)
    path = [y]
    for bp in reversed(back):
        y = bp.get(y, y)
        path.append(y)
    path.reverse()
    return [(x0 + i, path[i]) for i in range(len(path))]


def count_content_hits(path, content):
    """统计路径穿过的内容像素数。"""
    hits = 0
    H, W = content.shape
    for x, y in path:
        if 0 <= y < H and 0 <= x < W and content[y, x]:
            hits += 1
    return hits


def paint_by_seam(arr, content, panels, paths):
    """按 seam 边界把墨迹染色（通用：任意 panel 数、任意布局）。

    算法：逐行扫描。每一行取出"在该行生效"的竖线，用它们把 x 轴切成若干段，
    每段归给**包含该段中心点**的 panel 框（panels 是完全铺满的初始框）。
    这样跨行大 panel、不规则布局都能正确处理，无需假设谁是"下方大 panel"。
    """
    H, W = content.shape
    import colorsys
    names = sorted(panels)
    colors = {}
    for i, n in enumerate(names):
        r, g, b = colorsys.hsv_to_rgb(i / len(names), 0.75, 0.95)
        colors[n] = (int(r * 255), int(g * 255), int(b * 255))

    v_lines = [r for r in paths.values() if r["type"] == "vertical"]
    # 每条竖线：(x 逐行位置, y 起, y 止)
    vinfo = []
    for rec in v_lines:
        m = {}
        for x, y in rec["path"]:
            m.setdefault(y, x)
        vinfo.append(m)

    def owner_at(x, y):
        # 该行生效的竖线位置
        cuts = sorted(m[y] for m in vinfo if y in m)
        seg = sum(1 for c in cuts if x >= c)
        # 用切段序号确定"列"，再按 x 落在哪个 panel 框判定
        xs = [0] + [c for c in cuts] + [W]
        if seg >= len(xs) - 1:
            seg = len(xs) - 2
        cx = (xs[seg] + xs[seg + 1]) // 2
        for n in names:
            x0, y0, x1, y1 = panels[n]
            if x0 <= cx < x1 and y0 <= y < y1:
                return n
        # 兜底：按到各 panel 中心的距离
        return min(names, key=lambda n: (
            abs((panels[n][0] + panels[n][2]) / 2 - x) +
            abs((panels[n][1] + panels[n][3]) / 2 - y)))

    out = np.full((H, W, 3), 255, dtype=np.uint8)
    stats = {n: 0 for n in names}
    ys, xs = np.where(content)
    for y, x in zip(ys, xs):
        n = owner_at(int(x), int(y))
        out[y, x] = np.array(colors[n], dtype=np.uint8)
        stats[n] += 1
    return out, stats, colors


def main():
    ap = argparse.ArgumentParser(description="贯穿整图的弹性分割线（折线避让墨迹）")
    ap.add_argument("img")
    ap.add_argument("--panels", required=True, help="完全铺满的 panel 框 json")
    ap.add_argument("--out", default="seam_out")
    ap.add_argument("--penalty", type=float, default=1000.0, help="内容像素惩罚，默认 1000")
    ap.add_argument("--half-width", type=int, default=45, help="seam 搜索半宽 px，默认 45")
    ap.add_argument("--paint", action="store_true", help="输出墨迹染色图（每 panel 一色）")
    ap.add_argument("--labels", help="标签锚点 json（label_refine 输出）；给定后标签框也作为障碍，"
                                     "确保每个标签框完整落在自己的 panel 内")
    ap.add_argument("--label-box", default="21x16", help="标签组宽x高，默认 21x16")
    ap.add_argument("--label-pad", type=int, default=4, help="标签框外扩，默认 4")
    ap.add_argument("--step", type=int, default=3,
                    help="每行允许横移 px，默认 3（太小绕不开内容块，太大会锯齿）")
    ap.add_argument("--dilate", type=int, default=2,
                    help="内容障碍膨胀 px，默认 2（给 seam 提前绕行空间，避免临到块边缘才躲）")
    args = ap.parse_args()

    arr, content = load_content(args.img)
    H, W = content.shape
    panels = json.loads(pathlib.Path(args.panels).read_text(encoding="utf-8"))

    # 标签框作为障碍：seam 必须绕开，否则标签会被切到邻 panel
    label_boxes = {}
    if args.labels:
        labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
        try:
            bw, bh = (int(v) for v in args.label_box.lower().split("x"))
        except ValueError:
            bw, bh = 21, 16
        pad = args.label_pad
        for name, (lx, ly) in labels.items():
            bx0 = max(0, int(lx) - pad); by0 = max(0, int(ly) - pad)
            bx1 = min(W, int(lx) + bw + pad); by1 = min(H, int(ly) + bh + pad)
            label_boxes[name] = (bx0, by0, bx1, by1)
            content[by0:by1, bx0:bx1] = True      # 障碍
        print("  标签框作为障碍: %s" % ", ".join(
            "(%s)%s" % (k.lower(), v) for k, v in sorted(label_boxes.items())))

    lab = load_labels(content)
    if lab is not None:
        print("  连通域 %d 个（用于切块惩罚）" % int(lab.max()))

    if args.dilate > 0:
        # 膨胀障碍：让 seam 提前开始绕行，而不是临到块边缘才被迫穿过
        d = args.dilate
        padded = np.zeros((H + 2 * d, W + 2 * d), dtype=bool)
        padded[d:d + H, d:d + W] = content
        grown = padded.copy()
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                grown |= np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
        content = grown[d:d + H, d:d + W]
        print("  障碍膨胀 %d px（内容像素 %d → %d）" % (d, int(padded.sum()), int(content.sum())))
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print("  图: %dx%d, %d 个 panel" % (W, H, len(panels)))

    # 从 panel 框推出需要哪些分割线：相邻且共边的面板之间画一条
    # 竖向线：同一行（y 范围相同）相邻面板的共享 x
    v_cuts, h_cuts = [], []
    names = sorted(panels)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ax0, ay0, ax1, ay1 = panels[a]
            bx0, by0, bx1, by1 = panels[b]
            # 竖向共边：x 相接 且 y 范围**有重叠**（跨行大 panel 只与行内 panel 部分重叠）
            if abs(ax1 - bx0) <= 1 and min(ay1, by1) > max(ay0, by0):
                v_cuts.append((a, b, int((ax1 + bx0) // 2),
                               max(ay0, by0), min(ay1, by1)))
            elif abs(bx1 - ax0) <= 1 and min(ay1, by1) > max(ay0, by0):
                v_cuts.append((b, a, int((bx1 + ax0) // 2),
                               max(ay0, by0), min(ay1, by1)))
            # 横向共边：上方面板底 == 下方面板顶，x 范围**只要有重叠**即可
            # （下方大 panel 通常比上方各块宽，取并集画一条贯穿线）
            elif abs(ay1 - by0) <= 1 and min(ax1, bx1) > max(ax0, bx0):
                h_cuts.append((a, b, int((ay1 + by0) // 2),
                               min(ax0, bx0), max(ax1, bx1)))
            elif abs(by1 - ay0) <= 1 and min(ax1, bx1) > max(ax0, bx0):
                h_cuts.append((b, a, int((by1 + ay0) // 2),
                               min(ax0, bx0), max(ax1, bx1)))

    # 竖向 ideal 修正：右 panel 的标签必须落在线的右侧
    # （标签可能叠在左 panel 内容的水平范围内，例如 (c) 标签在 (b) 的 RPE 文字右侧但同一 y 带）
    if label_boxes:
        fixed = []
        for above, below, x_ideal, ya, yb in v_cuts:
            right = below if below in label_boxes else (above if above in label_boxes else None)
            left = above if right == below else (below if right == above else None)
            if right and right in label_boxes:
                lb = label_boxes[right]
                # 线必须在右 panel 标签框左侧
                x_ideal = min(x_ideal, lb[0] - 1)
            fixed.append((above, below, x_ideal, ya, yb))
        v_cuts = fixed

    # 同 x 的竖线合并：y 取并集（大 panel 跨多行时，同一 x 会有多段）
    v_merged = {}
    for above, below, x_cut, ya, yb in v_cuts:
        if x_cut in v_merged:
            rec = v_merged[x_cut]
            rec["ya"] = min(rec["ya"], ya)
            rec["yb"] = max(rec["yb"], yb)
            if below not in rec["belows"]:
                rec["belows"].append(below)
        else:
            v_merged[x_cut] = {"ya": ya, "yb": yb,
                               "aboves": [above], "belows": [below]}
    v_cuts = [(rec["aboves"][0], "+".join(rec["belows"]), x, rec["ya"], rec["yb"])
              for x, rec in sorted(v_merged.items())]

    v_cuts.sort(key=lambda t: t[2])
    # 同 y 的横线合并：x 取并集，between 合并
    h_merged = {}
    for above, below, y_cut, xa, xb in h_cuts:
        if y_cut in h_merged:
            rec = h_merged[y_cut]
            rec["x0"] = min(rec["x0"], xa)
            rec["x1"] = max(rec["x1"], xb)
            if below not in rec["belows"]:
                rec["belows"].append(below)
        else:
            h_merged[y_cut] = {"x0": xa, "x1": xb,
                               "aboves": [above], "belows": [below]}
    h_cuts = [(rec["aboves"][0], "+".join(rec["belows"]), y, rec["x0"], rec["x1"])
              for y, rec in sorted(h_merged.items())]
    v_cuts.sort(key=lambda t: t[2])
    print("  需要 %d 条竖线、%d 条横线（已按 y 合并）" % (len(v_cuts), len(h_cuts)))

    # 计算 seam
    paths = {}
    for above, below, x_ideal, ya, yb in v_cuts:
        p = seam_vertical(content, x_ideal, ya, yb - 1, args.half_width,
                          penalty=args.penalty, step=args.step, lab=lab)
        key = "V_%s|%s" % (above, below)
        paths[key] = {"type": "vertical", "between": [above, below],
                      "ideal_x": x_ideal, "path": p,
                      "content_hits": count_content_hits(p, content)}
        print("     %s: x_ideal=%d, 路径 x %d~%d, 穿内容 %d px" % (
            key, x_ideal, min(q[0] for q in p), max(q[0] for q in p),
            paths[key]["content_hits"]))

    for above, below, y_ideal, xa, xb in h_cuts:
        p = seam_horizontal(content, y_ideal, xa, xb - 1, args.half_width,
                            penalty=args.penalty, step=args.step, lab=lab)
        key = "H_%s|%s" % (above, below)
        paths[key] = {"type": "horizontal", "between": [above, below],
                      "ideal_y": y_ideal, "path": p,
                      "content_hits": count_content_hits(p, content)}
        print("     %s: y_ideal=%d, 路径 y %d~%d, 穿内容 %d px" % (
            key, y_ideal, min(q[1] for q in p), max(q[1] for q in p),
            paths[key]["content_hits"]))

    # 画线
    img = Image.open(args.img).convert("RGB")
    draw = ImageDraw.Draw(img)
    RED = (220, 30, 30)
    for key, rec in paths.items():
        pts = rec["path"]
        draw.line([tuple(q) for q in pts], fill=RED, width=3)

    # 画标签框（各 panel 颜色，与分割线一起显示）
    if label_boxes:
        import colorsys
        for i, n in enumerate(sorted(label_boxes)):
            r, g, b = colorsys.hsv_to_rgb(i / max(1, len(label_boxes)), 0.85, 0.85)
            col = (int(r * 255), int(g * 255), int(b * 255))
            x0, y0, x1, y1 = label_boxes[n]
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=3)

    out_png = outdir / "seam_分割线.png"
    img.save(out_png)
    (outdir / "seam_路径.json").write_text(
        json.dumps(paths, ensure_ascii=False), encoding="utf-8")

    total_hits = sum(r["content_hits"] for r in paths.values())
    report = []
    report.append("图: %dx%d" % (W, H))
    report.append("竖线 %d 条 / 横线 %d 条" % (len(v_cuts), len(h_cuts)))
    for key, rec in paths.items():
        report.append("%s: 穿内容 %d px" % (key, rec["content_hits"]))
    report.append("合计穿内容像素: %d" % total_hits)
    (outdir / "seam_报告.txt").write_text("\n".join(report), encoding="utf-8")

    if args.paint:
        painted, stats, colors = paint_by_seam(arr, content, panels, paths)
        pp = outdir / "seam_墨迹染色.png"
        Image.fromarray(painted).save(pp)
        print("\n  ✓ %s" % pp)
        print("  每 panel 墨迹像素数（各 panel 墨迹不重叠、无越界）：")
        for n in sorted(stats):
            print("     (%s): %6d px" % (n.lower(), stats[n]))
        print("     合计: %d px（= 全图内容像素数）" % sum(stats.values()))

    print("\n  ✓ %s" % out_png)
    print("  ✓ seam_路径.json / seam_报告.txt")
    print("  合计穿内容像素: %d %s" % (total_hits, "（全程走空白）" if total_hits == 0 else "⚠ 有线穿过墨迹"))


if __name__ == "__main__":
    main()
