#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""panel_split.py — 文献多 panel 组图 → 独立子图

职责划分
--------
* **视觉模型看图给区间**：每个 panel 占哪块写进 panels.json（可多矩形、可重叠，按 JSON 顺序首匹配）。
* **脚本只做机械活**：切墨迹 → 合并成「词/图形单元」→ 归属 → 上色 → 实心化 → 面积守恒 → 切子图。

三个已验证的坑，本脚本在算法层直接规避（不再靠调边界）
------------------------------------------------------
1. **小连通块被丢弃**（散点图里 1–2 px 的点、图例小圆点）→ 本脚本**不做最小面积过滤**，
   每个连通组都参与归属。
2. **彩色像素被判成"非墨迹"**（有 255 通道的浅色点，如 RGB(252,209,210)）→ 墨迹判据用
   **任一通道 < 225**；上色时**直接刷该组的全部墨迹像素**，不再用「最大通道 < 250」当门槛
   （那个门槛只能用来补浅灰抗锯齿笔画）。
3. **词被区间边界切碎**（如 `UMAP_1` 的下划线在字母基线下方 2 px，落到相邻 panel）→
   **先按 (3×5) 形态学膨胀把字符合成词**，再用**组内像素多数票**决定归属（不是拿左上角一个点），
   1–2 px 的越界无法翻转整词。

用法
----
    python panel_split.py 图.png labels.json --out out --panels panels.json [--check checks.json] [--min-area 1]

panels.json:  {"A": [x0,y0,x1,y1], "E": [[...],[...]], ...}   顺序即优先级
labels.json:  {"A": [x, y], ...}（label_refine.py 输出，仅用于无 panels 时的预览与提示）
checks.json:  {"A": {"部件名": [x0,y0,x1,y1]}, ...}  可选，逐部件覆盖率自查

输出：01_标签框.png / 02_色块归属.png / 03_实心区域.png / 04_区域叠原图.png / panel_<字母>.png / report.txt
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

STRICT_THR = 225          # 任一通道 < 该值 = 墨迹（含彩色点）
LOOSE_THR = 250           # 最大通道 < 该值 = 浅灰抗锯齿（仅用于补笔画）
GROUP_KERNEL = (3, 5)     # 形态学膨胀核（高,宽）：把字符合成"词"
PAINT_PAD = 2             # 补浅灰笔画时的邻域半径


def load(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    strict = (arr < STRICT_THR).any(axis=2)
    loose = arr.max(axis=2) < LOOSE_THR
    return arr, strict, loose


def safe_palette(orig, n):
    """挑 n 个原图里根本不存在的颜色，否则自查时与原图像素撞色。"""
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
    if isinstance(val, (list, tuple)) and len(val) == 4 and all(isinstance(v, (int, float)) for v in val):
        return [tuple(int(v) for v in val)]
    return [tuple(int(v) for v in b) for b in val]


def group_units(strict, kernel=GROUP_KERNEL, min_area=1):
    """把墨迹按形态学膨胀合并成"单元"（词 / 图形），返回 (group_label, [(bbox, mask)], groups_total)。"""
    grown = ndimage.binary_dilation(strict, structure=np.ones(kernel, bool))
    glab, gn = ndimage.label(grown, structure=np.ones((3, 3), bool))
    objs = ndimage.find_objects(glab)
    units = []
    for gi, sl in enumerate(objs):
        if sl is None:
            continue
        m = (glab[sl] == gi + 1) & strict[sl]
        a = int(m.sum())
        if a < min_area:
            continue
        units.append((sl, m, a))
    return glab, units


def assign_units(units, boxes, order, W, H):
    """按「组内像素多数票」归属：统计该组墨迹像素落在哪个 panel 的区间里最多。

    用多数票而不是左上角单点，1–2 px 越界无法翻转整词。
    """
    owner = []
    for sl, m, a in units:
        ys, xs = np.nonzero(m)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        votes = {}
        # 逐像素投票（组通常很小，直接算）
        for y, x in zip(ys, xs):
            for n in order:
                hit = False
                for (x0, y0, x1, y1) in boxes[n]:
                    if x0 <= x < x1 and y0 <= y < y1:
                        votes[n] = votes.get(n, 0) + 1
                        hit = True
                        break
                if hit:
                    break
        if votes:
            owner.append(max(votes, key=lambda k: (votes[k], -order.index(k))))
        else:
            owner.append(None)
    return owner


def paint_units(out, glab, units, owners, color, loose, W, H, pad=PAINT_PAD):
    """上色：① 组内全部墨迹像素（含彩色点）② pad 邻域内的浅灰抗锯齿笔画。"""
    painted = 0
    for (sl, m, a), ow in zip(units, owners):
        if ow is None:
            continue
        out[sl][m] = color[ow]
        painted += a
        if pad:
            gid = int(glab[sl][m][0])
            y0 = max(0, sl[0].start - pad); y1 = min(H, sl[0].stop + pad)
            x0 = max(0, sl[1].start - pad); x1 = min(W, sl[1].stop + pad)
            near = ndimage.binary_dilation(
                glab[y0:y1, x0:x1] == gid, structure=np.ones((2 * pad + 1,) * 2, bool))
            out[y0:y1, x0:x1][near & loose[y0:y1, x0:x1]] = color[ow]
    return painted



def measure_label_boxes(arr, labels, thr=235):
    """以视觉给定的标签位置为锚，量该字母**含抗锯齿**的墨迹 bbox。

    验收标准（见 SKILL.md「标签框验收标准」）：
      框线必须落在墨迹外的空白上（bbox 外扩 1px 画线），框内只能有该字母。
    """
    ink = (arr < thr).any(axis=2)
    H, W = ink.shape
    out = {}
    ncomp = {}
    for n, (x, y) in labels.items():
        x0, y0 = max(0, int(x) - 4), max(0, int(y) - 4)
        x1, y1 = min(W, int(x) + 28), min(H, int(y) + 28)
        sub = ink[y0:y1, x0:x1]
        lab, k = ndimage.label(sub, structure=np.ones((3, 3), bool))
        if k == 0:
            out[n] = [int(x), int(y), int(x) + 10, int(y) + 12]
            continue
        best = None
        for i, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None:
                continue
            a = int((lab[sl] == i + 1).sum())
            if a < 25:
                continue
            bx0, bx1 = sl[1].start + x0, sl[1].stop - 1 + x0
            by0, by1 = sl[0].start + y0, sl[0].stop - 1 + y0
            d = abs(bx0 - x) + abs(by0 - y)
            if best is None or d < best[0]:
                best = (d, bx0, by0, bx1, by1)
        if best is None:
            out[n] = [int(x), int(y), int(x) + 10, int(y) + 12]
            ncomp[n] = 1
            continue
        _, gx0, gy0, gx1, gy1 = best
        # 合并同组字形：括号组 (A) = ( + A + ) 三个连通域，只框一个括号是错的
        ah = gy1 - gy0 + 1
        comps = []
        for i, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None:
                continue
            a = int((lab[sl] == i + 1).sum())
            if a < 12:
                continue
            bx0, bx1 = sl[1].start + x0, sl[1].stop - 1 + x0
            by0, by1 = sl[0].start + y0, sl[0].stop - 1 + y0
            comps.append((bx0, by0, bx1, by1, a))
        changed = True
        while changed:
            changed = False
            cy = (gy0 + gy1) / 2.0
            for (bx0, by0, bx1, by1, a) in comps:
                if bx0 >= gx0 and bx1 <= gx1 and by0 >= gy0 and by1 <= gy1:
                    continue
                h = by1 - by0 + 1
                if not (0.6 * ah <= h <= 1.5 * ah):
                    continue
                if abs((by0 + by1) / 2.0 - cy) > 5:
                    continue
                gap = max(bx0 - gx1, gx0 - bx1) - 1
                if gap > 6:
                    continue
                gx0, gy0 = min(gx0, bx0), min(gy0, by0)
                gx1, gy1 = max(gx1, bx1), max(gy1, by1)
                changed = True
        out[n] = [gx0, gy0, gx1, gy1]
        ncomp[n] = sum(1 for (bx0, by0, bx1, by1, a) in comps
                       if bx0 >= gx0 and bx1 <= gx1 and by0 >= gy0 and by1 <= gy1)
        ncomp[n] = max(1, ncomp[n])
    return out, ncomp


def check_label_boxes(arr, boxes, expected=None, thr=235):
    """返回 [(name, 框线压墨迹px, 框内完整连通块数, 跨框块数)]。"""
    ink = (arr < thr).any(axis=2)
    lab, _ = ndimage.label(ink, structure=np.ones((3, 3), bool))
    H, W = ink.shape
    res = []
    for n, (x0, y0, x1, y1) in boxes.items():
        ring = np.zeros((H, W), bool)
        ring[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = True
        ring[y0:y1 + 1, x0:x1 + 1] = False
        on_line = int((ink & ring).sum())
        ids = set(np.unique(lab[y0:y1 + 1, x0:x1 + 1])) - {0}
        full = 0
        for i in ids:
            ys, xs = np.where(lab == i)
            if ys.min() >= y0 and ys.max() <= y1 and xs.min() >= x0 and xs.max() <= x1:
                full += 1
        res.append((n, on_line, full, len(ids) - full, (expected or {}).get(n, 1)))
    return res


def main():
    ap = argparse.ArgumentParser(description="多 panel 组图拆分（视觉给区间 + 按组多数票归属 + 全墨迹上色）")
    ap.add_argument("img")
    ap.add_argument("labels")
    ap.add_argument("--out", default="panel_out")
    ap.add_argument("--panels", required=True, help="看图为每个 panel 给的区间（可多矩形，顺序即优先级）")
    ap.add_argument("--check", help="逐部件覆盖率自查 json")
    ap.add_argument("--min-area", type=int, default=1, help="组最小面积，默认 1（不丢小点）")
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.img).convert("RGB")
    orig, strict, loose = load(args.img)
    H, W = strict.shape
    labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
    panels = json.loads(pathlib.Path(args.panels).read_text(encoding="utf-8"))
    order = list(panels.keys())
    boxes = {n: as_boxes(panels[n]) for n in order}

    report = []
    def say(s=""):
        print(s); report.append(s)

    say("== panel_split ==")
    say("图 %dx%d ；墨迹像素 %d ；panel %d（JSON 顺序即优先级）"
        % (W, H, int(strict.sum()), len(order)))

    # 01 标签框预览：框 = 字母墨迹 bbox + 外扩 1px（SKILL.md「标签框验收标准」）
    lboxes, lncomp = measure_label_boxes(orig, labels)
    (outdir / "label_boxes.json").write_text(json.dumps(lboxes, ensure_ascii=False, indent=1), encoding="utf-8")
    pv = img.copy(); d = ImageDraw.Draw(pv)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for n in labels:
        b = lboxes[n]
        d.rectangle([b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], outline=(220, 30, 30), width=1)
        d.text((b[0], b[3] + 3), "%s" % n.lower(), fill=(220, 30, 30), font=font)
    pv.save(outdir / "01_标签框.png")
    say("→ 01_标签框.png（框 = 字母墨迹 bbox 外扩 1px）")
    say("")
    say("标签框验收（① 框线压墨迹=0 ② 框内完整块=该组字形数 ③ 跨框块=0）:")
    lbad = 0
    for n, on_line, full, stray, exp in check_label_boxes(orig, lboxes, lncomp):
        ok = (on_line == 0 and full == exp and stray == 0)
        if not ok: lbad += 1
        say("   (%s) 框 %dx%d 框线压墨迹 %d px 框内完整块 %d/%d 跨框块 %d %s"
            % (n.lower(), lboxes[n][2] - lboxes[n][0] + 1, lboxes[n][3] - lboxes[n][1] + 1,
               on_line, full, exp, stray, "OK" if ok else "!!"))
    say("   标签框不合格数: %d %s" % (lbad, "OK" if lbad == 0 else "!!"))

    say("")
    say("区间（视觉给定）:")
    for n in order:
        for bi, b in enumerate(boxes[n]):
            say("   (%s)%s x%d-%d y%d-%d" % (n.lower(), "" if len(boxes[n]) == 1 else "#%d" % (bi + 1), *b))

    # ── 合并成单元 + 多数票归属 ──
    glab, units = group_units(strict, GROUP_KERNEL, args.min_area)
    owners = assign_units(units, boxes, order, W, H)
    say("")
    say("分组：%d 个单元（核 %dx%d，无最小面积过滤）；未归属 %d 个"
        % (len(units), GROUP_KERNEL[0], GROUP_KERNEL[1], sum(1 for o in owners if o is None)))

    pal = safe_palette(orig, len(order)); color = dict(zip(order, pal))
    out = orig.copy()
    painted = paint_units(out, glab, units, owners, color, loose, W, H)

    # 每 panel 的墨迹掩码 + bbox
    mask_by = {}
    bbox = {}
    for n in order:
        m = np.zeros((H, W), bool)
        for (sl, mm, a), ow in zip(units, owners):
            if ow == n:
                m[sl] |= mm
        mask_by[n] = m
        ys, xs = np.nonzero(m)
        bbox[n] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else [0, 0, 0, 0]

    counts = {n: int(np.count_nonzero(mask_by[n])) for n in order}
    say("色块（墨迹像素数）: " + "  ".join("(%s)%d" % (n.lower(), counts[n]) for n in order)
        + "  合计 %d / 全图墨迹 %d" % (sum(counts.values()), int(strict.sum())))
    empty = [n for n in order if counts[n] == 0]
    if empty:
        say("!! 这些 panel 没分到任何墨迹 → 区间给错: %s" % " ".join(n.lower() for n in empty))

    Image.fromarray(out).save(outdir / "02_色块归属.png")
    say("→ 02_色块归属.png（人眼确认：词不裂、彩色点已上色）")

    # ── 全覆盖自查：每个 panel 区间内的墨迹是否都被自己染上 ──
    say("")
    say("① 无主墨迹自查（全图墨迹必须 100% 归到某个 panel）:")
    any_mask = np.zeros((H, W), bool)
    for n in order:
        any_mask |= mask_by[n]
    unassigned = strict & ~any_mask
    total_miss = int(unassigned.sum())
    say("   全图墨迹 %d ；已归属 %d ；**无主 %d** %s"
        % (int(strict.sum()), int(any_mask.sum()), total_miss, "OK" if total_miss == 0 else "!!"))
    if total_miss:
        ys, xs = np.nonzero(unassigned)
        say("     无主墨迹 bbox: x %d-%d y %d-%d" % (xs.min(), xs.max(), ys.min(), ys.max()))

    say("② 越界自查（归属像素落在本 panel 所有区间之外 = 跨界词，信息性）:")
    cross_total = 0
    for n in order:
        own = mask_by[n]
        inside = np.zeros((H, W), bool)
        for (x0, y0, x1, y1) in boxes[n]:
            inside[y0:y1, x0:x1] = True
        c = int((own & ~inside).sum())
        cross_total += c
        say("   (%s) 越界像素 %d %s" % (n.lower(), c, "OK" if c == 0 else "（跨界词，按多数票归本 panel）"))
    say("   越界合计 %d（不阻断；仅提示区间可再收紧）" % cross_total)

    # ── 逐部件覆盖（可选）──
    low = 0
    if args.check:
        checks = json.loads(pathlib.Path(args.check).read_text(encoding="utf-8"))
        say("")
        say("逐部件覆盖率（须 >97%）:")
        for n, parts in checks.items():
            if n not in mask_by:
                continue
            for tag, (x0, y0, x1, y1) in parts.items():
                tot = int(strict[y0:y1, x0:x1].sum())
                if tot == 0:
                    say("   (%s) %-14s 框内无墨迹（采样框要校正）" % (n.lower(), tag)); continue
                got = int((strict[y0:y1, x0:x1] & mask_by[n][y0:y1, x0:x1]).sum())
                pct = 100.0 * got / tot
                if pct <= 97: low += 1
                say("   (%s) %-14s %5.1f%% %s" % (n.lower(), tag, pct, "OK" if pct > 97 else "!! 漏"))
        say("   不达标部件数: %d %s" % (low, "OK" if low == 0 else "!!"))

    # ── 实心化 + 互斥划分 ──
    used_r = {}
    for n, m in mask_by.items():
        cur = m
        for R in (4, 8, 12, 16, 20, 26, 32, 40):
            c = ndimage.binary_closing(cur, structure=np.ones((2 * R + 1, 2 * R + 1), bool))
            cur, used_r[n] = c, R
            if ndimage.label(c, structure=np.ones((3, 3), bool))[1] == 1:
                break
    bmin = np.full((H, W), 1e18); best = np.full((H, W), -1, dtype=np.int16)
    for k, n in enumerate(order):
        dd = ndimage.distance_transform_edt(~mask_by[n])
        m2 = dd < bmin
        bmin[m2] = dd[m2]; best[m2] = k

    solid = orig.copy()
    for k, n in enumerate(order):
        solid[best == k] = color[n]
    Image.fromarray(solid).save(outdir / "03_实心区域.png")
    Image.blend(img, Image.fromarray(solid), 0.42).save(outdir / "04_区域叠原图.png")

    say("")
    say("实心区域 + 面积守恒终检:")
    total = 0; bad_conn = 0
    for k, n in enumerate(order):
        m3 = (best == k); a = int(m3.sum()); total += a
        ncc = ndimage.label(m3, structure=np.ones((3, 3), bool))[1]
        # 墨迹本身就分成多块时（如图例/标签远离主体），区域允许跟着分块
        ink_ncc = ndimage.label(mask_by[n], structure=np.ones((3, 3), bool))[1]
        allow = max(1, ink_ncc)
        if ncc > allow:
            bad_conn += 1
        say("   (%s) 闭运算R=%-2d 区域块=%d（墨迹块=%d）面积=%d %s"
            % (n.lower(), used_r[n], ncc, ink_ncc, a, "OK" if ncc <= allow else "!! 碎裂"))
    say("   面积合计 %d ; 全图 %d ; 差 %d %s"
        % (total, H * W, H * W - total, "OK 完全铺满" if total == H * W else "!! 漏格/叠格"))

    if not args.no_crop:
        say("")
        say("子图（本 panel 墨迹 bbox +2px，只保留同 panel 像素）:")
        for k, n in enumerate(order):
            if counts[n] == 0: continue
            m4 = (best == k); b = bbox[n]; M = 2
            x0 = max(0, b[0] - M); y0 = max(0, b[1] - M)
            x1 = min(W, b[2] + 1 + M); y1 = min(H, b[3] + 1 + M)
            tile = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
            mm = m4[y0:y1, x0:x1]
            tile[mm] = orig[y0:y1, x0:x1][mm]
            Image.fromarray(tile).save(outdir / ("panel_%s.png" % n))
            say("   (%s) panel_%s.png %dx%d bbox=(%d,%d,%d,%d)"
                % (n.lower(), n, x1 - x0, y1 - y0, x0, y0, x1, y1))

    (outdir / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n报告: %s" % (outdir / "report.txt"))
    ok = (total_miss == 0) and (total == H * W) and (bad_conn == 0) and not empty and (low == 0) and (lbad == 0)
    print("全部检查通过" if ok else "!! 有检查项未通过，见 report.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())