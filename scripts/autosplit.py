#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""autosplit.py — multi-panel 文献组图一条龙拆分 v2
用法:
  python autosplit.py <image> <outdir>            # 第1步: 标签候选 → label_candidates.png
  python autosplit.py <image> <outdir> --labels <labels.json>   # 第2步: 自动分块+bbox+间隙+画线+crop+验证

labels.json 格式: {"A": [x, y], "B": [x, y], ...}   (标签字母左上角)
init.json 可选: {"A": [x1,y1,x2,y2], ...}  (面板初始区域，跳过自动推断)
输出: 分割线预览.png / panel/panel_X.png / contact_sheet.png / split_report.txt
"""
import sys, os, io, json, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def load(src):
    im = Image.open(src).convert('RGB')
    arr = np.array(im).astype(int)
    W, H = arr.shape[1], arr.shape[0]
    # 2026-08-11 教训：必须用彩色掩码（任一 RGB<225），不能用近白判断——
    # 浅色内容（粉色细胞/浅蓝染色/浅色柱状图）会被 bg_white/gray 误判为背景，
    # 导致分割线穿过彩色内容。彩色掩码比近白掩码多检出 ~15% 内容像素。
    content = (arr[:,:,0] < 225) | (arr[:,:,1] < 225) | (arr[:,:,2] < 225)
    return im, arr, W, H, content

def find_label_candidates(dark, min_h=8, max_h=60, min_w=6, max_w=40, min_area=30, max_area=3000):
    struct = np.ones((3,3), dtype=bool)
    labeled, n = ndimage.label(dark, structure=struct)
    sizes = ndimage.sum(dark, labeled, range(1, n+1))
    cands = []
    for i in range(1, n+1):
        area = sizes[i-1]
        ys, xs = np.where(labeled == i)
        if len(ys) == 0: continue
        h = ys.max()-ys.min()+1; w = xs.max()-xs.min()+1
        if min_h <= h <= max_h and min_w <= w <= max_w and min_area <= area <= max_area:
            region = dark[max(0,ys.min()-2):ys.max()+3, max(0,xs.min()-2):xs.max()+3]
            iso = region.sum()/area
            cands.append((int(xs.min()), int(ys.min()), int(w), int(h), int(area), round(iso,2)))
    cands.sort(key=lambda c: (c[1], c[0]))
    dedup = []
    for c in cands:
        if dedup and abs(c[0]-dedup[-1][0])<25 and abs(c[1]-dedup[-1][1])<25:
            if c[4] > dedup[-1][4]: dedup[-1] = c
        else:
            dedup.append(c)
    return dedup

def save_candidate_grid(im, cands, outpath, cols=6):
    cell_w, cell_h, pad = 90, 100, 8
    rows = (len(cands)+cols-1)//cols
    grid = Image.new('RGB', (cols*cell_w, rows*cell_h), (255,255,255))
    d = ImageDraw.Draw(grid)
    for idx, (x, y, w, h, area, iso) in enumerate(cands):
        r, c = divmod(idx, cols)
        crop = im.crop((max(0,x-15), max(0,y-20), x+40, y+45))
        crop = crop.resize((cell_w-2*pad, cell_h-2*pad), Image.LANCZOS)
        grid.paste(crop, (c*cell_w+pad, r*cell_h+pad))
        d.text((c*cell_w+2, r*cell_h+2), str(idx), fill=(200,0,0))
    grid.save(outpath)

def detect_extent(content, x1, y1, x2, y2, thr=0.003):
    seg = content[y1:y2, x1:x2]
    colc = seg.mean(axis=0); rowc = seg.mean(axis=1)
    cx = np.where(colc > thr)[0]; cy = np.where(rowc > thr)[0]
    if len(cx)==0 or len(cy)==0: return None
    return (x1+int(cx.min()), y1+int(cy.min()), x1+int(cx.max()), y1+int(cy.max()))

def auto_init_from_labels(labels, W, H, content):
    """按标签 y 分行带 → 行带间取全宽行间隙(宽度>=15) → 行带内按标签 x + 列间隙分列"""
    labs = sorted(labels.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    rows = []
    for name, (x, y) in labs:
        if rows and abs(y - rows[-1][0][1][1]) < 40:
            rows[-1].append((name, (x, y)))
        else:
            rows.append([(name, (x, y))])
    # 预先计算所有 >=15px 的全宽行间隙段
    gaps = []
    inb = False; s = 0
    for gy in range(H):
        blank = content[max(0,gy-2):gy+3].mean() < 0.03
        if blank and not inb: inb = True; s = gy
        elif not blank and inb:
            if gy - s >= 15: gaps.append((s, gy-1))
            inb = False
    if inb and H - s >= 15: gaps.append((s, H-1))
    row_y = []
    for i, row in enumerate(rows):
        ys = [p[1] for _, p in row]
        top = max(0, min(ys) - 20)
        bot = H
        if i+1 < len(rows):
            nxt = min(p[1] for _, p in rows[i+1])
            best = None
            for (gs, ge) in gaps:
                if gs >= max(ys)+5 and ge <= nxt-5:
                    if best is None or (gs+ge) < (best[0]+best[1]): best = (gs, ge)
            if best:
                bot = (best[0]+best[1])//2
            else:
                bot = (max(ys) + nxt)//2
        else:
            # 最后一行：取最后一个大间隙为底
            for (gs, ge) in gaps:
                if gs >= max(ys)+5 and bot == H:
                    bot = (gs+ge)//2
        row_y.append((top, bot))
    init = {}
    for row, (yt, yb) in zip(rows, row_y):
        # 该行带内列间隙段（>=15px）
        seg = content[max(0,yt):min(H,yb)]
        colc = seg.mean(axis=0)
        col_gaps = []
        inb = False; s = 0
        for x in range(W+1):
            b = (x < W and colc[x] < 0.03)
            if b and not inb: inb = True; s = x
            elif not b and inb:
                if x - s >= 15: col_gaps.append((s, x-1))
                inb = False
        if inb and W - s >= 15: col_gaps.append((s, W-1))
        row_sorted = sorted(row, key=lambda t: t[1][0])
        def gap_between(a, b):
            for (gs, ge) in col_gaps:
                if gs > a and ge < b:
                    return (gs+ge)//2
            return (a+b)//2
        for j, (name, (x, y)) in enumerate(row_sorted):
            xl = 0 if j == 0 else gap_between(row_sorted[j-1][1][0], x)
            xr = W if j == len(row_sorted)-1 else gap_between(x, row_sorted[j+1][1][0])
            init[name] = (xl, yt, xr, yb)
    return init

def draw_lines(im, arr, content, vlines, hlines, pad=2):
    content_dil = ndimage.binary_dilation(content, iterations=pad)
    safe = ~content_dil
    draw = ImageDraw.Draw(im)
    W, H = arr.shape[1], arr.shape[0]
    LINE = (255, 0, 0); w = 2
    for x, y0, y1 in vlines:
        for y in range(y0, y1):
            for dx in range(w):
                xx = x+dx
                if xx < W and not content[y, xx] and safe[y, xx]:
                    draw.point((xx, y), fill=LINE)
    for y, x0, x1 in hlines:
        for x in range(x0, x1):
            for dy in range(w):
                yy = y+dy
                if yy < H and not content[yy, x] and safe[yy, x]:
                    draw.point((x, yy), fill=LINE)

def verify(panels, content, report):
    ok = True
    for lab in sorted(panels):
        x1, y1, x2, y2 = panels[lab]
        seg = content[y1:y2, x1:x2]
        colc = seg.mean(axis=0); rowc = seg.mean(axis=1)
        cx = np.where(colc > 0.01)[0]; cy = np.where(rowc > 0.01)[0]
        if len(cx)==0 or len(cy)==0:
            report.write(f"  {lab}: 空内容!\n"); ok = False; continue
        cxl, cxr = x1+int(cx.min()), x1+int(cx.max()); cyt, cyb = y1+int(cy.min()), y1+int(cy.max())
        ml, mr, mt, mb = cxl-x1, x2-1-cxr, cyt-y1, y2-1-cyb
        flags = []
        if ml < 2: flags.append(f"L{ml}")
        if mr < 2: flags.append(f"R{mr}")
        if mt < 2: flags.append(f"T{mt}")
        if mb < 2: flags.append(f"B{mb}")
        if flags: ok = False
        report.write(f"  {lab}: 余量L{ml}R{mr}T{mt}B{mb} {'!' if flags else 'OK'}\n")
    return ok

def main():
    t0 = time.time()
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    im, arr, W, H, content = load(src)
    dark = (arr[:,:,0]<130) & (arr[:,:,1]<130) & (arr[:,:,2]<130)

    if '--labels' in sys.argv:
        lp = sys.argv[sys.argv.index('--labels')+1]
        with io.open(lp, encoding='utf-8') as f:
            labels = json.load(f)
        # 初始区域：手工 init.json 优先，否则自动推断
        ip = os.path.join(outdir, 'init.json')
        if os.path.exists(ip):
            with io.open(ip, encoding='utf-8') as f:
                init = {k: tuple(v) for k, v in json.load(f).items()}
        else:
            init = auto_init_from_labels(labels, W, H, content)
            with io.open(ip, 'w', encoding='utf-8') as f:
                json.dump({k: list(v) for k, v in init.items()}, f, ensure_ascii=False, indent=1)
            print(f"[auto] 自动分块 init.json 已生成（可手工改后重跑）")
        # bbox
        ext = {}
        for lab, b in init.items():
            e = detect_extent(content, *b)
            if e: ext[lab] = e
        # 相邻间隙中心 = 分割线（水平/竖直分别处理）
        vlines, hlines = [], []
        labs = sorted(ext.keys())
        # 水平线：相邻面板 y 重叠且 x 相接（且中间无其他面板阻挡）
        labs_l = list(ext.keys())
        for a in labs_l:
            for b in labs_l:
                if a >= b: continue
                ea, eb = ext[a], ext[b]
                xov = min(ea[2], eb[2]) - max(ea[0], eb[0])
                if xov <= 0: continue
                # a 在 b 上方 或 b 在 a 上方
                upper, lower = (ea, eb) if ea[3] < eb[1] else (eb, ea)
                gap_y = (upper[3], lower[1])
                if lower[1] - upper[3] <= 0: continue
                # 中间是否被其他面板阻挡
                blocked = False
                for c in labs_l:
                    if c in (a, b): continue
                    ec = ext[c]
                    if ec[3] <= upper[3] or ec[1] >= lower[1]: continue  # c 完全在间隙外
                    cxov = min(ec[2], min(ea[2], eb[2])) - max(ec[0], max(ea[0], eb[0]))
                    if cxov > 0: blocked = True; break
                if blocked: continue
                mid_y = (upper[3]+lower[1])//2
                seg_x = (max(ea[0], eb[0]), min(ea[2], eb[2]))
                hlines.append((mid_y, seg_x[0], seg_x[1]))
        # 垂直线：左右相邻面板，y 重叠，且中间无其他面板阻挡
        for a in labs_l:
            for b in labs_l:
                if a >= b: continue
                ea, eb = ext[a], ext[b]
                yov = min(ea[3], eb[3]) - max(ea[1], eb[1])
                if yov <= 0: continue
                left, right = (ea, eb) if ea[2] < eb[0] else (eb, ea)
                if right[0] - left[2] <= 0: continue
                blocked = False
                for c in labs_l:
                    if c in (a, b): continue
                    ec = ext[c]
                    if ec[2] <= left[2] or ec[0] >= right[0]: continue  # c 完全在间隙外
                    cyov = min(ec[3], min(ea[3], eb[3])) - max(ec[1], max(ea[1], eb[1]))
                    if cyov > 0: blocked = True; break
                if blocked: continue
                mid_x = (left[2]+right[0])//2
                seg_y = (max(ea[1], eb[1]), min(ea[3], eb[3]))
                vlines.append((mid_x, seg_y[0], seg_y[1]))
        # 合并重复线
        def merge_lines(lines, axis):
            out = []
            for l in sorted(lines):
                if axis == 'v':
                    if out and abs(out[-1][0]-l[0]) <= 3 and out[-1][2] >= l[1]-1:
                        out[-1] = (out[-1][0], min(out[-1][1], l[1]), max(out[-1][2], l[2]))
                    else: out.append(l)
                else:
                    if out and abs(out[-1][0]-l[0]) <= 3 and out[-1][2] >= l[1]-1:
                        out[-1] = (out[-1][0], min(out[-1][1], l[1]), max(out[-1][2], l[2]))
                    else: out.append(l)
            return out
        vlines = merge_lines(vlines, 'v'); hlines = merge_lines(hlines, 'h')
        # 面板 = 分割线围成的区域（用 ext 外扩到分割线）
        panels = {}
        for lab, e in ext.items():
            cx, cy = (e[0]+e[2])//2, (e[1]+e[3])//2
            x1 = max(0, e[0]-15); x2 = min(W, e[2]+15)
            y1 = max(0, e[1]-15); y2 = min(H, e[3]+15)
            for vx, vy0, vy1 in vlines:
                if vy0 <= cy <= vy1:
                    if e[0]-60 <= vx <= e[0] and vx < cx: x1 = vx
                    if e[2] <= vx <= e[2]+60 and vx > cx: x2 = vx
            for hy, hx0, hx1 in hlines:
                if hx0 <= cx <= hx1:
                    if e[1]-60 <= hy <= e[1] and hy < cy: y1 = hy
                    if e[3] <= hy <= e[3]+60 and hy > cy: y2 = hy
            panels[lab] = (x1, y1, x2, y2)
        # 画线 + crop
        draw_lines(im, arr, content, vlines, hlines)
        im.save(os.path.join(outdir, '分割线预览.png'))
        im2 = Image.open(src).convert('RGB')
        pdir = os.path.join(outdir, 'panel'); os.makedirs(pdir, exist_ok=True)
        for lab, b in panels.items():
            im2.crop(b).save(os.path.join(pdir, f'panel_{lab}.png'))
        # contact sheet
        labels_sorted = sorted(panels.keys())
        n = len(labels_sorted); grid = min(3, max(1, n)); rows = (n+grid-1)//grid
        cell_w = 320
        sheet = Image.new('RGB', (grid*cell_w, rows*(cell_w+20)), (245,245,245))
        d2 = ImageDraw.Draw(sheet)
        for i, lab in enumerate(labels_sorted):
            r, c = divmod(i, grid)
            imc = Image.open(os.path.join(pdir, f'panel_{lab}.png'))
            hh = cell_w; ww = int(imc.width*hh/imc.height)
            if ww > cell_w: ww = cell_w; hh = int(imc.height*ww/imc.width)
            imc = imc.resize((ww,hh), Image.LANCZOS)
            x = c*cell_w + (cell_w-ww)//2; y = r*(cell_w+20) + 10
            sheet.paste(imc, (x, y))
            d2.text((c*cell_w+5, r*(cell_w+20)+2), lab, fill=(200,0,0))
        sheet.save(os.path.join(outdir, 'contact_sheet.png'))
        # report
        with io.open(os.path.join(outdir, 'split_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"面板: {labels_sorted}\n")
            for lab in labels_sorted: f.write(f"  {lab}: {panels[lab]}\n")
            f.write(f"竖线: {vlines}\n横线: {hlines}\n")
            ok = verify(panels, content, f)
            f.write(f"\n验证: {'ALL OK' if ok else 'NEED FIX'}\n耗时: {time.time()-t0:.1f}s\n")
        print(f"[done] 分割线预览/crop/contact_sheet 已生成，耗时 {time.time()-t0:.1f}s")
        print(f"       验证: {'ALL OK' if ok else 'NEED FIX — 看 split_report.txt'}")
    else:
        cands = find_label_candidates(dark)
        cg = os.path.join(outdir, 'label_candidates.png')
        save_candidate_grid(im, cands, cg)
        with io.open(os.path.join(outdir, 'label_candidates.json'), 'w', encoding='utf-8') as f:
            json.dump([{'x':c[0],'y':c[1],'w':c[2],'h':c[3],'area':c[4],'iso':c[5]} for c in cands], f, ensure_ascii=False, indent=1)
        print(f"[1] 标签候选 {len(cands)} 个 → {cg} (编号与 label_candidates.json 对应)")
        print(f"    下一步: 视觉一次认完字母 → 写 {outdir}/labels.json 如 {{\"A\":[45,21],...}}")
        print(f"    然后: python autosplit.py <img> <outdir> --labels <outdir>/labels.json")

if __name__ == '__main__':
    main()
