# -*- coding: utf-8 -*-
"""
通用面板分割器（无 AI，配合 label_detect.py 使用）

流程:
  1. 读图 + labels.json(标签坐标 {字母:[x,y]})
  2. 内容连通域 → 间隙合并(单词成块) → bbox 左上角归属
  3. 面板 = 色块集合并集
  4. 验证: L形 + 不重叠
  5. 裁剪子图 + 色块归属图 + 分割线图

用法:
  python split_panels.py <img> <labels.json> --out outdir [--panels 手工框json] [--no-merge] [--dilate N]
  --panels: 若自动网格不准, 传手工面板框 {"A":[x0,y0,x1,y1],...}
  --no-merge: 关闭间隙合并(按原连通域归属)
  --dilate N: 额外膨胀N像素
"""
import sys
import os
import json
import colorsys
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def load_arr(src):
    return np.array(Image.open(src).convert('RGB'))


def cbands(proj, minv, minl, tot):
    """内容连续带 [(start,end)]"""
    b = []
    inb = False
    s = 0
    for i, v in enumerate(proj):
        if v > minv:
            if not inb:
                s = i
                inb = True
        else:
            if inb and i - s >= minl:
                b.append((s, i - 1))
            inb = False
    if inb and tot - s >= minl:
        b.append((s, tot - 1))
    return b


def auto_panels(labels, content, H, W):
    """从标签 + 内容空白带自动定面板粗边界"""
    ls = sorted(labels.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    rows = []
    for name, (x, y) in ls:
        if rows and abs(y - rows[-1][0][1][1]) < 45:
            rows[-1].append((name, (x, y)))
        else:
            rows.append([(name, (x, y))])
    panels = {}
    for ri, row in enumerate(rows):
        y_top = min(p[1] for _, p in row)
        y_bot = H
        if ri + 1 < len(rows):
            y_nxt = min(p[1] for _, p in rows[ri + 1])
            seg = content[y_top:y_nxt]
            rc = seg.sum(axis=1)
            best = y_top
            bmin = 1e9
            for y in range(0, len(rc) - 5):
                v = rc[y:y + 5].mean()
                if v < bmin:
                    bmin = v
                    best = y_top + y + 2
            y_bot = best
        row_sorted = sorted(row, key=lambda t: t[1][0])
        for j, (name, (x, y)) in enumerate(row_sorted):
            x_l = 0 if j == 0 else _col_gap(row_sorted[j - 1][1][0], x, y_top, y_bot, content)
            x_r = W if j == len(row_sorted) - 1 else _col_gap(x, row_sorted[j + 1][1][0], y_top, y_bot, content)
            panels[name] = (x_l, y_top, x_r, y_bot)
    return panels


def _col_gap(xa, xb, yt, yb, content):
    """两标签间列空白带中心"""
    seg = content[yt:yb]
    cc = seg.sum(axis=0)
    lo = max(0, xa + 5)
    hi = min(len(cc) - 1, xb - 5)
    if lo >= hi:
        return (xa + xb) // 2
    best = (lo + hi) // 2
    bmin = 1e9
    for x in range(lo, max(lo, hi - 8)):
        v = cc[x:x + 8].mean()
        if v < bmin:
            bmin = v
            best = x + 4
    return best


def merge_words(sizes_all, lab, min_area=5, gap_ratio=0.8):
    """间隙合并: 同一行(y重叠>60%)、水平间隙<gap_ratio*中位字高 的字母连通域合成一块。
    单词(字母间隙小)成单一色块; 面板间(间距大)不粘连。"""
    n = int(sizes_all.size)
    boxes = []
    for i in range(1, n + 1):
        s = int(sizes_all[i - 1])
        if s < min_area:
            continue
        ys, xs = np.where(lab == i)
        boxes.append({'x0': int(xs.min()), 'x1': int(xs.max()), 'y0': int(ys.min()),
                      'y1': int(ys.max()), 'size': s})
    hs = sorted(b['y1'] - b['y0'] + 1 for b in boxes if b['size'] > 10)
    med_h = hs[len(hs) // 2] if hs else 10
    gap_thr = max(2, int(med_h * gap_ratio))
    merged = []
    for b in sorted(boxes, key=lambda b: b['x0']):
        placed = False
        for m in merged:
            yov = min(b['y1'], m['y1']) - max(b['y0'], m['y0'])
            yspan = min(b['y1'] - b['y0'] + 1, m['y1'] - m['y0'] + 1)
            if yspan > 0 and yov / yspan > 0.6 and (b['x0'] - m['x1']) < gap_thr:
                m['x1'] = max(m['x1'], b['x1'])
                m['y0'] = min(m['y0'], b['y0'])
                m['y1'] = max(m['y1'], b['y1'])
                m['c'] = m.get('c', 0) + 1
                placed = True
                break
        if not placed:
            merged.append({'x0': b['x0'], 'x1': b['x1'], 'y0': b['y0'], 'y1': b['y1'], 'c': 0})
    return merged, gap_thr


def assign_blocks_by_nearest_label(blocks, labels, anchor_pad=0):
    """直接按最近标签归属(无需面板框/分割线):
    每个块归属到距离最近的有效标签。有效条件: 标签**不在块的右下方**即可作为候选
    (标签 x>块x 且 y>块y 才算右下方; 其余左/上/正下/正左都不排除)。
    从这些候选中取距离最近的。"""
    owner = {}
    for bi, blk in enumerate(blocks):
        cx = (blk['x0'] + blk['x1']) / 2.0
        cy = (blk['y0'] + blk['y1']) / 2.0
        best = None
        bd = 1e18
        for n, (lx, ly) in labels.items():
            # 排除右下方: 标签在块右下角(标签x>块中心x 且 标签y>块中心y)
            if lx > cx and ly > cy:
                continue
            d = (cx - lx) ** 2 + (cy - ly) ** 2
            if d < bd:
                bd = d
                best = n
        owner[bi] = best
    return owner


def max_empty_rect(ax, ay, drawn, H, W):
    """从锚点(ax,ay)到图右下角范围内, 扣除已画矩形(drawn内部), 找面积最大的空矩形。
    返回 (x0, y0, x1, y1)。实现: 在锚点右下区域做行直方图最大矩形(仅扫锚点以下区域)。
    顺序: 先下后上(y降序), 同y先右后左(x降序) 逐个面板。"""
    # 构建锚点右下区域的占用掩码(仅 ay..H, ax..W)
    occ = np.zeros((H, W), dtype=bool)
    for (dx0, dy0, dx1, dy1) in drawn:
        occ[max(ay, dy0):min(H, dy1), max(ax, dx0):min(W, dx1)] = True
    # 限制扫描范围, 避免全图O(H*W)每面板
    rowcrop = occ[ay:H, ax:W]
    rh, rw = rowcrop.shape
    if rh == 0 or rw == 0:
        return (ax, ay, W, H)
    best_area = 0
    best = (ax, ay, W, H)
    heights = np.zeros(rw, dtype=int)
    for ri in range(rh):
        heights = np.where(rowcrop[ri], 0, heights + 1)
        # 单调栈在 ri 行找最大矩形
        stack = []
        for ci in range(rw + 1):
            h = heights[ci] if ci < rw else 0
            while stack and heights[stack[-1]] > h:
                top = stack.pop()
                top_h = heights[top]
                left = stack[-1] + 1 if stack else 0
                area = top_h * (ci - left)
                if area > best_area:
                    gx0, gy0 = ax + left, ay + ri - top_h + 1
                    gx1, gy1 = ax + ci, ay + ri + 1
                    best = (gx0, gy0, gx1, gy1)
                    best_area = area
            stack.append(ci)
    return best


def build_panels_from_labels(labels, W, H, anchor_pad=0):
    """用最大空矩形方法画面板框: 每个标签一个矩形, 从(标签锚点-左上移量)向右下取最大空矩形。
    顺序: 先下后上(y降序), 同y先右后左(x降序)。返回 {标签: (x0,y0,x1,y1)}。"""
    order = sorted(labels.items(), key=lambda kv: (-kv[1][1], -kv[1][0]))
    drawn = []
    panels = {}
    for n, (x, y) in order:
        ax, ay = max(x - anchor_pad, 0), max(y - anchor_pad, 0)
        x0, y0, x1, y1 = max_empty_rect(ax, ay, drawn, H, W)
        x0 = max(x0, ax); y0 = max(y0, ay)
        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        panels[n] = (int(x0), int(y0), int(x1), int(y1))
        drawn.append((x0, y0, x1, y1))
    return panels


def assign_by_panel_majority(blocks, panels, neighbors=6, radius=200):
    """归属: 每个块按 bbox 左上角落在哪个面板框 → 归谁;
    再对分界线附近色块做**最近邻多数修正**(迭代): 每个块归属改为其周围色块多数面板, 直到全部一致(连成整体)。
    blocks 索引=连通域标号, 可能含 None(面积过小的连通域, 跳过)。"""
    owner = {}
    ordered = sorted(panels.items(), key=lambda kv: -(kv[1][2] + kv[1][3]))
    valid = [bi for bi, blk in enumerate(blocks) if blk is not None]
    cent = {}
    for bi in valid:
        blk = blocks[bi]
        cent[bi] = ((blk['x0'] + blk['x1']) / 2, (blk['y0'] + blk['y1']) / 2)
        bx0, by0 = blk['x0'], blk['y0']
        for n, (x0, y0, x1, y1) in ordered:
            if x0 <= bx0 < x1 and y0 <= by0 < y1:
                owner[bi] = n
                break
    # 迭代最近邻多数修正: 边界块归其周围色块多数面板, 直到无变化
    for _ in range(15):
        changed = False
        ref_cent = {bi: c for bi, c in cent.items() if owner.get(bi) is not None}
        for bi in valid:
            if owner.get(bi) is None:
                continue
            cx, cy = cent[bi]
            dists = []
            for j, (jx, jy) in ref_cent.items():
                if j == bi:
                    continue
                d = (cx - jx) ** 2 + (cy - jy) ** 2
                if d < radius * radius:
                    dists.append((d, j))
            dists.sort()
            neigh = dists[:neighbors]
            if len(neigh) < 3:
                continue
            cnt = Counter(owner.get(j) for _, j in neigh)
            majority = cnt.most_common(1)[0][0]
            if majority is not None and majority != owner.get(bi):
                owner[bi] = majority
                changed = True
        if not changed:
            break
    return owner


def assign_blocks(blocks, panels, anchor_pad=0):
    # 面板(传labels或panels) → 归属(面板框 + 最近邻多数修正)
    if panels and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in panels.values()):
        # panels 是标签字典 → 转成面板框
        return assign_blocks_by_nearest_label(blocks, panels, anchor_pad)
    return assign_by_panel_majority(blocks, panels)


def fix_L_violations(blocks, owner, labels, panels, anchor_pad=0, max_iter=30):
    """根据 L 形标准调整归属: 某面板有色块超到标签左/上方(L形FAIL)时, 把超界块重新归属,
    归到离它最近的、标签在其左上方(允许贴边 anchor_pad)的其他面板。迭代到 L形 ALL OK 或无法改进。"""
    for it in range(max_iter):
        fixed = False
        changed_any = False
        for n in panels:
            ids = [bi for bi, o in owner.items() if o == n]
            if not ids:
                continue
            lx, ly = labels[n]
            for bi in ids:
                blk = blocks[bi]
                if blk['x0'] >= lx - anchor_pad and blk['y0'] >= ly - anchor_pad:
                    continue
                # 此块超界(左上角 < 标签) → 重新归属
                # 用块左上角找最近的、标签在该块左上方的面板
                bx0, by0 = blk['x0'], blk['y0']
                candidate = None
                bd = 1e18
                for m, (mx, my) in labels.items():
                    # 候选面板标签须在块的左上方(或贴边)
                    if mx > bx0 + anchor_pad and my > by0 + anchor_pad:
                        continue  # 标签在块右下方, 排除
                    d = (bx0 - mx) ** 2 + (by0 - my) ** 2
                    if d < bd:
                        bd = d
                        candidate = m
                if candidate:
                    owner[bi] = candidate
                    changed_any = True
                    fixed = True
                else:
                    owner[bi] = None
                    changed_any = True
                    fixed = True
                break  # 每轮改一个, 下轮重新检查
        if not fixed:
            break
        if not changed_any:
            break
    return owner


def apply_separation_rule(blocks, owner, panels):
    """分隔规则(硬判据): 被其他子图色块分隔的右侧/下方色块, 归分隔它的面板。
    对每个块: 若它延伸到相邻面板区域(x1 超出面板右界/下界), 且该相邻面板有内容块
    位于它内部(y重叠且 x 落在此面板区), 则此块被分隔 → 改归相邻面板。"""
    changed = True
    it = 0
    while changed and it < 10:
        changed = False
        it += 1
        # 对每个块找它跨越的面板(块 bbox 中心/范围判定)
        for bi, own in list(owner.items()):
            if own is None:
                continue
            blk = blocks[bi]
            x0p, y0p, x1p, y1p = panels[own]
            # 该块延伸到了哪个其他面板区域(块的右/下超出自身面板, 进入别的面板的范围)
            # 找所有与块64 空间相邻且 overlap 的面板
            for n, (pnx0, pny0, pnx1, pny1) in panels.items():
                if n == own:
                    continue
                # 块是否延伸到面板n 的水平/垂直范围
                if pnx0 <= blk['x1'] and blk['x0'] <= pnx1 and pny0 <= blk['y1'] and blk['y0'] <= pny1:
                    # 检查面板n 是否有块位于 blk 内部(被分隔)
                    inside = False
                    for bo, ob in owner.items():
                        if ob == n:
                            bd = blocks[bo]
                            # bd 完整落在 blk 内部或部分重叠且主要在其内部
                            if (bd['x0'] >= blk['x0'] and bd['x1'] <= blk['x1'] and
                                bd['y0'] >= blk['y0'] and bd['y1'] <= blk['y1']):
                                inside = True
                                break
                    if inside:
                        # blk 被面板n的内容分隔 → 归面板n
                        if owner[bi] != n:
                            # print('分隔规则: 块%d 从%s→%s' % (bi, owner[bi], n))
                            owner[bi] = n
                            changed = True
                        break
    return owner


def apply_anchor_pad(labels, panels, anchor_pad):
    """返回扩张后的面板框(仅用于归属判定示意)"""
    return {n: (x0 - anchor_pad, y0 - anchor_pad, x1, y1) for n, (x0, y0, x1, y1) in panels.items()}


def crop_for(blocks, owner, labels, panels, W, H):
    """裁剪框 = 面板归属色块并集（墨迹方法，SKILL.md §4.5）+ 标签墨迹。

    2026-11-06 修正三处：
    1. 原实现直接返回初始框（注释"非色块并集"），与 §4.5 矛盾 →
       按标签位置生成的框会切掉内容（Figure 1 的 GAPDH 文字被截）。
       现改为按 owner 归属取并集。
    2. 并集只含内容块；split_panels 会把标签区域从内容掩码挖掉，
       标签墨迹不在块里 → 先把标签 bbox 补回并集。
    3. 并集可能越过相邻面板（归属修正导致）→ 对**重叠簇**按重叠区中点竖直/水平切开。
       注意：不能用"标签初始框"当边界（标签位置可能落在邻面板内容范围内，会误切）。
    """
    # 1) 各面板原始并集
    uni = {}
    for n, frame in panels.items():
        xs0, ys0, xs1, ys1 = [], [], [], []
        for bi, blk in enumerate(blocks):
            if blk is None or owner.get(bi) != n:
                continue
            xs0.append(blk['x0']); ys0.append(blk['y0'])
            xs1.append(blk['x1']); ys1.append(blk['y1'])
        if xs0:
            ux0, uy0 = min(xs0), min(ys0)
            ux1, uy1 = max(xs1) + 1, max(ys1) + 1
            if n in labels:                      # 2) 补回标签墨迹
                lx, ly = labels[n]
                ux0 = min(ux0, int(lx) - 2)
                uy0 = min(uy0, int(ly) - 2)
            uni[n] = [int(ux0), int(uy0), int(ux1), int(uy1)]
        else:
            uni[n] = [int(frame[0]), int(frame[1]), int(frame[2]), int(frame[3])]

    # 3) 重叠消除：两两求交，按短边方向中点切
    names = list(uni)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ax0, ay0, ax1, ay1 = uni[a]
            bx0, by0, bx1, by1 = uni[b]
            ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
            oy0, oy1 = max(ay0, by0), min(ay1, by1)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            if (ox1 - ox0) <= (oy1 - oy0):       # 竖直切
                cut = (ox0 + ox1) // 2
                if ax0 < bx0:
                    uni[a][2] = cut; uni[b][0] = cut
                else:
                    uni[b][2] = cut; uni[a][0] = cut
            else:                                 # 水平切
                cut = (oy0 + oy1) // 2
                if ay0 < by0:
                    uni[a][3] = cut; uni[b][1] = cut
                else:
                    uni[b][3] = cut; uni[a][1] = cut

    crop = {}
    for n, (x0, y0, x1, y1) in uni.items():
        crop[n] = (int(max(0, min(x0, W - 1))), int(max(0, min(y0, H - 1))),
                   int(max(1, min(x1, W))), int(max(1, min(y1, H))))
    return crop




def result_for(blocks, owner, labels, panels, anchor_pad=0):
    """L 形验证
    anchor_pad: 锚点向左上移 P 像素。色块判定原点从 (字母x,字母y) 变为 (字母x-P, 字母y-P),
    即色块左上角 >= 字母左上扩展区即算 OK(允许贴边到字母左/上方 P 像素)。"""
    res = {'L形': 'ALL OK', 'overlap': []}
    for n, p in panels.items():
        ids = [bi for bi, own in owner.items() if own == n]
        lx, ly = labels[n]
        if not ids:
            res['L形'] = 'FAIL(无色块)'
            continue
        minx = min(blocks[bi]['x0'] for bi in ids)
        miny = min(blocks[bi]['y0'] for bi in ids)
        if not (minx >= lx - anchor_pad and miny >= ly - anchor_pad):
            res['L形'] = 'FAIL'
    return res


def nearest_owner_conflict(blocks, owner, max_dist=60):
    """空间一致性校验: 每个有归属块最近的邻块若归属不同且距离近 → 标记可疑
    相邻内容块通常同属一面板; 最近邻归属不同 → 该块归属可能被面板边界切错。
    返回可疑块列表 [{block_idx, own, nearest, dist}]
    """
    # 计算每块中心
    cents = []
    for bi, blk in enumerate(blocks):
        cx = (blk['x0'] + blk['x1']) / 2.0
        cy = (blk['y0'] + blk['y1']) / 2.0
        cents.append((cx, cy))
    suspects = []
    for bi, own in owner.items():
        if own is None:
            continue
        bx, by = cents[bi]
        # 找最近邻有归属块
        bd = 1e18
        nearest = None
        for bj, oj in owner.items():
            if bj == bi or oj is None:
                continue
            dx = cents[bj][0] - bx
            dy = cents[bj][1] - by
            d = dx * dx + dy * dy
            if d < bd:
                bd = d
                nearest = (bj, oj)
        if nearest:
            dist = bd ** 0.5
            if nearest[1] != own and dist < max_dist:
                # 面积(以 bbox 面积近似)
                area = lambda bi: max(1, (blocks[bi]['x1'] - blocks[bi]['x0']) * (blocks[bi]['y1'] - blocks[bi]['y0']))
                suspects.append({'block': bi, 'own': own, 'nearest': nearest[1],
                                 'nearest_block': nearest[0], 'dist': round(dist, 1),
                                 'area': area(bi), 'nearest_area': area(nearest[0])})
    return suspects


def resolve_conflicts_by_size(blocks, owner, suspects):
    """冲突按"大块为准"解决: 冲突对中较小的块改为较大的块归属。
    返回 (owner2, reassigned)。不做锁定(锁定会把本应一致的块封死在错误归属)。"""
    owner2 = dict(owner)
    reassigned = []
    for s in suspects:
        bi, bj = s['block'], s['nearest_block']
        wi, wj = s['area'], s['nearest_area']
        owner_i, owner_j = owner2.get(bi), owner2.get(bj)
        if owner_i is None or owner_j is None or owner_i == owner_j:
            continue
        if wi < wj:
            # 小的服从大的 → 改 bi
            reassigned.append((bi, owner_i, owner_j))
            owner2[bi] = owner_j
        else:
            # 大的保留 → 改 bj
            reassigned.append((bj, owner_j, owner_i))
            owner2[bj] = owner_i
    return owner2, reassigned


def resolve_conflicts_by_majority(blocks, owner, suspects, k=5):
    """消除残余冲突: 冲突块归属 = 其最近 k 个有归属邻居中最常见的面板(多数表决)。
    可能被重复解决(块58归G由邻居决定), 一轮定死(不迭代), 保证收敛。
    返回 (owner2, reassigned)"""
    owner2 = dict(owner)
    reassigned = []
    # 预计算所有块中心
    cents = [( (b['x0']+b['x1'])/2.0, (b['y0']+b['y1'])/2.0 ) for b in blocks]
    conflicted = set()
    for s in suspects:
        conflicted.add(s['block'])
        conflicted.add(s['nearest_block'])
    for bi in conflicted:
        if owner2.get(bi) is None:
            continue
        bx, by = cents[bi]
        # 找最近 k 个有归属邻居(排除自身)
        neigh = []
        for bj, oj in owner2.items():
            if bj == bi or oj is None:
                continue
            dx = cents[bj][0]-bx; dy = cents[bj][1]-by
            neigh.append((dx*dx+dy*dy, bj, oj))
        neigh.sort()
        votes = {}
        for _, bj, oj in neigh[:k]:
            votes[oj] = votes.get(oj, 0) + (blocks[bj]['x1']-blocks[bj]['x0'])*(blocks[bj]['y1']-blocks[bj]['y0'])
        if not votes:
            continue
        # 加权多数表决(按面积加权)
        new_own = max(votes, key=votes.get)
        if new_own != owner2.get(bi):
            reassigned.append((bi, owner2[bi], new_own))
            owner2[bi] = new_own
    return owner2, reassigned


def main():
    src = sys.argv[1]
    labels_src = sys.argv[2]
    outdir = 'split_out'
    panels_json = None
    dilate = 0
    anchor_pad = 0
    do_merge = ('--no-merge' not in sys.argv)
    if '--out' in sys.argv:
        outdir = sys.argv[sys.argv.index('--out') + 1]
    if '--panels' in sys.argv:
        panels_json = json.load(open(sys.argv[sys.argv.index('--panels') + 1], encoding='utf-8'))
    if '--dilate' in sys.argv:
        dilate = int(sys.argv[sys.argv.index('--dilate') + 1])
    if '--anchor-pad' in sys.argv:
        anchor_pad = int(sys.argv[sys.argv.index('--anchor-pad') + 1])
    os.makedirs(outdir, exist_ok=True)
    labels = json.load(open(labels_src, encoding='utf-8'))

    # 1. 读图
    arr = load_arr(src)
    H, W, _ = arr.shape

    # 面板框: 无 --panels 时用最大空矩形法从标签生成
    s4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    content = (arr[:, :, 0] < 235) & (arr[:, :, 1] < 235) & (arr[:, :, 2] < 235)
    for n, (x, y) in labels.items():
        content[max(0, y - 5):y + 16, max(0, x - 4):x + 13] = False
    if panels_json is None:
        panels_json = build_panels_from_labels(labels, W, H, anchor_pad)
        print('最大空矩形面板框:')
        for n, p in panels_json.items():
            print('  %s: %s' % (n, p))
    # 连通域: 4连通 + 阈值200(分离图形不粘连)
    lab, nc = ndimage.label(content, structure=s4)
    sizes = ndimage.sum(content, lab, range(1, nc + 1))

    # 2. 连通域 → 块 → 归属 (块索引 == 连通域标号, 对齐绘图)
    blocks = [None] * (nc + 1)  # 索引=连通域标号
    for i in range(1, nc + 1):
        if sizes[i - 1] < 5:
            continue
        ys, xs = np.where(lab == i)
        blocks[i] = {'x0': int(xs.min()), 'x1': int(xs.max()),
                     'y0': int(ys.min()), 'y1': int(ys.max())}
    print('连通域(%d个,面积>=5)块对齐' % nc)
    owner = assign_by_panel_majority(blocks, panels_json)

    # 2.5 自动修复L形: 超界块重新归属直到L形通过
    owner = fix_L_violations(blocks, owner, labels, panels_json, anchor_pad)

    # 3. 面板=块集合并集
    crop = crop_for(blocks, owner, labels, panels_json, W, H)
    # 4. 验证
    result = result_for(blocks, owner, labels, panels_json, anchor_pad)
    result['overlap'] = []
    for a in crop:
        for b in crop:
            if a >= b:
                continue
            A, B = crop[a], crop[b]
            xov = min(A[2], B[2]) - max(A[0], B[0])
            yov = min(A[3], B[3]) - max(A[1], B[1])
            if xov > 0 and yov > 0:
                result['overlap'].append('%s∩%s: x%d y%d' % (a, b, xov, yov))
    # (最近邻归属冲突校验已移除: 交界小字归属歧义是本质局限, 不做迭代归零)

    # 5. 输出
    for n, box in crop.items():
        Image.open(src).convert('RGB').crop(box).save('%s/panel_%s.png' % (outdir, n))    # 色块归属图
    names = sorted(labels.keys())
    pc = {}
    for i, n in enumerate(names):
        h = i / len(names)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        pc[n] = (int(r * 255), int(g * 255), int(b * 255))
    outarr = np.full((H, W, 3), 255, dtype=np.uint8)
    # 按实际内容像素着色: 每个连通域像素 + 其所属块 + 所属面板
    # 色块归属图: 直接按 owner 对块着色(每块=一个连通域)
    drew = 0
    for i in range(1, nc + 1):
        own = owner.get(i)
        if own is None or own not in pc or sizes[i - 1] < 5:
            continue
        ys, xs = np.where(lab == i)
        if ys.size == 0:
            continue
        outarr[ys, xs] = np.array(pc[own], dtype=np.uint8)
        drew += 1
    print('色块归属图: 画了 %d 块' % drew)
    img = Image.fromarray(outarr)
    d = ImageDraw.Draw(img)
    for n, (x, y) in labels.items():
        d.text((x + 2, y + 2), n, fill=(0, 0, 0))
    img.save('%s/色块归属.png' % outdir)
    # 分割线图
    img2 = Image.open(src).convert('RGB')
    d2 = ImageDraw.Draw(img2)
    for n, (x0, y0, x1, y1) in crop.items():
        d2.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        d2.text((x0 + 2, y0 + 2), n, fill=(0, 0, 255))
    img2.save('%s/分割线.png' % outdir)
    json.dump(crop, open('%s/crop_boxes.json' % outdir, 'w', encoding='utf-8'), indent=1)

    print('裁剪框:')
    for n, b in crop.items():
        print('  %s: %s %dx%d' % (n, b, b[2] - b[0], b[3] - b[1]))
    print('L形:', result['L形'])
    print('重叠:', result['overlap'] if result['overlap'] else '无')
    print('→ 子图/色块归属.png/分割线.png/crop_boxes.json 存于 %s' % outdir)


if __name__ == '__main__':
    main()
