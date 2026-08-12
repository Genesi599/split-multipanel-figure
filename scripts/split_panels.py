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


def assign_blocks(blocks, panels, anchor_pad=0):
    """每个合并块按 bbox 左上角归属面板 → {块Index: 面板名}
    注: anchor_pad 保留参数签名但本实现固定为0(实测锚点左移对交界小字无效且会致面板重叠)。"""
    owner = {}
    for bi, blk in enumerate(blocks):
        bx0, by0 = blk['x0'], blk['y0']
        for n, (x0, y0, x1, y1) in panels.items():
            if x0 <= bx0 < x1 and y0 <= by0 < y1:
                owner[bi] = n
                break
    return owner


def apply_anchor_pad(labels, panels, anchor_pad):
    """返回扩张后的面板框(仅用于归属判定示意)"""
    return {n: (x0 - anchor_pad, y0 - anchor_pad, x1, y1) for n, (x0, y0, x1, y1) in panels.items()}


def crop_for(blocks, owner, labels, panels, W, H):
    """面板 = 归属它的块集合并集 bbox"""
    crop = {}
    for n, (x0, y0, x1, y1) in panels.items():
        lx, ly = labels[n]
        allx = [lx]
        ally = [ly]
        for bi, own in owner.items():
            if own == n:
                b = blocks[bi]
                allx.extend([b['x0'], b['x1']])
                ally.extend([b['y0'], b['y1']])
        crop[n] = (int(max(0, min(allx) - 3)), int(max(0, min(ally) - 3)),
                   int(min(W, max(allx) + 3)), int(min(H, max(ally) + 3)))
    return crop


def result_for(blocks, owner, labels, panels, anchor_pad=0):
    """L 形验证 + 重叠检查
    anchor_pad: L形锚点左移量, 允许色块贴边到字母左上扩展区。"""
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

    # 1. 面板粗边界
    arr = load_arr(src)
    H, W, _ = arr.shape
    content = ~((arr[:, :, 0] >= 235) & (arr[:, :, 1] >= 235) & (arr[:, :, 2] >= 235))
    if panels_json is None:
        panels_json = auto_panels(labels, content, H, W)
        print('自动面板边界:')
        for n, p in panels_json.items():
            print('  %s: %s' % (n, p))
    # 抠掉标签区
    for n, (x, y) in labels.items():
        content[max(0, y - 5):y + 16, max(0, x - 4):x + 13] = False
    if dilate > 0:
        content = ndimage.binary_dilation(content, structure=np.ones((3, 3)), iterations=dilate)
    lab, nc = ndimage.label(content, structure=np.ones((3, 3)))
    sizes = ndimage.sum(content, lab, range(1, nc + 1))

    # 2. 连通域 / 合并块 → 归属
    if do_merge:
        blocks, gap_thr = merge_words(sizes, lab)
        print('间隙合并(阈值%spx): %d连通域 → %d块' % (gap_thr, nc, len(blocks)))
        owner = assign_blocks(blocks, panels_json, anchor_pad)
    else:
        blocks = []
        owner = {}
        for i in range(1, nc + 1):
            if sizes[i - 1] < 12:
                continue
            ys, xs = np.where(lab == i)
            blocks.append({'x0': int(xs.min()), 'x1': int(xs.max()),
                           'y0': int(ys.min()), 'y1': int(ys.max())})
            bi = len(blocks) - 1
            for n, (x0, y0, x1, y1) in panels_json.items():
                if x0 <= int(xs.min()) < x1 and y0 <= int(ys.min()) < y1:
                    owner[bi] = n
                    break

    # 3. 面板=块集合并集
    crop = crop_for(blocks, owner, labels, panels_json, W, H)
    # 4. 验证
    result = result_for(blocks, owner, labels, panels_json)
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
    # 5. 空间一致性校验: 迭代解决最近邻归属冲突(大块为准), 直到无冲突或不再变化
    suspects_idx = set()
    orig_owner = dict(owner)
    it = 0
    final_suspects = []
    prev_n = None
    # 阶段1: 大块为准迭代(到振荡/停滞)
    while True:
        it += 1
        suspects = nearest_owner_conflict(blocks, owner, max_dist=60)
        if not suspects:
            print('无冲突 ✓')
            break
        for s in suspects:
            suspects_idx.add(s['block'])
            suspects_idx.add(s['nearest_block'])
        owner, reassigned = resolve_conflicts_by_size(blocks, owner, suspects)
        if not reassigned:
            final_suspects = suspects
            break
        print('大块为准 迭代%d: 冲突%d, 重分配%d' % (it, len(suspects), len(reassigned)))
        if it >= 10:
            final_suspects = suspects
            break
    # 阶段2: 多数表决收敛残余冲突
    if final_suspects:
        owner, reassigned = resolve_conflicts_by_majority(blocks, owner, final_suspects, k=8)
        print('多数表决重分配 %d 块:' % len(reassigned))
        for bi, old, new in reassigned:
            print('  块[%d]: %s → %s' % (bi, old, new))
        # 阶段3: 最后一轮大块为准, 让面积大的冲突块定输赢
        final3 = nearest_owner_conflict(blocks, owner, max_dist=60)
        if final3:
            owner, reassigned3 = resolve_conflicts_by_size(blocks, owner, final3)
            if reassigned3:
                print('最终大块为准重分配 %d 块:' % len(reassigned3))
                for bi, old, new in reassigned3:
                    print('  块[%d]: %s → %s' % (bi, old, new))
            final3 = nearest_owner_conflict(blocks, owner, max_dist=60)
        if final3:
            print('最终仍剩 %d 组冲突(无法完全消除):' % len(final3))
            for s in final3[:10]:
                blk = blocks[s['block']]
                print('  块[%d]归%s(area%d) vs 块[%d]归%s(area%d) 距离%.1f (%d,%d)-(%d,%d)' % (
                    s['block'], s['own'], s['area'], s['nearest_block'], s['nearest'], s['nearest_area'],
                    s['dist'], blk['x0'], blk['y0'], blk['x1'], blk['y1']))
            final_suspects = final3
        else:
            final_suspects = []
    if 'suspects' in locals() and suspects:
        for s in suspects:
            blk = blocks[s['block']]
            print('  块[%d]归%s(area%d) vs 块[%d]归%s(area%d) 距离%.1f (%d,%d)-(%d,%d)' % (
                s['block'], s['own'], s['area'], s['nearest_block'], s['nearest'], s['nearest_area'], s['dist'],
                blk['x0'], blk['y0'], blk['x1'], blk['y1']))
    # 重算裁剪框 + 重叠
    crop = crop_for(blocks, owner, labels, panels_json, W, H)
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
    drew = 0
    for bi, own in owner.items():
        if own is None or own not in pc:
            continue
        blk = blocks[bi]
        # 填充块 bbox 区域(仅内容像素?简化: 整块矩形填色)
        outarr[max(0, blk['y0']):blk['y1'] + 1, max(0, blk['x0']):blk['x1'] + 1] = np.array(pc[own], dtype=np.uint8)
        drew += 1
    print('色块归属图: 画了 %d 块' % drew)
    img = Image.fromarray(outarr)
    d = ImageDraw.Draw(img)
    for n, (x, y) in labels.items():
        d.text((x + 2, y + 2), n, fill=(0, 0, 0))
    # 可疑块画特殊标记: 已解决冲突=黑框红×; 剩余无法解决的冲突=绿框
    for bi in suspects_idx:
        if bi >= len(blocks):
            continue
        blk = blocks[bi]
        # 是否是剩余冲突(绿框)
        is_final = any(s['block'] == bi or s['nearest_block'] == bi for s in final_suspects)
        outline = (0, 180, 0) if is_final else (0, 0, 0)
        d.rectangle([blk['x0'] - 1, blk['y0'] - 1, blk['x1'] + 1, blk['y1'] + 1],
                    outline=outline, width=2)
        cx = (blk['x0'] + blk['x1']) // 2
        cy = (blk['y0'] + blk['y1']) // 2
        if not is_final:
            d.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0), width=2)
            d.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=(255, 0, 0), width=2)
        else:
            # 剩余冲突画双圈
            d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(0, 180, 0), width=2)
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
