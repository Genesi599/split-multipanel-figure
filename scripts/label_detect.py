# -*- coding: utf-8 -*-
"""
纯代码面板标签检测器（无 AI）—— 高度峰法
原理(已验证):
  1. 二值化找暗色连通域, 统计每个方块(连通域)的高度
  2. 找"正文文字高度峰"(最常见高度) = body_h; 标签应显著高于正文
  3. 筛出 高度>[body_h×1.4] 的方块
  4. 再过滤: 宽高比 0.6-2.2(排除竖条/长字串), 填充率 0.25-0.95, 孤立(周围无粘连)
  5. 得到标签候选(位置+尺寸)
用法:
  python label_detect.py <img> [--out labels.json] [--view] [--ratio 1.4]
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def detect(src, ratio=1.4):
    arr = np.array(Image.open(src).convert('RGB'))
    dark = (arr[:, :, 0] < 200) & (arr[:, :, 1] < 200) & (arr[:, :, 2] < 200)
    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
    sizes = ndimage.sum(dark, lab, range(1, n + 1))

    # 高度直方图: 正文文字 = 最常见高度(数量最多的档, 排除1-2px噪声)
    from collections import Counter
    hcount = Counter()
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h <= 60 and w <= 60:  # 只统计小方块(文字/字母), 排除大图块
            hcount[h] += 1
    if not hcount:
        return []
    body_h = hcount.most_common(1)[0][0]  # 正文高度峰
    min_h = int(body_h * ratio)  # 标签最小高度

    # 筛高度达标 + 形状像孤立大写字母
    cands = []
    for i in range(1, n + 1):
        s = int(sizes[i - 1])
        if s < 20 or s > 400:
            continue
        ys, xs = np.where(lab == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h < min_h:
            continue
        if not (0.6 <= h / w <= 2.2):
            continue
        fill = s / (w * h)
        if not (0.25 <= fill <= 0.95):
            continue
        # 孤立度: 周围4px无其它暗像素
        pad = 4
        x0, x1 = max(0, xs.min() - pad), xs.max() + pad + 1
        y0, y1 = max(0, ys.min() - pad), ys.max() + pad + 1
        region = dark[y0:y1, x0:x1].copy()
        region[ys.min() - y0:ys.max() - y0 + 1, xs.min() - x0:xs.max() - x0 + 1] = False
        if region.sum() > 10:
            continue
        cands.append({'x': int(xs.min()), 'y': int(ys.min()),
                      'w': int(w), 'h': int(h), 'area': int(s)})
    return body_h, cands


def main():
    src = sys.argv[1]
    out = 'labels.json'
    ratio = 1.4
    view = '--view' in sys.argv
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    if '--ratio' in sys.argv:
        ratio = float(sys.argv[sys.argv.index('--ratio') + 1])
    body_h, cands = detect(src, ratio)
    print(f'正文高度峰={body_h}px, 标签需>={int(body_h*ratio)}px, 候选{len(cands)}个:')
    cands.sort(key=lambda c: (c['y'], c['x']))
    for c in cands:
        print(f"  ({c['x']:4d},{c['y']:4d}) {c['w']}x{c['h']} area={c['area']}")
    # 存候选(不含身份)
    json.dump(cands, open(out, 'w', encoding='utf-8'), indent=1)
    print('已存', out)
    if view:
        arr = np.array(Image.open(src).convert('RGB'))
        img = Image.fromarray(arr); d = ImageDraw.Draw(img)
        for i, c in enumerate(cands):
            d.rectangle([c['x']-2, c['y']-2, c['x']+c['w']+2, c['y']+c['h']+2],
                        outline=(255, 0, 0), width=2)
            d.text((c['x']+2, c['y']-2), str(i), fill=(255, 0, 0))
        img.save('labels_detect.png')
        print('已存 labels_detect.png')


if __name__ == '__main__':
    main()
