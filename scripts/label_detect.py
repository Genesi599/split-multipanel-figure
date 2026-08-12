# -*- coding: utf-8 -*-
"""
纯代码面板标签检测器 —— 高度筛 + OCR 置信度（无 AI 视觉）

完整流程(已验证, G4/G5 实测):
  1. 二值化找暗色连通域, 统计方块高度
  2. 找"正文文字高度峰"(最常见高度) = body_h; 标签显著高于正文
  3. 筛 高度>[body_h×ratio] + 宽高比0.6-2.2(排除竖条) + 填充率0.25-0.95 + 孤立(周围无粘连)
     → 得到标签候选(位置+尺寸)
  4. （可选）用 OCR(EasyOCR) 读每个候选框内容, 取最高置信度
     → 置信度>conf_thr 的才是真标签, 读出的字母 = 身份
  5. 输出 {字母: [x,y]} 或候选列表

用法:
  python label_detect.py <img> [--out labels.json] [--ratio 1.5]
         [--ocr] [--conf 0.8] [--view]
  --ocr    启用 OCR 确认身份(需要 easyocr)
  --conf   置信度阈值, 默认0.8
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def load(src):
    arr = np.array(Image.open(src).convert('RGB'))
    return arr


def detect_candidates(arr, ratio=1.5, min_area=20, max_area=400):
    """高度峰筛法定位标签候选框, 返回 (body_h, candidates)"""
    dark = (arr[:, :, 0] < 235) & (arr[:, :, 1] < 235) & (arr[:, :, 2] < 235)
    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
    sizes = ndimage.sum(dark, lab, range(1, n + 1))

    # 高度直方图: 正文文字 = 最常见高度(在合理范围5-25px内选峰, 排除1-2px阴影碎片)
    from collections import Counter
    hcount = Counter()
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if 5 <= h <= 25 and w <= 60:
            hcount[h] += 1
    if not hcount:
        return None, []
    # 正文高度 = 5-25px 内最常见的高度(不是全局最多, 避免1-2px碎块)
    body_h = hcount.most_common(1)[0][0]
    min_h = max(10, int(body_h * ratio))

    cands = []
    for i in range(1, n + 1):
        s = int(sizes[i - 1])
        if s < min_area or s > max_area:
            continue
        ys, xs = np.where(lab == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h < min_h:
            continue
        if not (h / w >= 0.6 and w <= 20):  # 高宽比只要>=0.6(排除横条), 不设上限(容纳窄I/J); 宽度只设上限
            continue
        fill = s / (w * h)
        if not (0.2 <= fill <= 1.0):  # 填充率: 上限放宽到1.0(容纳实心窄字母I/J, fill≈1)
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

    # 窄字母(I/J)补充搜索: 高度在标签范围、宽1-8竖窄字, 放宽孤立度(否则I/J会漏检)
    # 用于补上高度筛+严格孤立滤掉的窄标签(如 I w3h14)
    if cands:
        known_h = [c['h'] for c in cands]
        tag_h = int(np.median(known_h)) if known_h else int(body_h * ratio)
        for i in range(1, n + 1):
            a = int(sizes[i - 1])
            if a < 8 or a > 200:
                continue
            ys, xs = np.where(lab == i)
            h2 = ys.max() - ys.min() + 1
            w2 = xs.max() - xs.min() + 1
            # 竖窄字: 高接近标签高度, 宽1-8, 高/宽>=1.5
            if not (tag_h - 3 <= h2 <= tag_h + 3 and 1 <= w2 <= 8):
                continue
            if h2 < w2 * 1.5:
                continue
            # 已在候选? (按位置去重)
            if any(abs(c['x'] - xs.min()) <= 2 and abs(c['y'] - ys.min()) <= 2 for c in cands):
                continue
            cands.append({'x': int(xs.min()), 'y': int(ys.min()),
                          'w': int(w2), 'h': int(h2), 'area': a})
    return body_h, cands


def ocr_verify(cands, src, conf_thr=0.8):
    """用 OCR 读候选框内容, 返回 (真标签{字母:[x,y]}, 保留最大识别字母列表)"""
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    img = Image.open(src).convert('RGB')
    labels = {}
    letter_kept = []
    for i, c in enumerate(cands):
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        pad = 4
        crop = img.crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
        crop = crop.resize((max(crop.width * 6, 60), max(crop.height * 6, 60)), Image.LANCZOS)
        res = reader.readtext(np.array(crop), detail=1,
                              allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=False)
        if not res:
            letter_kept.append((c, '', 0.0))
            continue
        box, text, conf = max(res, key=lambda r: r[2])
        letter_kept.append((c, text.upper().strip(), float(conf)))
        if conf >= conf_thr and len(text.strip()) == 1:
            # 按位置先后分配字母(同一label去重: 同字母只留置信度最高)
            ch = text.strip().upper()
            labels[ch] = [c['x'], c['y']]
    return labels, letter_kept


def main():
    src = sys.argv[1]
    out = 'labels.json'
    ratio = 1.5
    use_ocr = '--ocr' in sys.argv
    conf_thr = 0.8
    view = '--view' in sys.argv
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    if '--ratio' in sys.argv:
        ratio = float(sys.argv[sys.argv.index('--ratio') + 1])
    if '--conf' in sys.argv:
        conf_thr = float(sys.argv[sys.argv.index('--conf') + 1])

    arr = load(src)
    body_h, cands = detect_candidates(arr, ratio)
    print(f'正文高度峰={body_h}px, 标签需>={int(body_h*ratio)}px, 候选{len(cands)}个')
    cands.sort(key=lambda c: (c['y'], c['x']))

    if use_ocr:
        labels, kept = ocr_verify(cands, src, conf_thr)
        print(f'OCR 置信度>={conf_thr} 的真标签:')
        for ch in sorted(labels):
            print(f'  {ch}: {labels[ch]}')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=1)
        print('已存', out)
        if view:
            img = Image.fromarray(arr); d = ImageDraw.Draw(img)
            for ch, (x, y) in labels.items():
                d.ellipse([x - 3, y - 3, x + 12, y + 14], outline=(255, 0, 0), width=2)
                d.text((x + 1, y + 1), ch, fill=(255, 0, 0))
            img.save('labels_detect.png')
            print('已存 labels_detect.png')
    else:
        for i, c in enumerate(cands):
            print(f"  #{i} ({c['x']},{c['y']}) {c['w']}x{c['h']} area={c['area']}")
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(cands, f, indent=1)
        print('已存', out)
        if view:
            img = Image.fromarray(arr); d = ImageDraw.Draw(img)
            for i, c in enumerate(cands):
                d.rectangle([c['x']-2, c['y']-2, c['x']+c['w']+2, c['y']+c['h']+2],
                            outline=(255, 0, 0), width=2)
                d.text((c['x']+2, c['y']-2), str(i), fill=(255, 0, 0))
            img.save('labels_detect.png')
            print('已存 labels_detect.png')


if __name__ == '__main__':
    main()
