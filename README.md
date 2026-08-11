# split-multipanel-figure

把多面板（multi-panel）文献组图（A/B/C…多个 panel）拆成独立子图 PNG 的工具。
视觉模型坐标不可靠 → 用像素方法（标签连通域检测 + 内容间隙 + 弹性分割线）+ L 形归属规则，
已在多种布局的组图（规则网格 / 不规则 / 交错 / 悬浮标签 / 22 面板 A–V）上验证。

## 特性

- **像素级标签检测**：不依赖视觉模型坐标（视觉坐标常偏 20-60px / 方向错），用暗色连通域 + 高度直方图 / "找A按A高度模板匹配"找面板字母。
- **面板边界** = 标签锚点 + 内容间隙（白色/浅灰背景空行/空列带）。
- **分割线只在白色背景画**，相邻面板共用一条线，内容处折线绕行（可切矩形/弹性线）。
- **L 形归属验证**：每面板色块 bbox 左上角 ≥ 字母锚点，全绿才交付。
- **面板 = 色块集合并集**（不是固定矩形框），自动适配不规则面板。
- 交互式：先自动找标签候选 → 人工/视觉确认标签身份 → 自动分块 + 裁剪 + 验证。

## 安装

```bash
pip install numpy pillow scipy
```

## 用法

```bash
# 一键流程
python scripts/find_labels.py <image> --out labels.json   # 1. 自动定位标签字母
python scripts/verify.py <image> labels.json              # 2. 面板归属验证 → crop_boxes.json
python scripts/autosplit.py <image> outdir --labels labels.json  # 3. 分割+裁剪+验证

# 或分步
python scripts/autosplit.py <image> <outdir>                        # 第1步: 标签候选 → label_candidates.png
python scripts/autosplit.py <image> <outdir> --labels <labels.json> # 第2步: 自动分块+画线+crop+验证
```

`labels.json` 格式：`{"A": [x, y], "B": [x, y], ...}`（标签字母左上角坐标）
可选 `init.json`：`{"A": [x1,y1,x2,y2], ...}`（面板初始区域，跳过自动推断）

输出：`分割线预览.png` / `panel/panel_X.png` / `contact_sheet.png` / `split_report.txt`

## 完整工作流（详见 `SKILL.md`）

1. 全图布局侦察（视觉 glance 看面板数/布局）
2. 像素标签检测（暗色连通域 + 高度直方图 / 先找 A 按 A 高度模板匹配）
3. 面板间隙分析（彩色内容掩码 `任一RGB<225`，非灰度）
4. 归属验证（bbox 左上角归属 + L 形规则 + 分隔规则）
5. 分割线绘制（矩形 / 共享网格弹性线）+ 裁剪 + 逐像素验证

## 关键经验（详见 SKILL.md 历史教训节）

- 内容检测必须用**彩色掩码**（`任一 RGB<225`），灰度会漏浅色内容（粉色细胞/浅蓝染色）→ 分割线穿内容。
- **视觉模型坐标不可靠**（偏 20-60px / 方向错），只用于认字母身份，坐标用像素。
- 面板边界用**标签反推**（左界 = max(标签x−20, 上一面板内容右缘+2)），不用"相邻标签中点"。
- 验证"红线穿内容"必须用**画线前原图掩码**（画线后红黑互斥误报 0）。

## License

MIT
