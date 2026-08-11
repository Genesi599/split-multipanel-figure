# 示例：把一张含 A-G 7 面板的组图拆成 7 张子图

```bash
# 1. 自动定位标签（先找 A → 按 A 高度模板匹配）
python scripts/find_labels.py fig.png --out labels.json

# 2. 面板归属验证，得到裁剪框
python scripts/verify.py fig.png labels.json   # → crop_boxes.json

# 3. 分割 + 裁剪 + 输出
python scripts/autosplit.py fig.png out/ --labels labels.json
# 输出: out/分割线预览.png、out/panel/panel_A.png …、out/contact_sheet.png、out/split_report.txt
```

人工兜底：find_labels 给出候选集（含全部真标签），用字形 print 验证字母身份后，
在 labels.json 里修正误检项；verify.py 报 L形 FAIL 时微调面板边界（见 SKILL.md §4）。
