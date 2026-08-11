#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multi-panel 组图拆分子图工具集

一键流程:
  1. 标签自动定位   : python scripts/find_labels.py <img> [--out labels.json]
  2. 面板归属验证   : python scripts/verify.py <img> labels.json     → crop_boxes.json
  3. 分割+裁剪      : python scripts/autosplit.py <img> outdir --labels labels.json
  (autosplit 第1步也可单独跑产生标签候选网格供人工认字母)

单条 autosplit 用法:
  python scripts/autosplit.py <img> <outdir>                        # 标签候选网格
  python scripts/autosplit.py <img> <outdir> --labels <labels.json> # 分块+画线+crop+验证
"""
