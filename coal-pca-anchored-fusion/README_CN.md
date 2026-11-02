# PCA锚定双自编码器融合（GitHub公开代码）

这个仓库提供论文《PCA-Anchored Dual-Autoencoder Fusion of Raman and Near-Infrared Spectra for Cross-Source Coal Volatile Matter Prediction》对应的**代码整理版**，不含真实煤样NIR/Raman光谱、挥发分标签、样品编号、训练检查点和逐样品预测文件。

**特别说明：** 最初三种子论文结果对应的AE预训练完整源码未能从旧归档中恢复。此处公开的AE结构根据原检查点重建且推理结构已核对；`fresh` 模式的独立AE预训练代码来自后续冻结的10随机种子复核，不应描述为初始论文三种子的逐字节精确复现。`archived` 模式需要用户自行提供**可信的原始AE检查点**，可以复核历史编码器下的下游ANN表现。详见 [PROVENANCE.md](PROVENANCE.md)。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

## 不用真实数据先测试

```bash
coal-fusion demo-data --out data/synthetic_demo.npz
coal-fusion fresh --data data/synthetic_demo.npz --out runs/demo --holds S1 --seeds 11 --components 6 --ae-epochs 5 --ann-epochs 15
```

生成的纯虚构数据只是用于检验代码能否运行，没有科研意义。

## 使用自有数据开展五源留一验证

本地私有 `.npz` 文件必须包含 `xn`、`xr`、`y`、`sources`、`ids`。`xn/xr` 是**每个重复谱预处理后再按物理样品求平均**的模型输入矩阵，程序不会自动从未经处理的原始仪器文件转换。论文原有维度为NIR 239、Raman 1024；每个物理样本只能有一个对应的参考值及唯一ID。

从头独立训练（重新构建的AE训练实现）：

```bash
coal-fusion fresh --data /你的私有目录/data.npz --out runs/ten_seeds --seeds 11,23,47,71,101,131,173,211,257,307
```

历史编码器复核（需本地拥有原始checkpoint，且必须来源可信）：

```bash
coal-fusion archived --data /你的私有目录/data.npz --checkpoints /你的私有目录/outer --out runs/original_seeds --seeds 7,123,3407
```

增加传统PLS基线：

```bash
coal-fusion pls --data /你的私有目录/data.npz --out runs/pls --max-lv 20
```

结果会保存至本地 `runs/` 目录，并被 `.gitignore` 排除。这些结果通常包含样品编号、化验值和预测值，**绝不能因为代码可以公开就顺便上传**。

## 提交GitHub前

1. 运行 `python scripts/check_release.py`；
2. 确认只上传本仓库目录，不要把旧的40MB交接包或包含真实数据的复现ZIP上传；
3. 检查 `git status --short`，确认没有真实数据、`.pt` 检查点或带标签的CSV；
4. 选择软件许可证（例如MIT），作者最终确认后再添加；
5. 后续若找到原始AE预训练代码，应另外补充并注明版本来源，不要静默覆盖此重建实现。

投稿时如果披露代码仓库，请不要宣称所有原始结果都可以在无检查点情况下逐字节复现。详见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。
