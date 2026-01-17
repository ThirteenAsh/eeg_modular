# EEG Emotion 模块化工程指南

本文档面向项目组其他同学：快速理解工程结构、如何运行/复现实验、如何新增模型与扩展。

> 关键词：**配置驱动**、**统一数据入口**、**统一输出产物**、**可对比实验**、**可插拔模型**。


## 1. 解决了什么问题

在迁移前，训练代码往往表现为：
- 每个模型一份“巨脚本”，数据读取/预处理/训练/可视化混在一起
- 换一个模型或改一个预处理要复制粘贴
- 输出文件散落，难以对比多次实验结果

现在工程提供：
- **统一的数据预处理接口**（tabular 与 sequence 各一套）
- **统一模型接口**（sklearn / torch / tf）
- **统一训练入口**（脚本 + yaml 配置）
- **统一输出目录与指标格式**（`outputs/<run>/metrics.json`），可自动汇总排序对比
- **统一可视化产物**（至少混淆矩阵；LSTM 额外：曲线图、UMAP+SVM边界图等）


## 2. 快速开始（Quickstart）

### 2.1 环境依赖（建议 Python 3.11）

基础依赖（必须）：
- numpy / pandas / scikit-learn / joblib / matplotlib
- PyYAML（用于读配置）
- torch（CNN 路线）
- tensorflow（LSTM 路线）

可选依赖（不装也能跑主流程，只是跳过部分图）：
- seaborn（Seaborn 风格混淆矩阵）
- umap-learn（UMAP + SVM decision boundary）

安装示例：
```bash
pip install -U numpy pandas scikit-learn joblib matplotlib pyyaml
pip install -U torch tensorflow
pip install -U seaborn umap-learn   # 可选
```


### 2.2 工程最常用的 4 个命令

1) 训练 sklearn（统计特征 tabular 路线）
```bash
python -m scripts.train -c configs/<your_sklearn_config>.yaml
```

2) 训练 CNN（torch，多模态 + 可选 CVAE）
```bash
python -m scripts.train_cnn_dl -c configs/cnn.yaml
```

3) 训练 LSTM（tensorflow，AE 预训练 + 编码 + mixup + 分类器）
```bash
python -m scripts.train_lstm_dl -c configs/lstm.yaml
```

4) 汇总实验（把 outputs 下所有 run 自动对比排序）
```bash
python -m scripts.compare_runs --outputs outputs --out outputs/_summary
```


## 3. 工程目录结构（建议先看这个）

> 实际文件可能会随着你们继续扩展而增加，但“职责分层”建议保持不变。

```text
eeg_emotion/
  config/
    loader.py               # 读取/校验 yaml 配置
  features/
    csv_stats.py            # tabular 特征入口：data/<emotion>/... -> (X,y)
    sequence/
      extract.py            # sequence 特征入口：多 csv -> (X,y)
      augment.py            # sequence 数据增强：noise/warp/mask/crop + mixup
  preprocess/
    tabular.py              # tabular 预处理：impute/scale/(可选)PCA/(可选)augment
    sequence.py             # sequence 预处理：impute/scale/(可选)PCA（按时间步 reshape）
  models/
    sklearn/
      svm.py                # SVM adapter
      mlp.py                # MLP adapter
      rf.py                 # RandomForest adapter
    torch/
      multimodal_cvae_cnn.py # CNN 分类器（可拼接 CVAE latent）
      cvae.py               # CVAE 动态加载（checkpoint 兼容）
    tf/
      lstm_ae.py            # LSTM AutoEncoder（attention）
      lstm_clf.py           # 编码后分类器（MLP）
  dl/
    common.py               # 通用：seed 等
    torch/
      data.py               # 多模态 npy loader（强健 shape/label 处理）
      losses.py             # Weighted FocalLoss
      trainer.py            # torch kfold 训练+评估+写 metrics
    tf/
      trainer.py            # TF 侧统一输出（当前主要由 train_lstm_dl.py 驱动）
  train/
    metrics.py              # 统一分类指标：accuracy + report 等
    weights.py              # 类别权重计算（含 Sad 手动加权）
  viz/
    confusion_matrix.py     # matplotlib 混淆矩阵（默认）
    seaborn_cm.py           # seaborn 混淆矩阵（可选）
    umap_boundary.py        # UMAP + SVM decision boundary（可选）
  utils/
    logging.py              # 日志
    paths.py                # run 目录创建：outputs/<run>/{models,figures,logs,artifacts}

scripts/
  train.py                  # sklearn/tabular 统一训练入口（配置驱动）
  train_cnn_dl.py           # CNN 入口（torch；含 CVAE/Focal/Scheduler/KFold）
  train_lstm_dl.py          # LSTM 入口（tf；含 AE+编码+mixup+可视化）
  compare_runs.py           # 扫描 outputs/*/metrics.json 自动汇总

configs/
  cnn.yaml                  # CNN 配置模板
  lstm.yaml                 # LSTM 配置模板
  *.yaml                    # sklearn 侧各种实验配置（你们后续自己维护）

outputs/
  <run_name_or_timestamp>/
    metrics.json            # compare_runs 统一读取的产物
    summary.json            #（部分路线会有）
    models/                 # 权重/模型文件
    figures/                # 混淆矩阵、训练曲线、UMAP 等图
    logs/                   # train.log
    artifacts/              # 预处理器（imputer/scaler/pca）等
```


## 4. “统一输出产物”规则（强烈建议所有人遵守）

每次运行都会生成一个 run 目录：
- **metrics.json**：必须包含 `accuracy`，建议包含 `report` 与 `best_params`
- figures/：可视化全部放这里
- models/：模型权重/导出模型放这里
- logs/train.log：训练日志
- artifacts/：任何“可复用且和训练集拟合相关”的东西（预处理器、label encoder 等）

这样 `scripts.compare_runs` 才能跨模型/跨框架统一对比。


## 5. 三条主线怎么用

### 5.1 sklearn / tabular（统计特征）路线
适用场景：快速 baseline、可解释性、训练速度快。

典型流程：
1) 从 data/<emotion>/... 读取 csv -> 提取统计特征 -> 形成 X,y
2) fit tabular preprocessor（仅 train）
3) 训练模型（svm / mlp / rf）
4) 输出 metrics.json + confusion matrix

运行：
```bash
python -m scripts.train -c configs/<svm_or_mlp_or_rf>.yaml
```

你要新增一个 sklearn 模型：
- 在 `eeg_emotion/models/sklearn/` 里新增 `<model>.py`，实现统一接口（fit/predict/save/load）
- 写一个 yaml 指向该模型即可（不需要改训练脚本）


### 5.2 CNN / torch（多模态 npy + 可选 CVAE）路线
适用场景：你已经有 `X_train_*.npy` 这种多模态序列输入；想用 CNN 学到更鲁棒的模式。

运行：
```bash
python -m scripts.train_cnn_dl -c configs/cnn.yaml
```

数据要求（在 `cnn.data_dir` 目录下）：
- `X_train_<modality>.npy`
- `X_test_<modality>.npy`
- `y_train_filtered.npy`（可 one-hot；会自动转 int）
- `y_test_filtered.npy`
- `label_encoder.joblib`（可选，但推荐有；用于 class_names）

关于 CVAE：
- `configs/cnn.yaml` 里设置：
  - `cnn.cvae.enabled: true`
  - `cnn.cvae.checkpoint: ...`
- 默认会 `import cvae_model`；如果你的 `cvae_model.py` 不在工程内，填 `cnn.cvae.py_path` 指向它即可。


### 5.3 LSTM / tensorflow（AE 预训练 + 编码 + mixup + 分类器）路线
适用场景：你希望保持你原始 LSTM 思路：先学表征，再做分类。

运行：
```bash
python -m scripts.train_lstm_dl -c configs/lstm.yaml
```

关键参数都在 `configs/lstm.yaml`：
- `csv_files`：每个 sample 文件夹里希望读取的 csv 列表
- `time_steps`：序列统一长度（不足 padding，过长截断）
- `augment`：按类增强次数（train-only）
- `preprocess`：impute/scale/PCA（按每个时间步进行 reshape 处理）
- `autoencoder`：AE 训练参数
- `mixup`：在 encoded features 上做 mixup
- `classifier`：分类器训练参数 + 学习率策略

可视化（可选）：
- `viz.seaborn_confusion_matrix: true` -> 输出 seaborn 混淆矩阵
- `viz.umap_svm_boundary: true` -> 输出 UMAP+SVM 边界图
需要额外安装 seaborn 与 umap-learn；缺包会自动跳过，不影响主训练。


## 6. configs（配置）怎么写/怎么扩展

建议约定：
- 每个实验一个 yaml（比如 `svm_baseline.yaml`, `mlp_kbest_500.yaml`）
- yaml 里不要写绝对路径（除非团队机器差异太大），尽量用相对路径或在 README 指定目录结构

典型配置字段：
- output.base_dir / output.run_name
- data_dir / emotions / csv_files
- preprocess.xxx
- model.xxx
- train.xxx
- viz.xxx（可选）

工程对 “跑实验” 的核心约定：
- **只通过改 yaml 做实验对比**（尽量不要直接改 python 代码）
- 新功能/新模型通过新增模块实现，而不是在 train 脚本里堆条件分支


## 7. 常见问题（Troubleshooting）

### 7.1 LSTM：LearningRateSchedule + ReduceLROnPlateau 冲突
如果 classifier 使用 `CosineDecay` 等 schedule，就不能再用 ReduceLROnPlateau（它会尝试 set lr）。
工程已做了保护：当 `use_cosine_decay=true` 会自动不启用 ReduceLROnPlateau。

### 7.2 缺 seaborn / umap-learn
会在日志提示“跳过”，但训练仍正常完成。要生成对应图再安装：
```bash
pip install seaborn umap-learn
```

### 7.3 CNN：多模态 npy shape 不一致
loader 会尽量规范到 (N,T,F)（例如自动把 (N,F,T) 转置），但如果你本身 npy 不匹配，仍会报错并指出具体模态/shape。建议统一在生成 npy 的脚本处保证一致性。

### 7.4 如何复现实验结果
- 固定 `seed`
- 不要在训练中手动改脚本逻辑
- 每次运行保留 `configs/*.yaml`（metrics.json 会记录 config_path）


## 8. 团队协作建议

- 在仓库根目录提供：
  - `ENGINEERINREAD_GUIDE.md`（本文）
- 约定所有实验 run 都输出到 `outputs/`（但不要把 outputs 提交到 git；用 .gitignore）
- 每次新增模型/功能：
  - 新增模块文件
  - 新增 yaml 示例
  - README/指南里补一段“如何运行”
