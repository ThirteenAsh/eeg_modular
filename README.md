# EEG Emotion 模块化工程设计文档

## 1. 项目概述

EEG Emotion是一个基于脑电图(EEG)数据的情感分析模块化工程，采用配置驱动的设计理念，支持多种机器学习和深度学习模型。

### 1.1 核心功能

- 统一的数据预处理接口
- 多种模型支持（SVM、MLP、RF、XGBoost、LSTM、CNN、混合模型等）
- 配置驱动的训练流程
- 丰富的可视化输出
- 支持多模型比较
- 统一的输出目录结构
- 混合模型集成（soft voting + stacking）

### 1.2 项目结构

```
./
├── eeg_emotion/      # 主包目录
│   ├── config/       # 配置管理
│   ├── dl/           # 深度学习训练实现（PyTorch）
│   ├── features/     # 特征提取
│   ├── models/       # 模型实现
│   │   ├── sklearn/  # sklearn模型
│   │   └── torch/    # PyTorch模型
│   ├── preprocess/   # 预处理
│   ├── report/       # 报告生成
│   ├── train/        # 训练相关
│   ├── utils/        # 工具函数
│   └── viz/          # 可视化
├── configs/          # 配置文件目录
├── data/             # 数据目录
├── features/         # 特征相关文件
├── scripts/          # 训练脚本
└── outputs/          # 输出目录
```

## 2. 配置驱动设计

### 2.1 配置文件结构

配置文件采用YAML格式，包含数据、模型、预处理、训练和可视化等配置项：

```yaml
# 数据配置
data_dir: ./data
emotions: [happy, sad, normal]
csv_files: [att.csv, med.csv, powerspec.csv]

# 模型配置
model:
  type: svm
  solver: svc
  kernel: rbf
  C: 1.0
  gamma: scale
  class_weight: balanced

# 混合模型配置示例
model:
  type: hybrid
  voting_method: soft  # "soft" 或 "hard"
  use_stacking: true
  
  # MLP配置
  mlp_config:
    param_grid:
      hidden_layer_sizes: [(100,), (100, 50)]
      activation: [relu, tanh]
      solver: [adam, sgd]
    cv: 5
    n_jobs: -1
    random_state: 42
  
  # RF配置
  rf_config:
    param_grid:
      n_estimators: [100, 200]
      max_depth: [None, 10, 20]
      min_samples_split: [2, 5]
    cv: 5
    n_jobs: -1
    random_state: 42
  
  # SVM配置
  svm_config:
    solver: svc
    probability: true
    param_grid:
      kernel: [rbf, linear]
      C: [0.1, 1, 10]
      gamma: [scale, auto]
  
  # XGBoost配置
  xgb_config:
    param_grid:
      n_estimators: [100, 200]
      max_depth: [3, 6]
      learning_rate: [0.01, 0.1]
    cv: 5
    n_jobs: -1
    random_state: 42

# 预处理配置
preprocess:
  scale: true
  select_k_best: null
  pca_n_components: null
  augment: false

# 训练配置
train:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

# 可视化配置
viz:
  seaborn_confusion_matrix: true
  umap_boundary: true
  training_curves: true
```

### 2.2 配置加载与使用

```python
from eeg_emotion.config.loader import load_config, get, require

# 加载配置文件
cfg = load_config("configs/svm.yaml")

# 获取配置项
emotions = require(cfg, "emotions", list)
model_cfg = require(cfg, "model", dict)
model_type = require(model_cfg, "type", str)
viz_cfg = get(cfg, "viz", {})
seaborn_cm = bool(viz_cfg.get("seaborn_confusion_matrix", False))
```

## 3. 可视化功能

### 3.1 混淆矩阵

支持两种混淆矩阵风格：

1. **matplotlib风格**：简洁的混淆矩阵
2. **seaborn风格**：带有热力图效果的混淆矩阵

**配置示例**：

```yaml
viz:
  seaborn_confusion_matrix: true  # true: seaborn风格, false: matplotlib风格
```

### 3.2 训练曲线

支持绘制训练过程中的准确率和损失曲线，包含：

- 训练集和验证集的准确率曲线
- 训练集和验证集的损失曲线

**配置示例**：

```yaml
viz:
  training_curves: true
```

### 3.3 UMAP边界图

生成UMAP降维后的SVM决策边界图，支持：

- 不同的UMAP配置参数
- 多种边界绘制模式（填充、线条、两者结合）
- 自适应的颜色和样式

**配置示例**：

```yaml
viz:
  umap_boundary: true
```

### 3.4 多模型UMAP边界图

在单个UMAP投影上绘制多个模型的决策边界，便于直观比较不同模型的决策边界差异。

**注意**：目前支持sklearn模型（SVM、MLP、RF、XGBoost）和混合模型，CNN和LSTM等深度学习模型需要不同的处理方式，暂不支持。

**使用方法**：

```bash
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml configs/xgb.yaml configs/hybrid.yaml -o outputs/multi_model_umap
```

**输出结果**：
- `multi_model_umap_boundary.png`：包含所有模型决策边界的UMAP图，带有清晰的图例说明

**图例说明**：
- 左侧图例显示数据点对应的情感类别
- 右侧图例显示不同颜色和线型对应的模型名称
- 每种模型使用不同的颜色和线型，便于区分

**优势**：
- 只需要一张图就能直观比较多个模型的决策边界
- 清晰的图例说明，便于理解
- 避免了多个单独UMAP图的冗余
- 统一的UMAP投影，确保模型边界的可比性

## 4. 运行实例汇总

### 4.1 SVM模型训练

**配置文件**：`configs/svm.yaml`

**运行命令**：
```bash
python -m scripts.train -c configs/svm.yaml
```

### 4.2 MLP模型训练

**配置文件**：`configs/mlp.yaml`

**运行命令**：
```bash
python -m scripts.train -c configs/mlp.yaml
```

### 4.3 RF模型训练

**配置文件**：`configs/rf.yaml`

**运行命令**：
```bash
python -m scripts.train -c configs/rf.yaml
```
### 4.4 XGBoost模型训练

**配置文件**：`configs/xgb.yaml`

##### 运行

```bash
python -m scripts.train -c configs/xgb.yaml
```

### 4.5 LSTM模型训练

**配置文件**：`configs/lstm.yaml`

**运行命令**：
```bash
python -m scripts.train_lstm_dl -c configs/lstm.yaml
```

### 4.6 CNN模型训练

**配置文件**：`configs/cnn.yaml`

**运行命令**：
```bash
python -m scripts.train_cnn_dl -c configs/cnn.yaml
```

### 4.7 混合模型训练

**配置文件**：`configs/hybrid.yaml`

**运行命令**：
```bash
python -m scripts.train -c configs/hybrid.yaml
```

**模型说明**：
- 混合模型集成了4个机械模型：MLP、RF、SVM和XGBoost
- 支持soft voting和stacking两种集成方式
- 使用LogisticRegression作为stacking的元分类器

### 4.8 多模型UMAP比较

**运行命令**：
```bash
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml configs/xgb.yaml configs/hybrid.yaml -o outputs/multi_model_umap
```



##### 配置要点

- 固定参数放在：model.xgboost 下（例如 objective / eval_metric / tree_method）
- 网格搜索参数放在：model.param_grid 下
- 多分类建议：
  - objective: multi:softprob
  - eval_metric: mlogloss
  - num_class 可以不写；训练时会从标签自动推断（No-GridSearch 模式），或你也可在 model.xgboost.num_class 显式指定。

##### 并行说明

- GridSearchCV 自己会并行（n_jobs）
- 为避免“外层并行 + XGBoost 内层并行”线程嵌套，本补丁在启用 param_grid 时将 XGBoost 的 n_jobs 强制设为 1。

## 5. 可视化与输出

### 5.1 可视化开关

在配置文件中通过 `viz` 控制图表输出：

```yaml
viz:
  seaborn_confusion_matrix: true
  umap_boundary: true
```

当前三个主训练脚本都支持上述开关：
- `scripts/train.py`
- `scripts/train_lstm_dl.py`
- `scripts/train_cnn_dl.py`

### 5.2 标准输出目录

每次训练会在 `outputs/` 下创建运行目录，典型结构如下：

```text
outputs/<run>/
├── artifacts/
├── figures/
├── logs/
├── models/
└── metrics.json
```

## 6. 模块说明

### 6.1 `eeg_emotion/config`

- 配置加载与校验（YAML/JSON）
- 核心文件：`loader.py`

### 6.2 `eeg_emotion/features`

- CSV 统计特征提取
- NPY 多模态特征加载
- 序列特征提取与增强

### 6.3 `eeg_emotion/models`

- `sklearn/`：SVM、MLP、RF、XGBoost、Hybrid
- `torch/`：LSTM/CVAE/CNN 相关模型

### 6.4 `eeg_emotion/preprocess`

- 表格预处理（缺失值、标准化、特征选择、PCA）
- 序列预处理

### 6.5 `eeg_emotion/dl/torch`

- PyTorch 数据、损失函数、训练器（含 K-fold）

### 6.6 `eeg_emotion/viz`

- 混淆矩阵、训练曲线、UMAP 边界图

## 7. 最佳实践

1. 使用配置驱动实验，不在脚本里写死参数。
2. 训练集与测试集严格隔离，预处理仅在训练集拟合。
3. 固定随机种子并记录到运行日志/指标文件。
4. 所有实验统一输出到 `outputs/`，避免手工散落文件。
5. 变更训练流程后，先运行一致性校验再做横向对比。

## 8. 常见问题

### 8.1 UMAP 图未生成

- 检查 `viz.umap_boundary` 是否为 `true`
- 检查 `umap-learn` 是否安装

### 8.2 结果波动较大

- 检查 seed 设置是否一致
- 检查是否开启了随机增强

### 8.3 CPU 环境训练失败（XGBoost）

- 将配置中的 XGBoost `device` 调整为 `cpu`

## 9. 总结

当前工程以配置驱动为核心，覆盖 sklearn 与 PyTorch 两条训练路线，具备统一预处理、统一评估与统一输出结构。建议优先维护配置一致性与实验可复现性，以保证模型对比结论可靠。
