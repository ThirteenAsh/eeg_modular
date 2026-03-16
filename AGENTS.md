# EEG Emotion 模块化工程 - Agent 上下文文档

## 项目概述

EEG Emotion 是一个基于脑电图（EEG）数据的情感分析模块化工程，采用配置驱动的设计理念，支持多种机器学习和深度学习模型。该项目旨在为研究人员提供一个统一、可扩展的框架，用于快速构建和比较不同的 EEG 情感分析模型。

### 核心特性

- **统一的数据预处理接口**：支持多种数据格式和预处理策略
- **多模型支持**：支持 SVM、MLP、RF、XGBoost、LSTM、CNN、混合模型等
- **配置驱动**：所有实验通过 YAML 配置文件驱动，便于复现和比较
- **丰富的可视化**：混淆矩阵、训练曲线、UMAP 边界图等
- **多模型比较**：支持在统一框架下比较不同模型的性能
- **模块化设计**：清晰的代码结构，易于扩展和维护

## 项目结构

```
eeg_modular/
├── eeg_emotion/              # 主包目录
│   ├── config/               # 配置管理
│   │   ├── __init__.py
│   │   └── loader.py         # YAML/JSON 配置加载器
│   ├── dl/                   # 深度学习实现
│   │   ├── common.py         # 通用深度学习工具
│   │   └── torch/            # PyTorch 模型
│   │       ├── data.py       # 数据加载器
│   │       ├── losses.py     # 损失函数
│   │       ├── trainer.py    # 训练器
│   │       ├── lstm_ae.py    # LSTM 自编码器
│   │       ├── lstm_clf.py   # LSTM 分类器
│   │       └── cvae_model.py # CVAE 模型
│   ├── features/             # 特征提取
│   │   ├── csv_stats.py      # CSV 统计特征提取
│   │   └── sequence/         # 序列特征提取
│   │       ├── extract.py    # 特征提取
│   │       └── augment.py    # 数据增强
│   ├── models/               # 模型实现
│   │   ├── base.py           # 基础模型接口
│   │   ├── sklearn/          # sklearn 模型
│   │   │   ├── svm.py        # SVM 模型
│   │   │   ├── mlp.py        # MLP 模型
│   │   │   ├── rf.py         # 随机森林模型
│   │   │   ├── xgb.py        # XGBoost 模型
│   │   │   └── hybrid.py     # 混合模型
│   │   └── torch/            # PyTorch 模型
│   │       ├── lstm_ae.py
│   │       ├── lstm_clf.py
│   │       └── cvae_model.py
│   ├── preprocess/           # 预处理
│   │   ├── tabular.py        # 表格数据预处理
│   │   ├── sequence.py       # 序列数据预处理
│   │   └── pipeline.py       # 预处理管道
│   ├── report/               # 报告生成
│   │   └── runs.py           # 运行报告
│   ├── train/                # 训练相关
│   │   ├── metrics.py        # 评估指标
│   │   └── weights.py        # 权重管理
│   ├── utils/                # 工具函数
│   │   ├── logging.py        # 日志工具
│   │   ├── paths.py          # 路径管理
│   │   └── seed.py           # 随机种子管理
│   └── viz/                  # 可视化
│       ├── confusion_matrix.py    # 混淆矩阵
│       ├── seaborn_cm.py          # Seaborn 风格混淆矩阵
│       ├── training_curves.py     # 训练曲线
│       └── umap_boundary.py       # UMAP 边界图
├── configs/                  # 配置文件目录
│   ├── svm.yaml             # SVM 模型配置
│   ├── mlp.yaml             # MLP 模型配置
│   ├── rf.yaml              # 随机森林配置
│   ├── xgb.yaml             # XGBoost 配置
│   ├── hybrid.yaml          # 混合模型配置
│   ├── lstm.yaml            # LSTM 模型配置
│   └── cnn.yaml             # CNN 模型配置
├── data/                     # 数据目录
│   ├── happy/               # 快乐情感数据
│   ├── sad/                 # 悲伤情感数据
│   └── normal/              # 正常情感数据
├── features/                 # 预提取的特征文件
├── outputs/                  # 输出目录（包含各模型的实验结果）
├── scripts/                  # 训练脚本
│   ├── train.py             # 通用训练脚本（sklearn 模型）
│   ├── train_svm_modular.py # SVM 训练脚本
│   ├── train_lstm_dl.py     # LSTM 训练脚本
│   ├── train_cnn_dl.py      # CNN 训练脚本
│   ├── compare_runs.py      # 运行比较脚本
│   └── multi_model_umap.py  # 多模型 UMAP 可视化
├── requirements.txt          # Python 依赖
└── README.md                # 项目文档

```

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.x（可选，用于 GPU 加速）
- Windows/Linux/macOS

### 依赖安装

```bash
# 安装基础依赖
pip install numpy scikit-learn joblib matplotlib pandas scipy

# 安装深度学习依赖
pip install torch torchaudio torchvision

# 安装其他依赖
pip install xgboost seaborn umap-learn pyyaml
```

### 验证安装

```bash
python -c "import torch; print(torch.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
```

## 核心命令

### 训练模型

#### 1. 训练 sklearn 模型（SVM、MLP、RF、XGBoost、混合模型）

```bash
# 训练 SVM 模型
python -m scripts.train -c configs/svm.yaml

# 训练 MLP 模型
python -m scripts.train -c configs/mlp.yaml

# 训练随机森林模型
python -m scripts.train -c configs/rf.yaml

# 训练 XGBoost 模型
python -m scripts.train -c configs/xgb.yaml

# 训练混合模型
python -m scripts.train -c configs/hybrid.yaml
```

#### 2. 训练深度学习模型

```bash
# 训练 LSTM 模型
python -m scripts.train_lstm_dl -c configs/lstm.yaml

# 训练 CNN 模型
python -m scripts.train_cnn_dl -c configs/cnn.yaml
```

### 多模型比较

```bash
# 生成多模型 UMAP 边界图
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml configs/xgb.yaml -o outputs/multi_model_umap
```

### 运行比较

```bash
# 比较不同运行的实验结果
python scripts/compare_runs.py outputs/run1 outputs/run2
```

## 配置文件说明

### 配置文件结构

所有配置文件采用 YAML 格式，包含以下主要部分：

```yaml
# 基础配置
seed: 42                          # 随机种子
data_dir: ./data                  # 数据目录
emotions: [happy, sad, normal]    # 情感类别

# CSV 文件配置（可选）
csv_files: [att.csv, med.csv, powerspec.csv]

# 数据集划分配置
split:
  test_size: 0.30                 # 测试集比例
  val_size: 0.10                  # 验证集比例
  random_state: 42                # 划分随机种子

# 预处理配置
preprocess:
  impute_strategy: mean           # 缺失值填充策略
  scale: true                     # 是否归一化
  select_k_best: null             # 特征选择（K值）
  pca_n_components: null          # PCA 降维维度
  augment: true                   # 是否数据增强
  noise_std: 0.01                 # 噪声标准差
  time_jitter: 0.02               # 时间抖动参数

# 模型配置
model:
  type: svm                       # 模型类型
  cv: 5                           # 交叉验证折数
  n_jobs: -1                      # 并行任务数
  probability: true               # 是否输出概率
  param_grid:                     # 超参数网格
    kernel: [rbf, poly, sigmoid, linear]
    C: [0.1, 1, 10]
    gamma: [scale, auto]
    class_weight: [null, balanced]

# 输出配置
output:
  base_dir: outputs               # 输出基础目录
  run_name: svm_baseline          # 运行名称（可选）

# 可视化配置
viz:
  seaborn_confusion_matrix: true  # 使用 seaborn 风格混淆矩阵
  umap_boundary: true             # 生成 UMAP 边界图
  training_curves: true           # 生成训练曲线
```

### 混合模型配置

混合模型集成了多个基模型，支持 soft voting 和 stacking：

```yaml
model:
  type: hybrid
  voting_method: soft             # voting 方法：soft 或 hard
  use_stacking: true              # 是否使用 stacking
  
  # MLP 配置
  mlp_config:
    cv: 5
    n_jobs: -1
    param_grid:
      hidden_layer_sizes: [(100,), (100, 50)]
      activation: [relu, tanh]
  
  # 随机森林配置
  rf_config:
    cv: 5
    n_jobs: -1
    param_grid:
      n_estimators: [100, 200]
      max_depth: [None, 10, 20]
  
  # SVM 配置
  svm_config:
    cv: 5
    n_jobs: -1
    probability: true
    param_grid:
      kernel: [rbf, poly, sigmoid]
      C: [0.1, 1, 10]
  
  # XGBoost 配置
  xgb_config:
    cv: 5
    n_jobs: -1
    param_grid:
      n_estimators: [100, 200]
      learning_rate: [0.01, 0.1]
      max_depth: [3, 6]
```

### LSTM 模型配置

```yaml
# LSTM 特有配置
time_steps: 128                  # 时间步数
min_cols_per_file: 10            # 每个文件最小列数

# 自编码器配置
autoencoder:
  epochs: 100
  batch_size: 32
  lr: 0.001
  latent_dim: 128
  enc_units: 128
  enc_layers: 2
  enc_dropout: 0.25
  use_bidirectional_decoder: true

# 分类器配置
classifier:
  epochs: 200
  batch_size: 16
  mode: bilstm                   # bilstm 或 mlp
  lstm_units: 128
  num_layers: 2
  dropout: 0.35
  pooling: avgmax

# 数据增强配置
augment:
  enabled: true
  sad_times: 3
  other_times: 3
  noise:
    mean: 0.0
    std: 0.01

# Mixup 配置
mixup:
  enabled: true
  augment_ratio: 1.0
  alpha: 0.3
```

## 输出目录结构

每次训练运行会在 `outputs/` 目录下创建一个时间戳命名的子目录：

```
outputs/
└── run_20240316_143022/          # 时间戳命名的运行目录
    ├── models/                   # 模型文件
    │   ├── model.joblib          # sklearn 模型
    │   ├── best_autoencoder.pt   # LSTM 自编码器
    │   └── best_classifier.pt    # LSTM 分类器
    ├── artifacts/                # 预处理器等
    │   ├── scaler.joblib         # 归一化器
    │   ├── label_encoder.joblib  # 标签编码器
    │   └── onehot_encoder.joblib # One-Hot 编码器
    ├── figures/                  # 可视化图表
    │   ├── confusion_matrix.png  # 混淆矩阵
    │   ├── training_curves.png   # 训练曲线
    │   └── umap_boundary.png     # UMAP 边界图
    ├── logs/                     # 日志文件
    │   └── train.log             # 训练日志
    └── metrics.json              # 评估指标
```

## 核心模块说明

### 1. 配置模块（config）

- **loader.py**：配置文件加载器，支持 YAML 和 JSON 格式
  - `load_config(path)`：加载配置文件
  - `require(cfg, key, type)`：获取必需的配置项
  - `get(cfg, key, default)`：获取可选配置项

### 2. 特征模块（features）

- **csv_stats.py**：从 CSV 文件提取统计特征
  - `build_tabular_dataset()`：构建表格数据集
  - `extract_features_from_df()`：从 DataFrame 提取特征

- **sequence/extract.py**：序列特征提取
  - `extract_all_features()`：提取所有序列特征

- **sequence/augment.py**：数据增强
  - `augment_class_samples()`：类别样本增强
  - `mixup_augment()`：Mixup 数据增强
  - `apply_gaussian_noise_batch()`：高斯噪声增强

### 3. 模型模块（models）

- **base.py**：模型基类接口
  - `ModelAdapter`：模型适配器基类
  - 定义了 `fit()`, `predict()`, `save()`, `load()` 等标准接口

- **sklearn/**：sklearn 模型实现
  - `svm.py`：SVM 模型（支持 SVC 和 LinearSVC）
  - `mlp.py`：多层感知机模型
  - `rf.py`：随机森林模型
  - `xgb.py`：XGBoost 模型
  - `hybrid.py`：混合模型（Voting + Stacking）

- **torch/**：PyTorch 模型实现
  - `lstm_ae.py`：LSTM 自编码器
  - `lstm_clf.py`：LSTM 分类器
  - `cvae_model.py`：条件变分自编码器

### 4. 预处理模块（preprocess）

- **tabular.py**：表格数据预处理
  - `TabularPreprocessor`：支持缺失值填充、归一化、特征选择、PCA 降维

- **sequence.py**：序列数据预处理
  - `SequencePreprocessor`：序列数据预处理

### 5. 可视化模块（viz）

- **confusion_matrix.py**：混淆矩阵生成
- **seaborn_cm.py**：Seaborn 风格混淆矩阵
- **training_curves.py**：训练曲线绘制
- **umap_boundary.py**：UMAP 边界图生成

### 6. 工具模块（utils）

- **logging.py**：日志工具
- **paths.py**：路径管理
- **seed.py**：随机种子管理

## 可视化功能

### 1. 混淆矩阵

支持两种风格：
- **Matplotlib 风格**：简洁的混淆矩阵
- **Seaborn 风格**：带有热力图效果的混淆矩阵

配置：
```yaml
viz:
  seaborn_confusion_matrix: true  # true: seaborn, false: matplotlib
```

### 2. 训练曲线

绘制训练和验证的准确率/损失曲线：

```yaml
viz:
  training_curves: true
```

### 3. UMAP 边界图

生成 UMAP 降维后的决策边界图：

```yaml
viz:
  umap_boundary: true
```

### 4. 多模型 UMAP 比较

在单个 UMAP 投影上绘制多个模型的决策边界：

```bash
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml -o outputs/multi_model_umap
```

## 最佳实践

### 1. 配置文件管理

- 每个实验使用独立的配置文件
- 使用清晰的命名规则（如 `svm_rbf_c1.0.yaml`）
- 避免使用绝对路径
- 保持配置简洁和可读性

### 2. 实验管理

- 所有实验输出到 `outputs/` 目录
- 使用时间戳自动命名运行目录
- 记录配置文件路径以便复现
- 定期清理过期的实验结果

### 3. 模型选择

- **SVM**：适合中小规模数据集，支持多种核函数
- **MLP**：适合非线性分类问题，可调参数较多
- **RF**：适合高维数据，抗过拟合能力强
- **XGBoost**：性能强劲，支持 GPU 加速
- **LSTM**：适合序列数据，能捕捉时间依赖
- **混合模型**：集成多个模型，通常性能最佳

### 4. 超参数调优

- 使用交叉验证（cv=5 或 cv=10）
- 从小范围网格搜索开始
- 优先调整重要参数（如 C、gamma、learning_rate）
- 考虑使用随机搜索加速调优

### 5. 数据预处理

- 严格区分训练集和测试集的预处理
- 只在训练集上拟合预处理模型
- 谨慎使用数据增强
- 检查类别分布是否平衡

## 常见问题

### 1. 数据泄露问题

**问题**：预处理过程中使用了测试集数据

**解决方案**：
```python
# 正确做法：只在训练集上拟合
X_train_t = pp.fit_transform(X_train)
X_test_t = pp.transform(X_test)  # 不要使用 fit_transform
```

### 2. 模型过拟合

**问题**：训练集表现好，测试集表现差

**解决方案**：
- 增加数据量或使用数据增强
- 降低模型复杂度
- 使用正则化
- 增加验证集比例
- 使用早停

### 3. 结果不一致

**问题**：相同配置多次运行结果不同

**解决方案**：
- 确保随机种子设置正确
- 检查是否使用了随机数据增强
- 确保数据加载顺序一致

### 4. 内存不足

**问题**：训练时内存溢出

**解决方案**：
- 减小 batch_size
- 使用特征选择或 PCA 降维
- 减少训练样本数量
- 使用更小的模型

### 5. GPU 不可用

**问题**：PyTorch 无法使用 GPU

**解决方案**：
```python
# 检查 CUDA 是否可用
import torch
print(torch.cuda.is_available())

# 如果不可用，确保：
# 1. 安装了 CUDA 版本的 PyTorch
# 2. CUDA 驱动版本匹配
# 3. 显卡支持 CUDA
```

## 扩展指南

### 添加新模型

1. 在 `eeg_emotion/models/` 下创建新模块
2. 继承 `ModelAdapter` 基类
3. 实现 `fit()`, `predict()`, `save()`, `load()` 方法
4. 在 `scripts/train.py` 中注册新模型

### 添加新特征

1. 在 `eeg_emotion/features/` 下创建新模块
2. 实现特征提取函数
3. 在数据加载流程中集成

### 添加新可视化

1. 在 `eeg_emotion/viz/` 下创建新模块
2. 实现可视化函数
3. 在配置文件中添加开关
4. 在训练脚本中调用

## 性能优化

### 1. 并行化

- sklearn 模型：设置 `n_jobs=-1` 使用所有 CPU 核心
- XGBoost：设置 `n_jobs=-1` 并启用 GPU（`device: cuda`）
- PyTorch：使用 DataParallel 或 DistributedDataParallel

### 2. 混合精度训练

```yaml
train:
  use_amp: true  # 启用自动混合精度
```

### 3. 数据加载优化

- 使用 `pin_memory=True` 加速 GPU 数据传输
- 使用 `num_workers` 并行加载数据
- 预加载和缓存特征

## 贡献指南

### 代码风格

- 遵循 PEP 8 规范
- 使用类型注解
- 添加文档字符串
- 编写单元测试

### 提交规范

- 每次提交只做一件事
- 提交信息清晰描述改动
- 不要提交 `outputs/` 目录
- 不要提交 `__pycache__/` 目录

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues：https://github.com/ThirteenAsh/eeg_modular/issues
- 项目文档：README.md

## 更新日志

### 当前版本

- 支持多种机器学习和深度学习模型
- 配置驱动的训练流程
- 丰富的可视化功能
- 多模型比较功能

### 计划功能

- [ ] 支持更多深度学习模型（Transformer、GAN 等）
- [ ] 自动超参数优化
- [ ] 分布式训练支持
- [ ] Web 界面
- [ ] 模型解释性分析