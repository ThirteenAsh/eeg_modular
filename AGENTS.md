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

### 研究目标

本项目旨在探索基于单通道消费级脑电设备的情绪识别方法，主要研究价值包括：

1. **医疗健康应用**：抑郁症早期筛查、焦虑症日常监测、压力水平评估
2. **认知增强**：学习专注度监测、认知负荷评估、个性化学习推荐
3. **人机交互优化**：情感计算接口、智能助手情绪感知、虚拟现实情绪反馈
4. **便携式解决方案**：消费级设备价格仅为实验室设备的 1-5%
5. **实时应用**：模型推理时间约 8ms（GPU），满足<100ms 实时性要求

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
- 使用模型名称自动命名运行目录
- 记录配置文件路径以便复现
- 定期清理过期的实验结果

### 3. 模型选择

| 模型 | 适用场景 | 优势 | 局限 | 计算资源 | 推荐场景 |
|------|----------|------|------|----------|----------|
| **CNN** | 时序数据、特征自动学习 | 自动特征提取、层次化学习 | 需要大量数据 | 高（GPU） | 准确率要求高、有GPU |
| **RF** | 高维数据、非线性分类 | 抗过拟合强、可解释性好 | 边界碎片化 | 中等 | 快速原型、特征重要性分析 |
| **SVM** | 小样本、非线性分类 | 泛化能力强、理论完善 | 参数敏感 | 低 | 小样本、理论分析 |
| **XGBoost** | 结构化数据、性能优先 | 性能强劲、支持GPU | 易过拟合 | 中等（可GPU） | 竞赛、高性能需求 |
| **LSTM** | 长序列、时间依赖 | 捕捉长期依赖 | 训练慢 | 高（GPU） | 序列数据、有长期依赖 |
| **MLP** | 非线性分类 | 简单易用 | 难以处理序列 | 低 | 快速实验、基准测试 |
| **HYBRID** | 多模型集成 | 综合优势、稳定性好 | 复杂度高 | 高 | 最终部署、性能优化 |

### 4. 超参数调优

- 使用交叉验证（cv=5 或 cv=10）
- 从小范围网格搜索开始
- 优先调整重要参数（如 C、gamma、learning_rate）
- 考虑使用随机搜索加速调优

**CNN 超参数调优指南**：
```python
# 学习率优先级：最重要
lr: [0.0001, 0.0005, 0.001, 0.005]  # 最优：0.001

# 批次大小
batch_size: [64, 128, 256, 512]  # 最优：256

# 权重衰减
weight_decay: [0, 0.0001, 0.001, 0.01]  # 最优：0.001

# 网络深度
num_layers: [2, 3, 4]  # 最优：3

# 滤波器数量
filters: [64, 128, 256]  # 默认配置
```

**RF 超参数调优指南**：
```python
# 树的数量
n_estimators: [300, 500, 800, 1000]  # 最优：1000

# 最大深度
max_depth: [10, 20, 30, 40, 50]  # 最优：30

# 特征选择
max_features: [sqrt, log2]  # 默认：sqrt
```

**SVM 超参数调优指南**：
```python
# 正则化参数
C: [0.1, 1, 10, 50, 100]  # 最优：10

# 核系数
gamma: [0.001, 0.01, 0.1, 1]  # 最优：scale

# 核函数
kernel: [rbf]  # RBF 最优

# 类别权重
class_weight: [balanced]  # 处理类别不平衡
```

### 5. 数据预处理

- 严格区分训练集和测试集的预处理
- 只在训练集上拟合预处理模型
- 谨慎使用数据增强
- 检查类别分布是否平衡

**数据增强策略**：
```python
# 信号模态增强（4倍）
- 时间抖动：时间轴随机偏移，偏移量=信号长度×0.03
- 高斯噪声：噪声标准差=信号标准差×0.02
- Mixup：线性插值混合两个样本，α=0.4

# 标量模态增强（2倍）
- 微小噪声：噪声标准差=0.001，避免破坏语义

# 增强效果
- 无增强：83.9% → 有增强：88.9%（+5.0%）
- 时间抖动贡献：+2.1%
- 高斯噪声贡献：+1.8%
- Mixup贡献：+2.5%
```

### 6. 模型评估

- 使用多种评估指标：Accuracy、Precision、Recall、F1-score
- 生成混淆矩阵分析误分类模式
- 使用UMAP降维可视化特征空间
- 进行交叉验证确保结果可靠性

**评估指标解读**：
```python
# Accuracy（准确率）
# 定义：正确分类样本占总样本的比例
# 适用：类别平衡的数据集
# 局限：不适用于类别不平衡场景

# Precision（精确率）
# 定义：预测为正的样本中实际为正的比例
# 适用：关注误报（假阳性）的场景
# 示例：医疗诊断，避免误诊

# Recall（召回率）
# 定义：实际为正的样本中被正确预测的比例
# 适用：关注漏报（假阴性）的场景
# 示例：疾病筛查，避免漏诊

# F1-score（F1分数）
# 定义：精确率和召回率的调和平均
# 适用：需要平衡精确率和召回率的场景
# 优势：不受类别不平衡影响
```

### 7. 实验可复现性

- 设置随机种子（seed=42）
- 记录所有配置参数
- 保存训练日志
- 使用版本控制（Git）

**随机种子设置**：
```python
import random
import numpy as np
import torch

# Python随机种子
random.seed(42)

# NumPy随机种子
np.random.seed(42)

# PyTorch随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

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

**示例配置**：
```yaml
# sklearn 模型并行化
model:
  n_jobs: -1  # 使用所有CPU核心
  cv: 5  # 5折交叉验证

# XGBoost GPU加速
model:
  param_grid:
    tree_method: hist  # 直方图加速
    device: cuda  # 使用GPU
    n_jobs: -1  # 多线程
```

### 2. 混合精度训练

```yaml
train:
  use_amp: true  # 启用自动混合精度
```

**混合精度优势**：
- 训练速度提升约 2-3 倍
- 显存占用减少约 50%
- 模型精度基本不变
- 适用于 CNN、LSTM 等深度学习模型

### 3. 数据加载优化

- 使用 `pin_memory=True` 加速 GPU 数据传输
- 使用 `num_workers` 并行加载数据
- 预加载和缓存特征

**PyTorch 数据加载优化**：
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,  # 并行加载数据
    pin_memory=True,  # 加速GPU传输
    prefetch_factor=2  # 预取数据
)
```

### 4. 模型轻量化

**剪枝（Pruning）**：
```python
import torch.nn.utils.prune as prune

# 剪枝30%的权重
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

**量化（Quantization）**：
```python
# 训练后量化
import torch.quantization as quant

# 动态量化
model_quantized = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**知识蒸馏（Knowledge Distillation）**：
```python
# 教师模型（大模型）
teacher_model = load_large_model()

# 学生模型（小模型）
student_model = SmallModel()

# 蒸馏损失
def distillation_loss(student_output, teacher_output, labels, T=2.0, alpha=0.5):
    # 软标签损失
    soft_loss = nn.KLDivLoss()(
        F.log_softmax(student_output/T, dim=1),
        F.softmax(teacher_output/T, dim=1)
    )
    # 硬标签损失
    hard_loss = nn.CrossEntropyLoss()(student_output, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 5. 推理加速

**ONNX 导出**：
```python
import torch.onnx

# 导出为ONNX格式
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

**TensorRT 优化**：
```python
import tensorrt as trt

# 构建TensorRT引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)

# 解析ONNX模型
with open("model.onnx", 'rb') as model:
    parser.parse(model.read())

# 构建引擎
engine = builder.build_cuda_engine(network)
```

### 6. 边缘设备部署

**Jetson Nano 部署**：
```bash
# 安装TensorRT
sudo apt-get install tensorrt

# 转换模型
trtexec --onnx=model.onnx --saveEngine=model.trt

# Python推理
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 加载TensorRT引擎
with open("model.trt", "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
```

**树莓派部署**：
```bash
# 安装轻量级模型
pip install tflite-runtime

# 转换为TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存模型
with open("model.tflite", 'wb') as f:
    f.write(tflite_model)
```

## 高级功能

### 1. 自动超参数优化

**使用 Optuna**：
```python
import optuna

def objective(trial):
    # 定义搜索空间
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    
    # 训练模型
    model = train_model(lr=lr, batch_size=batch_size, hidden_dim=hidden_dim)
    
    # 返回目标值（验证准确率）
    return model.val_accuracy

# 创建研究
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 获取最佳参数
best_params = study.best_params
print(f"Best params: {best_params}")
```

### 2. 模型解释性分析

**LIME**：
```python
import lime.lime_tabular

# 创建LIME解释器
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['happy', 'sad', 'normal'],
    discretize_continuous=True
)

# 解释单个预测
exp = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=10
)

# 可视化解释
exp.show_in_notebook()
```

**SHAP**：
```python
import shap

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 可视化特征重要性
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### 3. 联邦学习

**使用 PySyft**：
```python
import torch
import syft as sy

# 创建虚拟工作者
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# 发送数据到工作者
data_alice = data[:50].send(alice)
data_bob = data[50:].send(bob)
target_alice = target[:50].send(alice)
target_bob = target[50:].send(bob)

# 联邦训练
for epoch in range(10):
    # 在Alice上训练
    model.send(alice)
    optimizer.zero_grad()
    pred = model(data_alice)
    loss = loss_fn(pred, target_alice)
    loss.backward()
    optimizer.step()
    model.get()
    
    # 在Bob上训练
    model.send(bob)
    optimizer.zero_grad()
    pred = model(data_bob)
    loss = loss_fn(pred, target_bob)
    loss.backward()
    optimizer.step()
    model.get()
```

### 4. 实时推理服务

**Flask API**：
```python
from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# 加载模型
model = torch.load('model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据
    data = request.json['data']
    x = np.array(data, dtype=np.float32)
    x_tensor = torch.from_numpy(x)
    
    # 推理
    with torch.no_grad():
        output = model(x_tensor)
        pred = torch.argmax(output, dim=1)
    
    # 返回结果
    return jsonify({'prediction': pred.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**WebSocket 实时流处理**：
```python
import asyncio
import websockets
import json
import torch

model = torch.load('model.pth')
model.eval()

async def handle_connection(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        x = np.array(data['signal'], dtype=np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        
        with torch.no_grad():
            output = model(x_tensor)
            pred = torch.argmax(output, dim=1)
        
        await websocket.send(json.dumps({
            'emotion': ['happy', 'sad', 'normal'][pred.item()],
            'confidence': torch.softmax(output, dim=1).tolist()[0]
        }))

start_server = websockets.serve(handle_connection, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

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
- 主项目文档：../AGENTS.md

## 项目版本历史

### v1.0.0 (2026-03-17) - 模型优化完成

**新功能**：
- ✅ 7个模型全部训练完成
- ✅ 6个模型达到60%+准确率
- ✅ CNN模型达到88.89%准确率（最优）
- ✅ 多模型UMAP可视化
- ✅ 数据增强策略优化

**性能提升**：
- SVM: 58.02% → 62.96% (+4.94%)
- LSTM: 59.26% → 60.49% (+1.23%)
- XGB: 58.02% → 60.49% (+2.47%)
- HYBRID: 59.26% → 64.20% (+4.94%)
- 数据增强: +5.0% 整体提升

**文档更新**：
- 更新所有模型配置文件
- 生成完整学术论文（14,000字）
- 验证60篇参考文献
- 创建文献获取指南

### v0.9.0 (2026-03-15) - 深度学习模型优化

**新功能**：
- CNN模型训练脚本优化
- LSTM模型架构优化
- 自动混合精度训练

**性能提升**：
- CNN: 初始训练完成
- LSTM: 性能优化至60.49%

### v0.8.0 (2026-03-10) - 机器学习模型优化

**新功能**：
- SVM、MLP、RF、XGBoost模型优化
- 超参数网格搜索
- 混合模型集成

**性能提升**：
- RF: 达到67.90%准确率
- SVM: 达到62.96%准确率
- HYBRID: 达到64.20%准确率

### v0.7.0 (2026-03-05) - 数据预处理优化

**新功能**：
- 数据增强策略优化
- 特征标准化流程
- 数据质量检查

**性能提升**：
- 数据增强: +5.0% 准确率提升

### v0.6.0 (2026-02-28) - 基础功能实现

**新功能**：
- 数据预处理流程
- 基础模型训练
- 可视化功能
- 配置文件系统

## 更新日志

### 2026-03-17
- ✅ 完成所有模型优化
- ✅ 生成完整学术论文
- ✅ 验证所有参考文献
- ✅ 更新项目文档

### 2026-03-15
- ✅ CNN模型训练完成
- ✅ LSTM模型优化完成
- ✅ 添加自动混合精度训练

### 2026-03-10
- ✅ RF模型优化完成
- ✅ SVM模型优化完成
- ✅ HYBRID模型优化完成

### 2026-03-05
- ✅ 数据增强策略优化
- ✅ 特征标准化流程优化
- ✅ 数据质量检查添加

### 2026-02-28
- ✅ 基础功能实现
- ✅ 数据预处理流程
- ✅ 可视化功能添加
- ✅ 配置文件系统建立

## 已知问题

1. **MLP 模型性能不足**
   - 当前准确率：59.26%
   - 目标准确率：60%
   - 原因：网络架构可能不适合该数据集
   - 解决方案：尝试更深的网络或不同的激活函数

2. **图片路径警告**
   - 警告：`multi_model_umap_boundary.png` 未找到
   - 影响：Word文档中图片显示为文本描述
   - 解决方案：重新生成多模型UMAP可视化

3. **个体差异未充分考虑**
   - 当前模型为跨被试者模型
   - 个体差异影响性能
   - 解决方案：未来可探索个性化校准策略

## 未来计划

### v2.0.0 - 计划中

**新增功能**：
- [ ] 增加更多模型（Transformer、GAN、ResNet）
- [ ] 实现个性化情绪识别
- [ ] 添加实时推理接口（REST API）
- [ ] 开发Web界面（Django/Flask）
- [ ] 集成自动超参数优化（Optuna）

**性能优化**：
- [ ] 模型轻量化（剪枝、量化）
- [ ] 推理加速（TensorRT、ONNX Runtime）
- [ ] 边缘设备部署支持
- [ ] 移动端优化

### v3.0.0 - 长期计划

**新功能**：
- [ ] 多模态生理信号融合（ECG、GSR、EMG）
- [ ] 边缘计算部署（Jetson Nano、树莓派）
- [ ] 移动端应用开发（Android/iOS）
- [ ] 云端服务部署（AWS、Azure）
- [ ] 实时流处理（WebSocket）

**高级功能**：
- [ ] 联邦学习支持
- [ ] 差分隐私保护
- [ ] 模型解释性分析（LIME、SHAP）
- [ ] 自动化实验管理（MLflow）
- [ ] 持续集成/持续部署（CI/CD）

---

## 模型性能总结

### 最终模型准确率（2026-03-17）

| 模型 | 准确率 | 状态 | 最佳配置 | 训练时间 | 推理时间 |
|------|--------|------|----------|----------|----------|
| **CNN** | 88.89% | ✅ 最优 | epochs=200, batch_size=256, lr=0.001 | ~15min | ~8ms (GPU) |
| **RF** | 67.90% | ✅ 已达标 | n_estimators=1000, max_depth=30 | ~5min | ~2ms (CPU) |
| **HYBRID** | 64.20% | ✅ 已达标 | Soft Voting, 100 features | ~10min | ~5ms (CPU) |
| **SVM** | 62.96% | ✅ 已达标 | kernel=rbf, C=10, gamma=scale | ~3min | ~1ms (CPU) |
| **XGB** | 60.49% | ✅ 已达标 | n_estimators=800, max_depth=5 | ~8min | ~3ms (CPU) |
| **LSTM** | 60.49% | ✅ 已达标 | BiLSTM, 512 units, 4 layers | ~20min | ~10ms (GPU) |
| **MLP** | 59.26% | ⚠️ 接近目标 | [1024, 512, 256] architecture | ~6min | ~1ms (CPU) |

### 数据集信息
- **测试集大小**：81 个样本（20% 测试集）
- **训练集大小**：1296 个样本（80% 训练集，包含数据增强）
- **情感类别**：happy, sad, normal（各 27 个测试样本）
- **特征模态**：filtered, powerspec, att, med（4 种模态）
- **数据增强**：信号模态 4 倍增强，标量模态 2 倍增强
- **特征维度**：(batch_size, 40, 160)
  - 40 = 4种模态 × 10个时间步
  - 160 = 10个时间步 × 16个特征/时间步

### 各类别详细性能（CNN模型）

| 类别 | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Happy | 95.8% | 85.2% | 90.2% | 27 |
| Sad | 78.1% | 92.6% | 84.7% | 27 |
| Normal | 96.0% | 88.9% | 92.3% | 27 |
| **Macro Avg** | **89.9%** | **88.9%** | **89.1%** | **81** |

### 优化策略总结

- **SVM**：增加特征数量（50 → 100），扩展参数搜索范围
- **LSTM**：增加网络容量（512 units, 4 layers），降低学习率（0.001 → 0.0005），增加训练轮数
- **XGB**：增加特征数量（50 → 100），优化关键参数
- **HYBRID**：增加特征数量（50 → 100），优化各基模型参数，使用 Soft Voting
- **MLP**：尝试多种特征数量和网络架构，最佳结果为 59.26%
- **RF**：使用深度决策树（max_depth=30），大量估计器（n_estimators=1000）
- **CNN**：5折交叉验证，自动混合精度训练，权重衰减

### 实验环境

**硬件配置**：
- CPU：Intel Core i7（8核）
- GPU：NVIDIA RTX 3060（12GB 显存）
- 内存：32GB DDR4
- 存储：SSD 500GB

**软件环境**：
- 操作系统：Windows 11
- Python：3.12.5
- PyTorch：2.0+
- scikit-learn：1.3+
- XGBoost：2.0+
- CUDA：11.x

### 数据增强效果

| 条件 | 准确率 | Macro F1 | 提升幅度 |
|------|--------|----------|----------|
| 无增强 | 83.9% | 84.2% | - |
| 有增强 | 88.9% | 89.1% | +5.0% |

**各增强策略贡献**：
- 时间抖动：+2.1%
- 高斯噪声：+1.8%
- Mixup：+2.5%
- 组合策略：+5.0%（协同效应）

---