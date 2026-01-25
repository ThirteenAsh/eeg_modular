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

### 1.2 项目结构

```
./
├── eeg_emotion/      # 主包目录
│   ├── config/       # 配置管理
│   ├── dl/           # 深度学习实现（TensorFlow和PyTorch）
│   ├── features/     # 特征提取
│   ├── models/       # 模型实现
│   │   ├── sklearn/  # sklearn模型
│   │   ├── tf/       # TensorFlow模型
│   │   └── torch/    # PyTorch模型
│   ├── preprocess/   # 预处理
│   ├── report/       # 报告生成
│   ├── train/        # 训练相关
│   ├── utils/        # 工具函数
│   └── viz/          # 可视化
├── configs/          # 配置文件目录
├── data/             # 数据目录
├── features/         # 特征相关文件
├── scripts/          # 训练脚本
└── outputs/          # 输出目录
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
model_type = require(cfg, "model.type", str)
seaborn_cm = get(cfg, "viz.seaborn_confusion_matrix", False)
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

**注意**：目前只支持sklearn模型（SVM、MLP、RF），CNN和LSTM等深度学习模型需要不同的处理方式，暂不支持。

**使用方法**：

```bash
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml -o outputs/multi_model_umap
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

### 4.4 LSTM模型训练

**配置文件**：`configs/lstm.yaml`

**运行命令**：
```bash
python -m scripts.train_lstm_dl -c configs/lstm.yaml
```

### 4.5 CNN模型训练

**配置文件**：`configs/cnn.yaml`

**运行命令**：
```bash
python -m scripts.train_cnn_dl -c configs/cnn.yaml
```

### 4.6 多模型UMAP比较

**运行命令**：
```bash
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml configs/xgb.yaml -o outputs/multi_model_umap
```

### 4.7XGBoost 接入说明

本补丁完成：

- scripts/train.py：新增 model.type = xgboost / xgb 分支（支持 GridSearchCV）
- configs/xgb.yaml：提供同口径配置模板

##### 运行

```bash
python -m scripts.train -c configs/xgb.yaml
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

## 5. 可视化选择

### 5.1 如何选择可视化风格

#### 1. 配置文件设置

在YAML配置文件中添加可视化风格选择参数：

```yaml
# configs/svm.yaml
viz:
  seaborn_confusion_matrix: true  # true: seaborn风格, false: matplotlib风格
  umap_boundary: true              # 是否生成UMAP边界图
  training_curves: true            # 是否生成训练曲线
```

#### 2. 训练脚本支持

所有训练脚本都支持根据配置文件选择可视化风格：

- `scripts/train.py`：支持sklearn模型（SVM、MLP、RF等）
- `scripts/train_lstm_dl.py`：支持LSTM模型
- `scripts/train_cnn_dl.py`：支持CNN模型
- `scripts/train_svm_modular.py`：支持SVM模型

### 5.2 训练曲线实现

训练曲线绘制目前在LSTM/tensorflow路线中实现，其他路线可参考扩展。

**核心实现**：

```python
def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    # 绘制准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

## 6. UMAP边界图生成

### 6.1 UMAP边界图问题分析

#### 1. 为什么UMAP没有生成？

- UMAP绘制函数已实现：`save_umap_svm_decision_boundary`
- 但训练脚本没有调用它
- 缺少配置选项处理
- 缺少依赖检查

#### 2. 解决方案

修改训练脚本以启用UMAP边界绘制：

```python
# 从配置文件读取UMAP设置
generate_umap = get(cfg, "viz.umap_boundary", False)

# 绘制UMAP边界图（如果配置启用）
if generate_umap:
    try:
        save_umap_svm_decision_boundary(
            X=X_test_t,
            y=y_test,
            class_names=emotions,
            save_path=os.path.join(run.figures_dir, "umap_boundary.png"),
            title="UMAP Projection with Decision Boundary (Test Set)",
        )
        logger.info("✅ UMAP boundary plot saved")
    except ImportError:
        logger.warning("⚠️ umap-learn not installed, skipping UMAP boundary plot")
    except Exception as e:
        logger.error(f"❌ Failed to generate UMAP boundary: {e}")
```

### 6.2 多模型UMAP边界图

**功能说明**：在单个UMAP投影上绘制多个模型的决策边界，便于直观比较不同模型的决策边界差异。

**使用方法**：

```bash
python scripts/multi_model_umap.py -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml -o outputs/multi_model_umap
```

**实现原理**：
1. 生成单个UMAP投影（仅一次）
2. 对每个模型，使用其预测结果在UMAP空间中训练代理SVM
3. 在同一UMAP投影上绘制所有代理SVM的决策边界
4. 使用不同颜色和线型区分不同模型的边界

## 7. 模块详细说明

### 7.1 配置模块（config）

**核心功能**：
- 加载和解析YAML配置文件
- 提供配置验证和默认值
- 支持嵌套配置访问

**主要文件**：
- `config/loader.py`：配置加载和解析

### 7.2 数据模块（data）

**核心功能**：
- 数据加载和预处理
- 支持多种数据格式
- 提供数据增强功能

**主要文件**：
- `data/loader.py`：数据加载
- `data/augment.py`：数据增强

### 7.3 模型模块（models）

**核心功能**：
- 支持多种模型类型
- 统一的模型接口
- 支持模型保存和加载

**主要文件**：
- `models/sklearn/`：sklearn模型实现
- `models/tensorflow/`：tensorflow模型实现

### 7.4 可视化模块（viz）

**核心功能**：
- 混淆矩阵生成
- 训练曲线绘制
- UMAP边界图生成
- 多模型可视化比较

**主要文件**：
- `viz/confusion_matrix.py`：混淆矩阵
- `viz/training_curves.py`：训练曲线
- `viz/umap_boundary.py`：UMAP边界图

## 8. 最佳实践

### 8.1 配置文件设计

- 每个实验使用独立的配置文件
- 使用清晰的命名规则（如`svm_baseline.yaml`）
- 避免使用绝对路径
- 保持配置简洁和可读性

### 8.2 实验命名
典型配置字段：
- output.base_dir / output.run_name
- data_dir / emotions / csv_files
- preprocess.xxx
- model.xxx
- train.xxx
- viz.xxx（可选）

- 使用有意义的实验名称，包含关键参数
- 示例：`svm_rbf_c1.0_gamma_scale`

### 8.3 结果分析

- 关注多个指标（准确率、精确率、召回率、F1值）
- 分析混淆矩阵，了解模型在不同类别上的表现
- 比较不同模型和参数的表现

## 9. 常见问题与解决方案

### 9.1 数据泄露问题

**问题**：预处理过程中使用了测试集数据，导致模型泛化能力下降。

**解决方案**：
- 严格遵循预处理流程，只在训练集上拟合预处理模型
- 使用`fit_transform_train`方法处理训练集
- 使用`transform`方法处理验证集和测试集

### 9.2 模型过拟合问题

**问题**：模型在训练集上表现良好，但在测试集上表现较差。

**解决方案**：
- 增加数据量或使用数据增强
- 降低模型复杂度
- 使用正则化技术
- 增加验证集比例
- 早期停止

### 9.3 实验结果不一致

**问题**：相同配置文件，多次运行结果不一致。

**解决方案**：
- 确保随机种子设置正确
- 检查是否使用了随机数据增强
- 确保数据加载顺序一致

## 10. 总结

EEG Emotion模块化工程是一个设计精良、易于扩展的情感分析框架，采用配置驱动的设计理念，支持多种模型和训练方式。

**核心优势**：
- 统一的数据预处理接口
- 统一的模型接口
- 统一的训练入口
- 统一的输出目录与指标格式
- 统一的可视化产物
- 可扩展的设计

通过使用该工程，研究人员可以快速构建和比较不同的EEG情感分析模型，加速研究进程，提高实验结果的可靠性和可重复性。
- 在仓库根目录提供：
  - `ENGINEERINREAD_GUIDE.md`（本文）
- 约定所有实验 run 都输出到 `outputs/`（但不要把 outputs 提交到 git；用 .gitignore）
- 每次新增模型/功能：
  - 新增模块文件
  - 新增 yaml 示例
  - README/指南里补一段“如何运行”
