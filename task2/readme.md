# 模型公平性检测与提升实验

本项目旨在探索和实践机器学习模型中的公平性问题，重点关注如何检测和提升模型的公平性。实验代码使用 PyTorch 框架，并以 ProPublica COMPAS 数据集作为案例进行分析。

## 项目背景

随着机器学习模型在社会各个关键领域（如信贷审批、招聘、司法判决等）的应用日益广泛，模型的公平性问题受到了越来越多的关注。由于训练数据中可能存在的历史偏见或不平衡，模型可能会对特定群体产生不公平的决策，从而加剧社会不平等。本项目旨在通过实践，理解算法歧视的成因，并探索一些常用的模型公平性提升方法。

## 目标

* 理解机器学习中的公平性定义和度量指标。
* 熟悉几种常见的模型公平性提升技术，如预处理、在训练中优化和后处理方法。
* 实践如何在 PyTorch 中实现这些公平性技术。
* 使用 COMPAS 数据集，分析不同公平性干预措施对模型性能和公平性指标的影响。

## 文件夹结构

```

公平性实验/
├── utils.py             \# 包含辅助函数 (如公平性指标计算、数据处理工具)
├── main.py              \# 主执行脚本，包含数据加载、模型训练、公平性方法实现和评估
├── Processed\_Compas.csv \# COMPAS 数据集 (预处理版本)
├── results/             \# (可选) 存放实验结果，如图片、指标数据等
├── models/              \# (可选) 存放训练好的模型文件
└── README.md            \# 本项目说明文件

````

## 数据集

本项目使用 **ProPublica COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)** 数据集的预处理版本 (`Processed_Compas.csv`)。该数据集常用于公平性研究，其中包含被告的犯罪记录、人口统计信息以及再犯风险评分。实验的目标是预测刑事被告是否会在两年内再次犯罪。

**敏感属性示例**: `race`, `sex`
**相关属性示例**: `age_cat` (年龄段), `priors_count` (前科数量) 等。

## 环境要求

* Python (推荐版本 3.8 或更高)
* PyTorch (推荐版本 1.8 或更高，实验基于 `torch 2.2.2` 开发)
* NumPy
* Pandas
* Scikit-learn
* Fairlearn (用于公平性指标对比)
* Matplotlib (可选，用于结果可视化)
* Argparse

您可以使用 pip 安装所需库：
```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn fairlearn matplotlib argparse
````

## 代码说明

### `utils.py`

该文件包含一系列用于支持主实验流程的工具函数：

  * `groupTPR()`: 计算每个群体的真正例率 (True Positive Rate)。
  * `groupTNR()`: 计算每个群体的真负例率 (True Negative Rate)。 (注意：原始PDF中此函数名为TNR但实现逻辑接近FNR，代码中已尝试修正为TNR)
  * `counter_sample()`: 生成对抗性样本，通过修改特定属性来创建新的数据点，用于评估模型的反事实公平性。
  * `cal_correlation()`: 计算敏感属性与相关属性之间的相关性系数。

### `main.py`

该文件是实验的核心，主要包含以下部分：

  * **参数解析**: 使用 `argparse` 处理命令行参数，允许用户配置实验设置，如使用的数据集、敏感属性、相关属性、学习率、训练轮次、模型类型以及应用的公平性方法等。
  * **数据加载与预处理**:
      * 支持加载 Adult 和 COMPAS 数据集 (本项目重点是 COMPAS)。
      * 对数据进行清洗、独热编码 (One-Hot Encoding) 和标准化 (Standard Scaling)。
      * 划分训练集和测试集。
  * **模型定义**:
      * `Classifier`: 一个基于全连接层 (MLP) 的分类器。
      * `ClassifierLr`: 一个线性回归/逻辑回归分类器 (用于LR和SVM)。
      * `loss_svm`: SVM 的 Hinge Loss 实现。
  * **训练逻辑**:
      * `pretrain_classifier()`: 基础分类器的预训练函数。
      * `Perturb_train()`: 基于反事实公平思想的扰动训练方法。
      * `CorreErase_train()`: 旨在消除特征与模型预测之间相关性的训练方法。
      * `Gfair_train()`: 基于群体公平性（如TPR均等）的正则化训练方法。
      * `CorreLearn_train()`: **(待完成)** 一个旨在学习相关属性权重以优化公平性的训练方法。根据源文件说明，此函数的具体实现逻辑需要用户根据相关论文补充。
  * **公平性方法选择**:
      * `base`: 标准模型训练，不进行公平性干预。
      * `remove`: 预处理方法，直接移除指定的（相关）属性。
      * `counterfactual`: 对应 `Perturb_train`。
      * `corre`: 对应 `CorreErase_train`。
      * `groupTPR`: 对应 `Gfair_train`。
      * `learnCorre`: 对应 `CorreLearn_train`。
  * **评估与结果输出**:
      * 使用测试集评估模型性能（准确率）。
      * 使用 `fairlearn.metrics.MetricFrame` 计算不同群体的准确率。
      * 计算自定义的公平性指标，如不同群体的选择率 (Selection Rate) 和均等化赔率 (Equalized Odds) 相关指标 (如TPR差异)。

## 如何运行

1.  **准备环境**: 确保已安装所有必需的库 (见 [环境要求](https://www.google.com/search?q=%23%E7%8E%AF%E5%A2%83%E8%A6%81%E6%B1%82))。

2.  **准备数据**: 将 `Processed_Compas.csv` 文件放置在项目根目录下。

3.  **(重要)** **完成 `CorreLearn_train` 函数**: 打开 `main.py` 文件，找到 `CorreLearn_train` 函数。根据实验要求或相关论文[1]的指导，补充该函数的实现逻辑。

4.  **运行实验**:
    打开终端，进入项目根目录 (`公平性实验/`)，然后执行 `main.py` 脚本。您可以通过命令行参数指定不同的实验配置。

    **示例命令:**

      * 运行基础模型 (不进行公平性干预) 在 COMPAS 数据集上，以 `race` 为敏感属性：

        ```bash
        python main.py --dataset compas --s race --method base --model MLP
        ```

      * 尝试使用相关性消除方法，以 `race` 为敏感属性，`age_cat` 和 `sex` 为相关属性：

        ```bash
        python main.py --dataset compas --s race --related age_cat sex --r_weight 0.1 0.1 --method corre --model MLP
        ```

      * 尝试移除相关属性 `age_cat` (假设 `race` 是敏感属性)：

        ```bash
        python main.py --dataset compas --s race --related age_cat --method remove --model MLP
        ```

      * 运行 `groupTPR` 公平性方法：

        ```bash
        python main.py --dataset compas --s race --related age_cat sex --r_weight 0.1 0.1 --method groupTPR --model MLP
        ```

    **可配置参数 (通过 `python main.py --help` 查看完整列表):**

      * `--dataset`: 使用的数据集 (`adult`, `compas`, `pokec`, `law`)。
      * `--s`: 敏感属性的名称 (例如，`sex`, `race`)。
      * `--related`: 一个或多个相关属性的名称。
      * `--r_weight`: 对应于 `--related` 属性的权重列表。
      * `--method`: 应用的公平性方法 (`base`, `remove`, `counterfactual`, `corre`, `groupTPR`, `learnCorre`)。
      * `--model`: 使用的模型类型 (`MLP`, `LR`, `SVM`)。
      * `--epoch`: 训练轮次。
      * `--lr`: 学习率。
      * 等。

## 预期结果

脚本运行后，将在控制台输出以下信息：

  * 敏感属性和相关属性之间的相关性系数。
  * 模型在测试集上的平均准确率。
  * 按敏感属性分组的准确率。
  * 自定义的公平性指标，如不同群体的均等化赔率（TPR）差异和选择率差异。
  * 如果使用 `learnCorre` 方法（并在您完成实现后），还会输出学习到的 lambda 权重。

根据 "第12章 模型公平性检测与提升原理与实践 (12.2 模型公平性检测与提升-实践).pptx" 文件的描述，可以通过绘制热力图来可视化不同参数设置下准确率、EO (Equalized Odds) 和 SR (Selection Rate) 指标的变化，以分析模型公平性提升前后的对比。

## 参考文献

[1] Zhao T, Dai E, Shu K, et al. Towards fair classifiers without sensitive attributes: Exploring biases in related features[C]//Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022: 1433-1442. (或其他与 `CorreLearn_train` 方法相关的参考论文)

## 注意事项

  * 代码中的 `CorreLearn_train` 函数是不完整的，需要用户根据参考资料自行实现。
  * 数据集 `Processed_Compas.csv` 必须存在于正确的路径。
  * 某些公平性方法和指标的计算可能对参数设置和数据特性非常敏感。

## 未来工作 (可选)

  * 完整实现 `CorreLearn_train` 函数。
  * 扩展支持更多的数据集和公平性指标。
  * 添加更详细的结果可视化功能。
  * 对比更多前沿的公平性提升算法。