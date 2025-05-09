# 基于影子模型的成员推理攻击 (Shadow Model Based Membership Inference Attack)

本项目实现了基于影子模型的成员推理攻击，用于评估机器学习模型泄露其训练数据成员信息的风险。攻击者通过训练与目标模型相似的影子模型，并利用这些影子模型的输出来训练一个攻击模型，该攻击模型可以判断给定数据样本是否是目标模型训练集的一部分。

本代码基于 PyTorch 框架实现，并参考了相关研究中的标准攻击流程。

## 项目结构

```
membership-inference-attack/
├── data/                     # 存放数据集 (例如 CIFAR10, MNIST)
├── models_output/            # 存放训练好的目标模型、影子模型和攻击模型
│   └── CIFAR10/
│   └── MNIST/
├── attack.py                 # 攻击逻辑实现、数据加载、模型训练和评估的核心脚本
├── cli.py                    # 命令行接口，用于运行不同配置的攻击实验
├── model.py                  # 定义目标模型、影子模型和攻击模型的网络结构
├── train.py                  # 包含模型训练、验证和攻击数据准备的辅助函数
├── requirements.txt          # 项目依赖的 Python 包
└── README.md                 # 本说明文件
```

## 原理简介

成员推理攻击旨在识别一个特定的数据记录是否被用于训练目标机器学习模型。在影子模型攻击的场景中，攻击流程如下：

1.  **数据准备**:
    * 攻击者需要一个与目标模型训练数据分布相似的数据集。
    * 这个数据集被划分为多个部分，用于训练多个影子模型以及目标模型（如果攻击者可以训练或拥有目标模型副本）。
    * 对于每个影子模型，一部分数据作为其训练集（成员），另一部分作为其测试集（非成员）。

2.  **影子模型训练**:
    * 攻击者训练多个影子模型。这些模型通常与目标模型具有相似的架构和训练过程。
    * 每个影子模型都从其分配的数据子集中学习。

3.  **攻击数据生成**:
    * 对于每个训练好的影子模型，攻击者收集其对其训练数据（成员）和未训练数据（非成员）的预测输出（通常是类别概率向量或置信度得分）。
    * 这些输出向量与相应的标签（成员为1，非成员为0）一起构成了攻击模型的训练数据集。

4.  **攻击模型训练**:
    * 一个二元分类器（攻击模型）在上述生成的数据集上进行训练。
    * 攻击模型学习从模型的输出后验概率中区分成员和非成员。

5.  **攻击执行**:
    * 当攻击者想要判断某个数据记录是否是目标模型训练集的一部分时，他们首先将该记录输入到目标模型中以获取其输出后验概率。
    * 然后，将这些后验概率输入到训练好的攻击模型中。
    * 攻击模型的输出（例如，类别1的概率）表示该记录是目标模型训练成员的置信度。

## 环境要求

* Python 3.10+
* PyTorch (详见 `requirements.txt`)
* Torchvision (详见 `requirements.txt`)
* Scikit-learn (详见 `requirements.txt`)
* NumPy (详见 `requirements.txt`)
* Click (详见 `requirements.txt`)

请通过以下命令安装所需依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据和模型路径

* 创建 `data` 目录用于存放数据集。代码会自动下载 CIFAR10 和 MNIST 数据集（如果本地不存在）。
* 创建 `models_output` 目录用于保存训练过程中的模型检查点。

默认情况下，`cli.py` 脚本期望 `data` 和 `models_output` 目录位于项目根目录的上一级（例如 `../data`, `../models_output`）。你可以通过命令行参数修改这些路径。

### 2. 运行实验

使用 `cli.py` 脚本来运行攻击实验。它提供了不同的命令和选项来控制实验流程。

**基本命令结构:**
```bash
python cli.py membership-inference <COMMAND> [OPTIONS]
```

**可用命令:**

* `train-dummy`: 训练目标模型、影子模型和攻击模型。
    ```bash
    python cli.py membership-inference train-dummy \
        --dataset CIFAR10 \
        --data-path ./data \
        --model-path ./models_output \
        --augm \
        --topk \
        --param-init \
        --verbose
    ```
* `pretrained-dummy`: 使用预训练好的目标模型和影子模型（假设模型文件已存在于 `model-path` 指定的路径下），仅训练和测试攻击模型。
    ```bash
    python cli.py membership-inference pretrained-dummy \
        --dataset CIFAR10 \
        --data-path ./data \
        --model-path ./models_output \
        --topk \
        --verbose
    ```

**常用选项:**

* `--dataset TEXT`: 选择数据集，可选 'CIFAR10' (默认) 或 'MNIST'。
* `--data-path PATH`: 数据集存放路径。
* `--model-path PATH`: 模型检查点保存或加载路径。
* `--no-train-target`: (仅用于 `train-dummy`) 不训练目标模型，尝试加载已有的。
* `--no-train-shadow`: (仅用于 `train-dummy`) 不训练影子模型，尝试加载已有的。
* `--augm`: 是否对目标和影子模型的训练数据进行数据增强 (仅用于 `train-dummy`)。
* `--topk`: 是否使用 Top-K 后验概率作为攻击模型的输入特征。
* `--param-init`: 是否启用自定义的模型参数初始化 (仅用于 `train-dummy`)。
* `--verbose`: 输出详细的训练和评估信息。
* `--no-early-stopping`: 禁用早停机制。

**示例：在 CIFAR10 数据集上从头开始训练所有模型并进行攻击**
```bash
python cli.py membership-inference train-dummy --dataset CIFAR10 --data-path ./data --model-path ./models_output
```

**示例：使用 MNIST 数据集，并假设目标和影子模型已预训练好**
```bash
python cli.py membership-inference train-dummy --dataset CIFAR10 --data-path ./data --model-path ./models_output
```