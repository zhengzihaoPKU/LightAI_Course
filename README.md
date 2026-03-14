# Pruning

剪枝包括：

1. 训练后剪枝（训练-剪枝-微调）
2. 训练时剪枝（rewind）
3. 稀疏训练

## 训练后剪枝（训练-剪枝-微调）

### 非结构化剪枝

| Index |  Model  | Dataset | acc Before prune | prune rate | sparsity before prune | acc After prune | sparsity after prune |
| :---: | :------: | :-----: | :--------------: | :--------: | :-------------------: | :-------------: | :------------------: |
|   1   | ResNet18 | CIFAR10 |      92.56      |    0.9    |          0.0          |      90.59      |        89.92        |
|   2   | ResNet18 | CIFAR10 |      92.56      |    0.8    |          0.0          |      91.34      |        79.93        |
|   3   | ResNet18 | CIFAR10 |      92.56      |    0.7    |          0.0          |      91.41      |        69.94        |
|   4   | ResNet18 | CIFAR10 |      92.56      |    0.6    |          0.0          |      91.33      |        59.95        |
|   5   | ResNet18 | CIFAR10 |      92.56      |    0.5    |          0.0          |      91.57      |        49.96        |

### 结构化剪枝

| Index |  Model  | Dataset | acc Before prune | prune rate | sparsity before prune | acc After prune | sparsity after prune |
| :---: | :------: | :-----: | :--------------: | :--------: | :-------------------: | :-------------: | :------------------: |
|   1   | ResNet18 | CIFAR10 |      92.56      |    0.9    |          0.0          |      18.46      |        89.92        |
|   2   | ResNet18 | CIFAR10 |      92.56      |    0.8    |          0.0          |      28.44      |        79.98        |
|   3   | ResNet18 | CIFAR10 |      92.56      |    0.7    |          0.0          |      37.78      |        69.88        |
|   4   | ResNet18 | CIFAR10 |      92.56      |    0.6    |          0.0          |      46.82      |        59.95        |
|   5   | ResNet18 | CIFAR10 |      92.56      |    0.5    |          0.0          |      56.08      |        49.96        |
|   6   | ResNet18 | CIFAR10 |      92.56      |    0.4    |          0.0          |      64.67      |        39.97        |
|   7   | ResNet18 | CIFAR10 |      92.56      |    0.3    |          0.0          |      73.05      |        30.03        |

# Knowledge distillation

知识蒸馏分为三种：

1. Output Transfer:将网络的输出（Soft-target）作为知识
2. Feature Transfer:将网络学习的特征作为知识
3. Relation Transfer:将网络或者样本的关系作为知识

## Output Transfer

| Index | Teacher model | Student model | Dataset  | Teacher model acc | Student model acc | KD acc |
| :---: | :-----------: | :-----------: | -------- | :---------------: | :---------------: | :----: |
|   1   |  FC-complex  |   FC-simple   | MNIST    |       97.46       |       80.89       | 85.53 |
|   2   |   ResNet18   |   simplecnn   | CIFAR10  |       90.56       |       74.43       | 76.89 |
|   3   |   ResNet50   |   simplecnn   | CIFAR100 |       66.99       |       36.18       | 42.07 |


# 量化
## PTDQ
PTDQ只能量化linear，不能量化conv
| Index |  Model  | Dataset | acc Before quant | data type | acc after quant |
| :---: | :-----: | :-----: | :--------------: | :-------: | :-------------: |
|   1   | ResNet18 | CIFAR10|      92.56       |    INT8   |      92.49      |

## PTSQ
PTSQ需要先建立有量化观测器的网络模型
| Index |  Model  | Dataset | acc Before quant | data type | acc after quant |
| :---: | :-----: | :-----: | :--------------: | :-------: | :-------------: |
|   1   | ResNet18 | CIFAR10|      92.56       |    INT8   |      91.92      |

## QAT
QAT也需要先建立有量化观测器的网络模型
| Index |  Model  | Dataset | acc Before quant | data type | acc after quant |
| :---: | :-----: | :-----: | :--------------: | :-------: | :-------------: |
|   1   | ResNet18 | CIFAR10|      92.56       |    INT8   |       92.51     |