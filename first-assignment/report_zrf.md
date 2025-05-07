# Report of the First Assignment: Building a neural network model for classification problem

## 1. Dataset Introduction

## 2. MLP

我们在进行模型的训练时，采用早停的策略来防止过拟合。我们将训练集划分为训练集和验证集，训练集用于模型的训练，验证集用于监控模型的性能。我们在每个epoch结束时评估模型在验证集上的性能，如果验证集的损失在连续10个epoch内没有下降，则停止训练。

### 2.1 Baseline

根据目前有关MLP在MNIST数据集上表现的文献，我们选择以下的网络结构作为基线：

#### Architecture

- Input Layer: 784 (28x28 pixels)
- 3 Hidden Layer: 128， 128， 128
- Output Layer: 47 (47 classes)

#### Techniques

- Learning Rate: 0.1(fixed)
- Activation Function: ReLU
- Optimizer: Adam
- Batch Normalization: True
- L1 & L2 Regularization: L1 （1e-5）
- Dropout: [0.25, 0.25, 0.0]

#### Results

| Model | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| Baseline |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |


### number of neuron in hidden layer (32, 64, 128)

| Number of Neuron | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 128 |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
| 64 |0.6098 ± 0.0047|0.7209 ± 0.0055|0.7183 ± 0.0029|
| 32 |0.4968 ± 0.0155|0.6474 ± 0.0151|0.6478 ± 0.0122|

由上表可知，随着隐藏层神经元数量的增加，模型的训练精度，验证精度以及测试精度均有所提升。我们可以认为128个神经元是一个合理的选择。所以后续的实验中我们将使用128个神经元。

### Number of Hidden Layer: 3, 4

| Number of Hidden Layer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 3 |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
| 4 |0.6370 ± 0.0029|0.7446 ± 0.0049|0.7406 ± 0.0025|
| 5 |0.6072 ± 0.0042|0.7286 ± 0.0058|0.7269 ± 0.0064|

由上表可知，随着隐藏层数量的增加，模型的训练精度，验证精度以及测试精度均出现了下降。我们可以认为3个隐藏层是一个合理的选择。所以后续的实验中我们将使用3个隐藏层。


### Learning Rate Scheduler: StepLR, ReduceLROnPlateau
| Learning Rate Scheduler | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
|None|0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
|StepLR|0.8140 ± 0.0022|0.8493 ± 0.0041|0.8453 ± 0.0028|
|ReduceLROnPlateau|0.8040 ± 0.0041|0.8445 ± 0.0033|0.8409 ± 0.0014|

由上表可知，使用学习率调度器后，模型的训练精度，验证精度以及测试精度均有所提升。并且使用StepLR的效果更好。所以后续的实验中我们将使用StepLR作为学习率调度器。


### Activation Function: ReLU, LeakyReLU, ELU

| Activation Function | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| ReLU |0.8140 ± 0.0022|0.8493 ± 0.0041|0.8453 ± 0.0028|
| LeakyReLU |0.8145 ± 0.0019|0.8486 ± 0.0028|0.8450 ± 0.0010|
| ELU |0.8325 ± 0.0020|0.8586 ± 0.0020|0.8553 ± 0.0016|

由上表可知，使用ELU作为激活函数更好，所以后续的实验中我们将使用ELU作为激活函数。

### Optimizer: SGD, Adam, RMSprop

| Optimizer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| Adam |0.8325 ± 0.0020|0.8586 ± 0.0020|0.8553 ± 0.0016|
| SGD |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| RMSprop |0.8065 ± 0.0014|0.8428 ± 0.0038|0.8396 ± 0.0009|

由上表可知，使用SGD作为优化器更好，所以后续的实验中我们将使用SGD作为优化器。

### Batch Normalization: True, False

| Batch Normalization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| False |0.8071 ± 0.0055|0.8303 ± 0.0043|0.8278 ± 0.0030|

由上表可知，使用Batch Normalization后，模型的训练精度，验证精度以及测试精度均有所提升。所以后续的实验中我们将使用Batch Normalization。

### L1 & L2 Regularization: L1, L2, None

| Regularization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| L1 |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| L2 |0.8420 ± 0.0014|0.8590 ± 0.0025|0.8553 ± 0.0005|
| None |0.8428 ± 0.0016|0.8602 ± 0.0019|0.8569 ± 0.0009|

由上表可知，使用L1正则化后，模型的训练精度，验证精度以及测试精度均有所提升。所以后续的实验中我们将使用L1正则化。

### Dropout: True, False

| Dropout | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| False |0.8827 ± 0.0016|0.8562 ± 0.0033|0.8546 ± 0.0023|


## 3. CNN

## 4. Resnet