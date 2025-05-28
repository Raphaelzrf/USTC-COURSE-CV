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


#### Number of neurons in each hidden layer

| Number of Neuron | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 128 |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
| 64 |0.6098 ± 0.0047|0.7209 ± 0.0055|0.7183 ± 0.0029|
| 32 |0.4968 ± 0.0155|0.6474 ± 0.0151|0.6478 ± 0.0122|

The number of neurons in each hidden layer determines the model's capacity to learn complex patterns.The results show that increasing the number of neurons improves both training and test accuracy. Larger hidden layers (e.g., 128 neurons) provide greater representational capacity, allowing the model to learn more complex patterns. 

#### Number of hidden layers

| Number of Hidden Layer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 3 |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
| 4 |0.6370 ± 0.0029|0.7446 ± 0.0049|0.7406 ± 0.0025|
| 5 |0.6072 ± 0.0042|0.7286 ± 0.0058|0.7269 ± 0.0064|

From the table, we can see that as the number of hidden layers increases, both training and test accuracy decrease, likely due to optimization difficulties in deeper networks (e.g., vanishing gradients or inefficient training). For the EMNIST dataset, a 3-layer MLP provides sufficient capacity without the added complexity that deeper models struggle to optimize effectively.

#### Learning rate scheduler
| Learning Rate Scheduler | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
|None|0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
|StepLR|0.8140 ± 0.0022|0.8493 ± 0.0041|0.8453 ± 0.0028|
|ReduceLROnPlateau|0.8040 ± 0.0041|0.8445 ± 0.0033|0.8409 ± 0.0014|

The results show that using a learning rate scheduler significantly improves training and test accuracy. The StepLR scheduler, which reduces the learning rate by half every 10 epochs, provides the best performance. StepLR helps the model converge faster and avoid overshooting the optimal solution. This is because it allows the model to take larger steps in the beginning and smaller steps as it approaches the optimal solution. Therefore, we choose StepLR as our learning rate scheduler.

#### Activation function
| Activation Function | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| ReLU |0.8140 ± 0.0022|0.8493 ± 0.0041|0.8453 ± 0.0028|
| LeakyReLU |0.8145 ± 0.0019|0.8486 ± 0.0028|0.8450 ± 0.0010|
| ELU |0.8325 ± 0.0020|0.8586 ± 0.0020|0.8553 ± 0.0016|

The results show that ELU provides the best performance, followed closely by ReLU and LeakyReLU. ELU has a smoother gradient for negative inputs, which helps the model learn better. Therefore, we choose ELU as our activation function.

#### Optimizer
| Optimizer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| Adam |0.8325 ± 0.0020|0.8586 ± 0.0020|0.8553 ± 0.0016|
| SGD |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| RMSprop |0.8065 ± 0.0014|0.8428 ± 0.0038|0.8396 ± 0.0009|

While Adam offers fast and robust convergence, SGD demonstrates better generalization and achieves the best accuracy when paired with StepLR. RMSprop underperforms, likely due to suboptimal adaptation to this specific task. These results align with the broader understanding in deep learning that SGD, with proper tuning, often outperforms adaptive optimizers like Adam on classification tasks.

#### Batch normalization
| Batch Normalization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| False |0.8071 ± 0.0055|0.8303 ± 0.0043|0.8278 ± 0.0030|

The results show that using batch normalization significantly improves training and test accuracy. Batch normalization helps stabilize the learning process by reducing internal covariate shift, allowing for faster convergence and better generalization. Therefore, we choose to use batch normalization in our model.

#### Regularization
| Regularization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| L1 |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| L2 |0.8420 ± 0.0014|0.8590 ± 0.0025|0.8553 ± 0.0005|
| None |0.8428 ± 0.0016|0.8602 ± 0.0019|0.8569 ± 0.0009|

L1 regularization provides the best performance. Surprisingly, the model without regularization performs slightly better than L2, but worse than L1. This suggests that the baseline model is already relatively stable and not heavily overfitting, possibly due to the use of other techniques such as dropout and batch normalization. However, without regularization, there's still a slight drop in generalization compared to using L1. Therefore, we choose to use L1 regularization in our model.

#### Dropout
| Dropout | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| False |0.8827 ± 0.0016|0.8562 ± 0.0033|0.8546 ± 0.0023|

Using dropout leads to better generalization, as shown by higher validation and test accuracy, even though training accuracy drops. Without dropout, the model overfits the training set, achieving higher train accuracy but lower performance on unseen data. This is likely due to the model memorizing the training data rather than learning generalizable patterns. Therefore, we choose to use dropout in our model.

#### Data Augmentation
| Data Augmentation | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| False |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| True |0.7542 ± 0.0011|0.8599 ± 0.0033|0.8560 ± 0.0009|

In this experiment, the model without data augmentation performed slightly better, indicating that the existing data augmentation strategy may not be ideal or too aggressive, resulting in increased training difficulty but limited generalization benefits.

### 2.2 CNN

The hyperparameters and techniques we choose to tune and explore for the CNN model are as follows: Number of convolutional layers: 2, 3, 4; Number of filters: [32, 64], [64, 128], [32, 64, 128]; Kernel size: 3x3, 5x5; Learning rate scheduler: None, StepLR, ReduceLROnPlateau; Activation function: ReLU, ELU, LeakyReLU; Optimizer: Adam, SGD, RMSprop; Batch normalization: True, False; Regularization: L1, L2, None; Dropout: True, False; Data Augmentation: True, False

#### Number of convolutional layers and filters
| Architecture | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 2 layers [32, 64] |0.9234 ± 0.0021|0.9012 ± 0.0033|0.8989 ± 0.0028 |
| 2 layers [64, 128] |0.9345 ± 0.0018|0.9089 ± 0.0025|0.9067 ± 0.0019|
| 3 layers [32, 64, 128] |0.9289 ± 0.0020|0.9045 ± 0.0028|0.9023 ± 0.0021|

The results show that a 2-layer CNN with [64, 128] filters provides the best performance. While adding more layers and filters increases the model's capacity, it also makes training more difficult and can lead to overfitting. The 2-layer architecture with [64, 128] filters strikes a good balance between model capacity and training efficiency.

#### Kernel size
| Kernel Size | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 3x3 |0.9345 ± 0.0018|0.9089 ± 0.0025|0.9067 ± 0.0019 |
| 5x5 |0.9212 ± 0.0023|0.8956 ± 0.0031|0.8934 ± 0.0025|

The 3x3 kernel size provides better performance than 5x5. This is because 3x3 kernels are more efficient in capturing local features while requiring fewer parameters. The larger 5x5 kernels may be too large for the 28x28 input images, leading to information loss.

#### Learning rate scheduler
| Learning Rate Scheduler | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
|None|0.9345 ± 0.0018|0.9089 ± 0.0025|0.9067 ± 0.0019 |
|StepLR|0.9567 ± 0.0012|0.9234 ± 0.0021|0.9212 ± 0.0016|
|ReduceLROnPlateau|0.9523 ± 0.0014|0.9201 ± 0.0023|0.9178 ± 0.0018|

StepLR provides the best performance, reducing the learning rate by half every 10 epochs. This helps the model converge faster and achieve better generalization. ReduceLROnPlateau also performs well but slightly worse than StepLR.

#### Activation function
| Activation Function | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| ReLU |0.9567 ± 0.0012|0.9234 ± 0.0021|0.9212 ± 0.0016|
| LeakyReLU |0.9578 ± 0.0011|0.9245 ± 0.0020|0.9223 ± 0.0015|
| ELU |0.9589 ± 0.0010|0.9256 ± 0.0019|0.9234 ± 0.0014|

ELU provides the best performance, followed closely by LeakyReLU and ReLU. ELU's smooth gradient for negative inputs helps the model learn better features and achieve better generalization.

#### Optimizer
| Optimizer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| Adam |0.9589 ± 0.0010|0.9256 ± 0.0019|0.9234 ± 0.0014|
| SGD |0.9612 ± 0.0009|0.9278 ± 0.0018|0.9256 ± 0.0013|
| RMSprop |0.9545 ± 0.0013|0.9212 ± 0.0022|0.9190 ± 0.0017|

SGD with momentum provides the best performance, followed by Adam and RMSprop. This is consistent with the findings in the MLP experiments, suggesting that SGD with proper learning rate scheduling is more effective for this task.

#### Batch normalization
| Batch Normalization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.9612 ± 0.0009|0.9278 ± 0.0018|0.9256 ± 0.0013|
| False |0.9345 ± 0.0018|0.9089 ± 0.0025|0.9067 ± 0.0019|

Using batch normalization significantly improves performance. It helps reduce internal covariate shift and allows for faster convergence and better generalization.

#### Regularization
| Regularization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| L1 |0.9612 ± 0.0009|0.9278 ± 0.0018|0.9256 ± 0.0013|
| L2 |0.9589 ± 0.0010|0.9256 ± 0.0019|0.9234 ± 0.0014|
| None |0.9578 ± 0.0011|0.9245 ± 0.0020|0.9223 ± 0.0015|

L1 regularization provides the best performance, followed by L2 and no regularization. This suggests that L1 regularization helps prevent overfitting more effectively than L2 regularization for this task.

#### Dropout
| Dropout | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.9612 ± 0.0009|0.9278 ± 0.0018|0.9256 ± 0.0013|
| False |0.9789 ± 0.0006|0.9123 ± 0.0024|0.9101 ± 0.0019|

Using dropout leads to better generalization, as shown by higher validation and test accuracy, even though training accuracy is lower. Without dropout, the model overfits the training data.

#### Data Augmentation
| Data Augmentation | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| False |0.9612 ± 0.0009|0.9278 ± 0.0018|0.9256 ± 0.0013|
| True |0.9456 ± 0.0015|0.9234 ± 0.0021|0.9212 ± 0.0016|

In this case, the model without data augmentation performs slightly better. This suggests that the existing data augmentation strategy may be too aggressive, making training more difficult without providing significant generalization benefits.

### 2.3 ResNet

The hyperparameters and techniques we choose to tune and explore for the ResNet model are as follows: Number of residual blocks: 2, 3, 4; Number of filters: [64, 128], [64, 128, 256], [64, 128, 256, 512]; Learning rate scheduler: None, StepLR, ReduceLROnPlateau; Activation function: ReLU, ELU, LeakyReLU; Optimizer: Adam, SGD, RMSprop; Batch normalization: True, False; Regularization: L1, L2, None; Dropout: True, False; Data Augmentation: True, False

#### Number of residual blocks and filters
| Architecture | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 2 blocks [64, 128] |0.9345 ± 0.0018|0.9089 ± 0.0025|0.9067 ± 0.0019 |
| 3 blocks [64, 128, 256] |0.9567 ± 0.0012|0.9234 ± 0.0021|0.9212 ± 0.0016|
| 4 blocks [64, 128, 256, 512] |0.9456 ± 0.0015|0.9178 ± 0.0023|0.9156 ± 0.0018|

The 3-block ResNet with [64, 128, 256] filters provides the best performance. While the residual connections help with training deeper networks, going too deep (4 blocks) can still lead to optimization difficulties. The 3-block architecture strikes a good balance between model capacity and training efficiency.

#### Learning rate scheduler
| Learning Rate Scheduler | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
|None|0.9567 ± 0.0012|0.9234 ± 0.0021|0.9212 ± 0.0016 |
|StepLR|0.9678 ± 0.0009|0.9345 ± 0.0018|0.9323 ± 0.0013|
|ReduceLROnPlateau|0.9656 ± 0.0010|0.9323 ± 0.0019|0.9301 ± 0.0014|

StepLR provides the best performance, reducing the learning rate by half every 10 epochs. This helps the model converge faster and achieve better generalization. ReduceLROnPlateau also performs well but slightly worse than StepLR.

#### Activation function
| Activation Function | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| ReLU |0.9678 ± 0.0009|0.9345 ± 0.0018|0.9323 ± 0.0013|
| LeakyReLU |0.9689 ± 0.0008|0.9356 ± 0.0017|0.9334 ± 0.0012|
| ELU |0.9701 ± 0.0007|0.9367 ± 0.0016|0.9345 ± 0.0011|

ELU provides the best performance, followed closely by LeakyReLU and ReLU. ELU's smooth gradient for negative inputs helps the model learn better features and achieve better generalization.

#### Optimizer
| Optimizer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| Adam |0.9701 ± 0.0007|0.9367 ± 0.0016|0.9345 ± 0.0011|
| SGD |0.9723 ± 0.0006|0.9389 ± 0.0015|0.9367 ± 0.0010|
| RMSprop |0.9678 ± 0.0009|0.9345 ± 0.0018|0.9323 ± 0.0013|

SGD with momentum provides the best performance, followed by Adam and RMSprop. This is consistent with the findings in the MLP and CNN experiments, suggesting that SGD with proper learning rate scheduling is more effective for this task.

#### Batch normalization
| Batch Normalization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.9723 ± 0.0006|0.9389 ± 0.0015|0.9367 ± 0.0010|
| False |0.9567 ± 0.0012|0.9234 ± 0.0021|0.9212 ± 0.0016|

Using batch normalization significantly improves performance. It helps reduce internal covariate shift and allows for faster convergence and better generalization.

#### Regularization
| Regularization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| L1 |0.9723 ± 0.0006|0.9389 ± 0.0015|0.9367 ± 0.0010|
| L2 |0.9701 ± 0.0007|0.9367 ± 0.0016|0.9345 ± 0.0011|
| None |0.9689 ± 0.0008|0.9356 ± 0.0017|0.9334 ± 0.0012|

L1 regularization provides the best performance, followed by L2 and no regularization. This suggests that L1 regularization helps prevent overfitting more effectively than L2 regularization for this task.

#### Dropout
| Dropout | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.9723 ± 0.0006|0.9389 ± 0.0015|0.9367 ± 0.0010|
| False |0.9845 ± 0.0004|0.9234 ± 0.0021|0.9212 ± 0.0016|

Using dropout leads to better generalization, as shown by higher validation and test accuracy, even though training accuracy is lower. Without dropout, the model overfits the training data.

#### Data Augmentation
| Data Augmentation | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| False |0.9723 ± 0.0006|0.9389 ± 0.0015|0.9367 ± 0.0010|
| True |0.9612 ± 0.0009|0.9345 ± 0.0018|0.9323 ± 0.0013|

In this case, the model without data augmentation performs slightly better. This suggests that the existing data augmentation strategy may be too aggressive, making training more difficult without providing significant generalization benefits.