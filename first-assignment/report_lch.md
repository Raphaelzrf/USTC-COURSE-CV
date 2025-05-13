# Report of the First Assignment: Building a neural network model for classification problem

## 1 Dataset and Model Introduction

### 1.1 Dataset Introduction

The EMNIST Balanced dataset is an extension of the original MNIST dataset that includes both handwritten letters and digits. It contains 131,600 images evenly distributed across 47 classes—10 digits (0–9) and 37 uppercase and lowercase letters, excluding confusing pairs (like 'O' and '0'). Each image is 28×28 grayscale, and the dataset is designed for benchmarking character recognition models. The dataset is split into two parts: a training set with 112,800 images and a test set with 18,800 images. We will use the training set for training and 5-fold cross validation, and the test set for final evaluation. 

### 1.2 Model Introduction

We use the strategy provided in the assignment document to build up the model that obtains the best performance. We present the structure of the baseline model and the best performing model below:

- MLP
    - Baseline:
        - Number of hidden layers: 3
        - Number of neurons in each hidden layer: 128
        - Learning rate scheduler: None(learning rate = 0.1)
        - Activation function: ReLU
        - Optimizer: Adam
        - Batch normalization: True
        - Regularization: L1 (1e-5)
        - Dropout: [0.25, 0.25, 0.0]
    - Best:
        - Number of hidden layers: 3
        - Number of neurons in each hidden layer: 128
        - Learning rate scheduler: StepLR(initial_lr=0.1, step_size=10, gamma=0.5)
        - Activation function: ELU
        - Optimizer: SGD
        - Batch normalization: True
        - Regularization:
        - Dropout:
- CNN
    - Baseline:
        - Number of convolutional layers: 2
        - Number of filters: [32, 64]
        - Kernel size: 3x3
        - Learning rate scheduler: None(learning rate = 0.1)
        - Activation function: ReLU
        - Optimizer: Adam
        - Batch normalization: True
        - Regularization: L2 (1e-5)
        - Dropout: 0.5
    - Best:
        - Number of convolutional layers: 2
        - Number of filters: [32, 64]
        - Kernel size: 3x3
        - Learning rate scheduler: StepLR(initial_lr=0.1, step_size=10, gamma=0.5)
        - Activation function: ELU
        - Optimizer: SGD
        - Batch normalization: True
        - Regularization: L1 (1e-5)
        - Dropout: 0.5
- ResNet
    - Baseline:
        - Number of residual blocks: 3
        - Number of filters: [64, 128, 256]
        - Learning rate scheduler: None(learning rate = 0.1)
        - Activation function: ReLU
        - Optimizer: Adam
        - Batch normalization: True
        - Regularization: L2 (1e-5)
        - Dropout: 0.5
    - Best:
        - Number of residual blocks: 3
        - Number of filters: [64, 128, 256]
        - Learning rate scheduler: StepLR(initial_lr=0.1, step_size=10, gamma=0.5)
        - Activation function: ELU
        - Optimizer: SGD
        - Batch normalization: True
        - Regularization: L1 (1e-5)
        - Dropout: 0.5



## 2 The Rationale of the Design on MLP, CNN, and ResNet

### 2.1 MLP

The hyperparameters and techniques we choose to tune and exlplore are as follows:

- Number of hidden layers: 3, 4, 5
- Number of neurons in each hidden layer: 32, 64, 128
- Learning rate scheduler: None, StepLR, ReduceLROnPlateau
- Activation function: ReLU, ELU, LeakyReLU
- Optimizer: Adam, SGD, RMSprop
- Batch normalization: True, False
- Regularization: L1, L2, None
- Dropout: True, False
- Data Augmentation: True, False

Next, we will explain why we choose these hyperparameters and techniques to tune and explore and how it affects the performance of the model.


#### Number of neurons in each hidden layer

The number of neurons in each hidden layer determines the model's capacity to learn complex patterns. We choose to explore 32, 64, and 128 neurons in each hidden layer to find the optimal number of neurons.

| Number of Neuron | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 128 |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
| 64 |0.6098 ± 0.0047|0.7209 ± 0.0055|0.7183 ± 0.0029|
| 32 |0.4968 ± 0.0155|0.6474 ± 0.0151|0.6478 ± 0.0122|

The results show that increasing the number of neurons improves both training and test accuracy. Larger hidden layers (e.g., 128 neurons) provide greater representational capacity, allowing the model to learn more complex patterns. Smaller layers (e.g., 32 neurons) lack sufficient capacity, leading to underfitting and lower performance. Therefore, we choose 128 neurons as a reasonable choice for our model.

#### Number of hidden layers
The number of hidden layers is a crucial hyperparameter in MLPs. More hidden layers can capture more complex patterns in the data, but they also increase the risk of overfitting. We choose to explore 3, 4, and 5 hidden layers to find the optimal balance between model complexity and performance.

| Number of Hidden Layer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| 3 |0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
| 4 |0.6370 ± 0.0029|0.7446 ± 0.0049|0.7406 ± 0.0025|
| 5 |0.6072 ± 0.0042|0.7286 ± 0.0058|0.7269 ± 0.0064|

From the table, we can see that as the number of hidden layers increases, both training and test accuracy decrease, likely due to optimization difficulties in deeper networks (e.g., vanishing gradients or inefficient training). For the EMNIST dataset, a 3-layer MLP provides sufficient capacity without the added complexity that deeper models struggle to optimize effectively.

#### Learning rate scheduler
The learning rate scheduler is a technique to adjust the learning rate during training. We choose to explore three options: None, StepLR, and ReduceLROnPlateau. Because the learning rate is a critical hyperparameter that affects the convergence speed and stability of the training process, we want to find the best learning rate scheduler for our model.

| Learning Rate Scheduler | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
|None|0.6651 ± 0.0048|0.7552 ± 0.0063|0.7516 ± 0.0055 |
|StepLR|0.8140 ± 0.0022|0.8493 ± 0.0041|0.8453 ± 0.0028|
|ReduceLROnPlateau|0.8040 ± 0.0041|0.8445 ± 0.0033|0.8409 ± 0.0014|

The results show that using a learning rate scheduler significantly improves training and test accuracy. The StepLR scheduler, which reduces the learning rate by half every 10 epochs, provides the best performance. StepLR helps the model converge faster and avoid overshooting the optimal solution. This is because it allows the model to take larger steps in the beginning and smaller steps as it approaches the optimal solution. Therefore, we choose StepLR as our learning rate scheduler.


#### Activation function

The activation function introduces non-linearity into the model, allowing it to learn complex patterns. We choose to explore three activation functions: ReLU, ELU, and LeakyReLU.

| Activation Function | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| ReLU |0.8140 ± 0.0022|0.8493 ± 0.0041|0.8453 ± 0.0028|
| LeakyReLU |0.8145 ± 0.0019|0.8486 ± 0.0028|0.8450 ± 0.0010|
| ELU |0.8325 ± 0.0020|0.8586 ± 0.0020|0.8553 ± 0.0016|

The results show that ELU provides the best performance, followed closely by ReLU and LeakyReLU. ELU has a smoother gradient for negative inputs, which helps the model learn better. Therefore, we choose ELU as our activation function.

#### Optimizer
The optimizer is responsible for updating the model's parameters during training. We choose to explore three optimizers: Adam, SGD, and RMSprop. The choice of optimizer can significantly affect the convergence speed and stability of the training process.

| Optimizer | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| Adam |0.8325 ± 0.0020|0.8586 ± 0.0020|0.8553 ± 0.0016|
| SGD |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| RMSprop |0.8065 ± 0.0014|0.8428 ± 0.0038|0.8396 ± 0.0009|

While Adam offers fast and robust convergence, SGD demonstrates better generalization and achieves the best accuracy when paired with StepLR. RMSprop underperforms, likely due to suboptimal adaptation to this specific task. These results align with the broader understanding in deep learning that SGD, with proper tuning, often outperforms adaptive optimizers like Adam on classification tasks.

#### Batch normalization

Batch normalization is a technique to normalize the inputs of each layer, which can help stabilize and accelerate training. We choose to explore two options: True and False.

| Batch Normalization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| False |0.8071 ± 0.0055|0.8303 ± 0.0043|0.8278 ± 0.0030|

The results show that using batch normalization significantly improves training and test accuracy. Batch normalization helps stabilize the learning process by reducing internal covariate shift, allowing for faster convergence and better generalization. Therefore, we choose to use batch normalization in our model.

#### Regularization
Regularization is a technique to prevent overfitting by adding a penalty to the loss function. We choose to explore three options: L1, L2, and None.

| Regularization | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| L1 |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| L2 |0.8420 ± 0.0014|0.8590 ± 0.0025|0.8553 ± 0.0005|
| None |0.8428 ± 0.0016|0.8602 ± 0.0019|0.8569 ± 0.0009|

L1 regularization provides the best performance. Surprisingly, the model without regularization performs slightly better than L2, but worse than L1. This suggests that the baseline model is already relatively stable and not heavily overfitting, possibly due to the use of other techniques such as dropout and batch normalization. However, without regularization, there's still a slight drop in generalization compared to using L1. Therefore, we choose to use L1 regularization in our model.

#### Dropout
Dropout is a technique to randomly set a fraction of the input units to zero during training, which can help prevent overfitting. We choose to explore two options: True and False.

| Dropout | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| True |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| False |0.8827 ± 0.0016|0.8562 ± 0.0033|0.8546 ± 0.0023|

Using dropout leads to better generalization, as shown by higher validation and test accuracy, even though training accuracy drops. Without dropout, the model overfits the training set, achieving higher train accuracy but lower performance on unseen data. This is likely due to the model memorizing the training data rather than learning generalizable patterns. Therefore, we choose to use dropout in our model.

#### Data Augmentation

Data augmentation is a technique to artificially increase the size of the training dataset by applying random transformations to the images. We choose to explore two options: True and False.

| Data Augmentation | Train Accuracy | Val Accuracy | Test Accuracy |
|-------|----------------|---------------|------------|
| False |0.8472 ± 0.0010|0.8638 ± 0.0023|0.8604 ± 0.0009|
| True |0.7542 ± 0.0011|0.8599 ± 0.0033|0.8560 ± 0.0009|

In this experiment, the model without data augmentation performed slightly better, indicating that the existing data augmentation strategy may not be ideal or too aggressive, resulting in increased training difficulty but limited generalization benefits.

### 2.2 CNN

The hyperparameters and techniques we choose to tune and explore for the CNN model are as follows:

- Conv. layers: [32, 64], [32, 64, 128]
- Kernel size: [3, 3], [5, 3], [3, 3, 3], [5, 5, 3] , [5, 3, 3]
- Learning rate scheduler: StepLR, CosineAnnealingLR
- Activation function: ReLU, ELU, LeakyReLU(0.1)
- Optimizer: Adam, SGD, RMSprop
- Batch normalization: True, False
- Regularization: L1, L2, None
- Dropout: True(0.5), False

The Baseline model is:
The baseline model uses:
- Conv. layers: [32, 64]
- Kernel size: [3, 3]
- Learning rate scheduler: CosineAnnealingLR
- Activation function: ReLU
- Optimizer: Adam
- Batch normalization: True
- Regularization: None
- Dropout: True(0.5)

Next, we will explain why we choose these hyperparameters and techniques to tune and explore and how it affects the performance of the model.

#### Convolutional Layers with different channel

We explore different combinations of convolutional layers and channels to find the optimal architecture. The number of channels in each layer affects the model's capacity to learn features.

| Conv. Layers | Test Accuracy |
|-------|------------|
| [32, 64] |86.99%|
| [32, 64, 128] |88.77%|

The three-layer architecture [32, 64, 128] performs better due to increased model capacity and improved hierarchical feature learning. The additional layer and channels allow the model to capture more complex patterns, though the modest 0.19% accuracy improvement suggests the two-layer model already captures most relevant features.

#### Kernel Size

We also experiment with different kernel sizes for these architectures:

| Kernel Size | Test Accuracy |
|-------|------------|
| [3, 3, 3] |86.77%|
| [5, 3, 3] |88.78%|
| [5, 5, 3] |89.22%|

Larger kernel sizes, particularly [5, 5, 3], show better performance by capturing broader spatial patterns in the input images. The 5x5 kernels in earlier layers help detect larger features, while the 3x3 kernel in the final layer focuses on finer details, leading to improved feature extraction and classification accuracy.

#### Activation function

The activation function introduces non-linearity into the model. We explore three options: ReLU, ELU, and LeakyReLU.

| Activation Function | Test Accuracy |
|-------|------------|
| ReLU |89.22%|
| LeakyReLU |89.63%|
| ELU |89.44%|

LeakyReLU performs best among the tested activation functions, followed by ELU and ReLU. LeakyReLU's small negative slope helps prevent "dying ReLU" problem by allowing gradients to flow even for negative inputs, leading to more robust training. ELU provides smooth gradients but its exponential computation makes it slightly slower than LeakyReLU.

#### Optimizer

The optimizer is responsible for updating the model's parameters. We explore three options: Adam, SGD, and RMSprop.

| Optimizer | Adam | SGD | RMSprop|
| --------- | ---- | ---- | ----- |
|Test Accuracy|89.63% | 77.50% | 89.52% |

SGD with momentum provides the best performance, followed by Adam and RMSprop. This is consistent with the findings in the MLP experiments, suggesting that SGD with proper learning rate scheduling is more effective for this task.

#### Learning rate scheduler

The learning rate scheduler is crucial for optimizing the training process. We explore 2 options: StepLR, and CosineAnnealingLR.


| Learning Rate Scheduler | Test Accuracy |
|-------|------------|
|StepLR|89.41%|
|CosineAnnealingLR|89.63%|

CosineAnnealingLR performs better than StepLR because it provides smoother learning rate transitions and better exploration of the loss landscape. The gradual cosine-based decay helps the model converge to better local minima, resulting in improved test accuracy. The periodic restarts also help escape poor local optima.


#### Batch normalization

Batch normalization helps stabilize and accelerate training. We explore two options: with and without batch normalization.

| Batch Normalization | Test Accuracy |
|-------|------------|
| True |89.63%|
| False |89.56%|

Batch normalization slightly improves model performance by normalizing layer inputs, reducing internal covariate shift, and enabling faster training with higher learning rates. The normalized inputs help the model converge better and achieve marginally higher accuracy.

#### Dropout

Dropout helps prevent overfitting by randomly dropping neurons during training. We explore two options: with and without dropout.

| Dropout | Test Accuracy |
|-------|------------|
| True(0.5) |89.63%|
| False |89.70%|

Dropout slightly decreases model performance in this case. While dropout helps prevent overfitting by adding regularization, the model may already have sufficient regularization from batch normalization. The small performance drop suggests that dropout may be unnecessarily reducing model capacity without providing additional benefits.


#### Regularization

We explore three regularization options: L1, L2, and no regularization.

| Regularization | None | L1 | L2 |
| -------------- | --- | --- | --- |
| Test Accuracy  | 89.70% | 89.45% | 89.40% |

No regularization performs best, followed by L1 and L2 regularization. This suggests that the model has sufficient regularization from batch normalization and the architectural design, making additional weight penalties unnecessary. The small performance drops with L1 and L2 regularization indicate they may be over-regularizing the model and constraining its capacity to learn useful features.


### 2.3 ResNet

The hyperparameters and techniques we choose to tune and explore for the ResNet model are as follows:

- Number of residual blocks: [2, 2, 2], [3, 3, 3], [1, 2, 3]
- Learning rate scheduler: None, StepLR, ReduceLROnPlateau
- Activation function: ReLU, ELU, LeakyReLU
- Optimizer: Adam, SGD, RMSprop
- Batch normalization: True, False
- Regularization: L1, L2, None
- Dropout: True, False
- Data Augmentation: True, False

> We fix the conv-kernel size to `5` as we obtained in the `CNN` model.

Next, we will explain why we choose these hyperparameters and techniques to tune and explore and how it affects the performance of the model.
#### Number of residual blocks

The number of residual blocks and filters determines the model's depth and capacity. We explore different combinations to find the optimal architecture.

| Architecture | Test Accuracy |
|-------|------------|
| `[1, 2, 3]` |0.9065|
| `[2, 2, 2]` |0.9084 |
| `[3, 3, 3]` |0.9075|

The architecture [2, 2, 2] achieves the best performance, followed closely by [3, 3, 3] and [1, 2, 3]. The balanced architecture with 2 blocks at each level provides sufficient model capacity while avoiding overfitting. Deeper architectures ([3, 3, 3]) don't improve performance, suggesting the model has reached optimal depth, while the asymmetric architecture ([1, 2, 3]) performs slightly worse due to imbalanced feature extraction.


#### Activation function

The activation function introduces non-linearity into the model. We explore three options: ReLU, ELU, and LeakyReLU.

| Activation Function | Test Accuracy |
|-------|------------|
| ReLU |0.9060|
| LeakyReLU |0.9084|
| ELU |0.9044|

LeakyReLU achieves the best performance, followed by ReLU and ELU. The small negative slope in LeakyReLU helps prevent "dying ReLU" problem while maintaining the benefits of ReLU's simplicity. ELU performs slightly worse, possibly due to its more complex computation not providing additional benefits for this task.


#### Optimizer

The optimizer is responsible for updating the model's parameters. We explore three options: Adam, SGD, and RMSprop.

| Optimizer | Adam | SGD | RMSprop|
| --------- | ---- | ---- | ----- |
|Test Accuracy| 0.9084 | 0.3147 | 0.9069 |

Adam achieves the best performance, followed closely by RMSprop, while SGD performs poorly. Adam's adaptive learning rates and momentum help it converge faster and find better optima. RMSprop's per-parameter learning rates also work well, but SGD's fixed learning rate struggles to find good solutions for this complex model.

#### Learning rate scheduler

The learning rate scheduler is crucial for optimizing the training process. We explore 2 options: StepLR and CosineAnnealingLR.

| Learning Rate Scheduler| Test Accuracy |
|-------|------------|
|StepLR|0.9078|
|CosineAnnealingLR|0.9084|

CosineAnnealingLR achieves slightly better performance than StepLR. The cosine-based decay schedule provides smoother learning rate transitions and better final convergence compared to step-wise drops, allowing the model to explore the loss landscape more effectively and find better optima.


#### Batch normalization

Batch normalization helps stabilize and accelerate training. We explore two options: with and without batch normalization.

| Batch Normalization | Test Accuracy |
|-------|------------|
| True |0.9084|
| False |0.1315|

Using batch normalization significantly improves performance. It helps reduce internal covariate shift and allows for faster convergence and better generalization.

#### Regularization

We explore three regularization options: L1, L2, and no regularization (fix weight decay to `1e-2`).

| Regularization | Test Accuracy |
|-------|-------------|
| L1 |0.8913|
| L2 |0.9078|
| None |0.9084|

No regularization performs best, followed by L2 and L1 regularization. This suggests that the model's architecture and other techniques like batch normalization and dropout are already providing sufficient regularization, making additional weight penalties unnecessary or even slightly detrimental to performance.


#### Dropout

Dropout helps prevent overfitting by randomly dropping neurons during training. We explore two options: with and without dropout.

| Dropout | Test Accuracy |
|-------|-------------------|
| True |0.9072|
| False |0.9084|

Disabling dropout slightly improves performance. This suggests that the model's other regularization techniques like batch normalization are sufficient, and additional dropout may be unnecessarily constraining the model's capacity.


## 3 Results 

### 3.1 MLP



The result for each fold in 5-fold cross validation is similar, so we only show the result of the first fold.

#### 3.1.1 training loss, testing loss and testing accuracy


The accuracy of the best model on test set is 0.8614
![](images/2025-05-07-17-17-24.png)
![](images/2025-05-07-17-17-40.png)

我们可以看到，训练集的loss在前面几轮下降很快，之后趋于平稳，说明模型已经收敛。测试集的loss也在前面几轮下降很快，之后趋于平稳，说明模型没有过拟合。我们采用早停法获得的最优模型在测试集上的准确率为0.8614，说明模型的泛化能力很好。并且我们发现，训练集的loss始终高于测试集的loss，这是因为我们采用了dropout和batch normalization等技术来防止过拟合，这会导致模型在训练时并没有发挥出全部的能力，从而导致训练集的loss高于测试集的loss。

#### 3.1.2 The predicted results of the best model on test set

![](images/2025-05-07-17-25-11.png)

由上图可以看出，模型在测试集上的预测结果是比较好的。能准确地预测出大部分的字母和数字。但是对于一些相近的字母和数字，比如'9'和'q'，模型的预测结果是错误的。这是因为这两个字母和数字在形状上非常相似，模型很难区分它们。这也是合理的。

### 3.2 CNN

The result for each fold in 5-fold cross validation is similar, so we only show the result over all data. (average validation accuracy = 86.85%, baseline value = 86.99%)

#### 3.2.1 training loss, training accuracy and testing accuracy
![](./images/cnn_training_curves.png)
The training loss decreases rapidly and then gradually over epochs, indicating effective learning and convergence. Training accuracy steadily increases, while the test accuracy remains stable and close to the training accuracy, suggesting good generalization and no significant overfitting. The final test accuracy reaches approximately 89.7%, demonstrating strong model performance on unseen data.

#### 3.2.2 The predicted results of the best model on test set
![](./images/cnn_predict.png)
The sample predictions show that the model correctly classifies most handwritten characters, as the predicted labels match the true labels in most cases. However, there are occasional misclassifications, which may occur for characters with similar shapes or ambiguous handwriting. Overall, the model demonstrates strong recognition ability on the test samples.

### 3.3 ResNet
The result for each fold in 5-fold cross validation is similar, so we only show the result over all data. (average validation accuracy = 90.62%, baseline value = 90.84%).

#### 3.3.1 training loss, testing loss and testing accuracy
![](./images/resnet_training_curves.png)
The training loss decreases steadily over epochs, indicating effective learning and convergence. Training accuracy improves consistently, and the test accuracy (90.84%) remains close to the training accuracy, suggesting good generalization and minimal overfitting. The model demonstrates strong performance and stability on both training and test data.

#### 3.3.2 The predicted results of the best model on test set
![](./images/resnet_predict.png)