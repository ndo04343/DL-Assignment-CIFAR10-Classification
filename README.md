# Intermediate Report

Kyounpook National University<br>
Computer Science and Engineering<br>
2017111066, Heesung Yang<br>


## 1. Claim
For a model with sufficiently high generalization performance, good
performance will be obtained if it is trained with a sufficient dataset.

## 2. Detail
- Model : ResNet-152 pretrained model
- Optimizer : Adam (learning rate = 0.0001, weight decay = 0)
- Learning rate scheduler : StepLR (step size = 20, gamma = 0.2)
- Loss : Cross entropy loss
- Epochs : 100

## 3. Data Augmentation

There is no way to increase the number of data. So, I increased the data
distribution through random data augmentation. The data augmentation that I used
are random horizontal flip and random color jittering.

#### 3.1. Random Horizontal Flip
In the case of flip, there are vertical and horizontal. I believe that horizontally
flipped data can exist in the natural world, but it is difficult to find vertical
flipped data, so only horizontal flip is performed.

#### 3.2. Random Color Jitering
Since image data handles continuous analog RGB values, even a small change
in color can change the value sensitively. Therefore, the color jitering was
selected.

## 4. Result 
Acc is 97.27, please watch the report.

## 5. Ablation Study
The distribution of augmented data may be slightly different from the actual
distribution. Therefore, if you learn without augmentation at the end, it is likely
that you can tune close to the actual data.<br><br>

Acc is 97.64, please watch the report.
