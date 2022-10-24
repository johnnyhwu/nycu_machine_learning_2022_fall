# Lab 2: Build a Naive Bayes Classifier on MNIST
This lab contains three tasks. In the first task, I have to implement **discrete** and **continuous** version of Naive Bayes Classifier on MNIST dataset. In the second task, I have to implment online learning with **Beta-Binomial Conjugation**. In the last task, try to prove Beta-Binomial Conjugation.

> This README is mainly for the first task !

## Problem Definition
Given MNIST dataset, build a Naive Bayes Classifier with training dataset, and evaluate it with testing dataset. The MNIST dataset in `part1/dataset` is different from the one we usually see. The dataset is represented by big-endian **byte** data instead of image data.

As we know, MNIST dataset is composed on 50000 training samples and 10000 testing samples. Each image is in grayscale and the size is 28x28.

## Implementation

### Read Dataset
`np.fromfile()` is good at reading data in byte format. Specify '`>`' in `dtype` to indicate the bytes are in **big-endian**. We read four integer (one integer = 4 bytes in big-endian, therefore 32 bytes in total) with `count=4` at the start of file, and reading all unsigned bytes with `count=-1` in the rest of file.

```python
with open(img_path, "rb") as f:
    _, num_img, num_row, num_col = np.fromfile(file=f, dtype=">i4", count=4)
    all_imgs = np.fromfile(file=f, dtype=">B", count=-1)
```

### Naive Bayes Classifier
No matter discrete or continuous Naive Bayes Classifier, the model we try to build is based on **Bayes Theorem**:

![Bayes Theorem](/Lab2/img/1.png)

In this case, $\theta$ is from 0 to 9. Given current data (image), we want to know the probability of $\theta$ from 0 to 9, which is posterior. In order to calculate it, we must determine the value of prior and likelihood.

Prior can be easily calculated from training data:

$$
p(\theta=0) = \dfrac{\text{Number of training image which is 0}}{\text{Number of training image}}
$$

$$
p(\theta=1) = \dfrac{\text{Number of training image which is 1}}{\text{Number of training image}}
$$

$$
p(\theta=2) = \dfrac{\text{Number of training image which is 2}}{\text{Number of training image}}
$$

$$
...
$$

The problem is how to calculate likelihood, $p(D\mid\theta)$. Each data (image) is composed of 28x28=784 pixels. All 784 pixels are features of this sample. 

For example, given $\theta = 0$, the likelihood is:
$$
p(D\mid\theta=0) = p(d_{1}\mid\theta=0)*p(d_{2}\mid\theta=0)* ...*p(d_{784}\mid\theta=0) 
$$

So, how to calculate $p(d_n\mid\theta=0)$ ? We can first estimate the **distribution** of each feature give $\theta=0$.  Speaking of distribution, discrete type and continuous type should be considered.

### Discrete Version
Take $\theta = 0$ for example, we will collect all training images representing 0 (imagine that we stack all images).

Because the distribution is discrete, we will represents the distribution of each feature by 32 bins (32 is just an example). Because the range of feature is from 0 to 255, if the value of this feature is 5, this feature belongs to 1st bin.

With this method, we calculate the number of samples in each bins, which forms a discrete distribution of this feature.

### Continuous Version
Take $\theta = 0$ for example, we will collect all training images representing 0 (imagine that we stack all images).

However, instead of calculating number of samples in each bins for current feature, we simply calculate the **mean** and **variance** of this feature. With mean and variance, we generate a **Gaussian** distribution for this feature.

### Inference (Predict)
After build prior and likelihood from training dataset, we can classify testing sample by calculating $p(\theta=0\mid D)$, $p(\theta=1\mid D)$, $p(\theta=2\mid D)$, ..., $p(\theta=9\mid D)$.

In order to prevent consecutive multiplication in $p(D\mid\theta) * p(\theta)$, we will take log on equation:

$$
log(p(\theta=0\mid D)) = log(p(d_1\mid\theta)) + log(p(d_2\mid\theta)) + log(p(d_3\mid\theta)) + ... + log(p(d_{784}\mid\theta)) + log(p(\theta))
$$

## Result 

### Accuracy
- Discrete NB: 85.04%
- Continuous NB: 64.83%

### Visulization

See the model's imagination of each digit:

- 2
    ![Digit 2](/Lab2/img/2.png)
- 3
    ![Digit 3](/Lab2/img/3.png)
- 7
    ![Digit 7](/Lab2/img/7.png)
- 9
    ![Digit 9](/Lab2/img/9.png)

