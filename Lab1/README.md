# Lab 1: Solve Regularized Linear Regression Model

The purpose of this lab is make us totally understand how to solve regularized linear regression problem with **Least Squared Error** and **Newton Method**.

> Of course, don't use any packages except visualization tools

## File Structure
- intro.pdf: introduction to this lab (document is provided by TA)
- main.py: entry point of this project
- solver.py: implement two metohds (LSE and Newton Method)
- utils.py: some utilities for matrix operation

## Problem Definition
Given a dataset, each sample is in format `(x, y)`, find the parameters $w_n$ of linear regression model which fits on this dataset.

Linear regression model:
$$
w_0x^0 + w_1x^1 + w_2x^2 + ... + w_nx^n = y
$$

If we have three samples in dataset:

- (1, 2)
- (3, 5)
- (2, 10)

and we insert the real value of each sample into this equation with $n=2$, we get:

$$
1w_0 + 1w_1 + 1w_2 = 2
$$

$$
1w_0 + 3w_1 + 9w_2 = 5
$$

$$
1w_0 + w_1 + 4w_2 = 10
$$

we can convert above equation into vector format:

$$
A\vec{x} = \vec{b}
$$

and we want to solve $\vec{x}$ !

## Least Squared Error

If we use LSE with regularization as our loss function, we have:

$$
\text{Loss} = \lVert A\vec{x}-\vec{b} \rVert + \lambda \lVert w \rVert^2
$$

in order to find best $\vec{x}$ which minimizes $\text{Loss}$, we calculate:

$$
\frac{d}{d\vec{x}}\text{Loss} = 0
$$

Finally, we get

$$
\vec{x} = (A^TA + \lambda I)^{-1}A^T\vec{b}
$$

> Just implment this equation to find $\vec{x}$ !

## Newton Metohd

We still use same loss function but without regularization term:

$$
\text{Loss} = \lVert A\vec{x}-\vec{b} \rVert
$$

In Newton Method, we can iterativly update $\vec{x}$ in this way:

$$
\vec{x}_{k+1} = \vec{x}_{k} - H^{-1}_{k} g_k
$$

where $H$ is hessian and $g$ is gradient. When we use LSE as loss function, we get:

$$
g = \frac{d}{d\vec{x}}\text{Loss} = 2A^TA\vec{x} - 2A^T\vec{b}
$$

$$
H = \frac{d^2}{d^2\vec{x}}\text{Loss} = 2A^TA
$$

At first, we can initialize $\vec{x}$ as zero vector.

> Just calculate $H$ and $g$ and iterativly update $\vec{x}$ until $g$ is small enough

## The Most Difficult Part for Me

I forgot how to calculate the inverse of matrix !!! After reviewing some tutorial, I find there are three methods to find the inverse of a matrix:

- Gauss-Jordan Decomposition
- LU Decomposition
- LL Decomposition

For convenience, I implment Gauss-Jordan Decomposition in this lab.