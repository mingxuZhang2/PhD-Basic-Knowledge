# PRML Chapter 1: Introduction

Generalization: The ability to perform accurately on new, unseen examples/tasks after having learned from a set of training examples.

## 1.1 Example: Polynomial Curve Fitting

### 1.1.1 Linear Basis Function Models

The simplest linear model for regression is one that involves a linear combination of fixed nonlinear functions of the input variables. This is known as a linear basis function model. The following equation is a linear model of the form:

$$y(x, \textbf{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j=0}^{M}w_jx^j$$

where $x$ is the input variable, $y$ is the output variable, $w$ is the weight vector, and $M$ is the order of the polynomial. Although the model is nonlinear in the input variable $x$, it is linear in the parameters $\textbf{w}$.

We need to minimize the error function, which is the sum of the squares of the differences between the target values $t$ and the values predicted by the model $y(x, \textbf{w})$:

$$E(\textbf{w}) = \frac{1}{2}\sum_{n=1}^{N}\{y(x_n, \textbf{w}) - t_n\}^2$$

where $N$ is the number of data points.

So in this problem, we just need to find the optimal values of the weight vector $\textbf{w}$ that minimize the error function $E(\textbf{w})$. And it has a closed-form solution, denoted as $\textbf{w}^*$:

$$\textbf{w}^* = (\textbf{Φ}^T\textbf{Φ})^{-1}\textbf{Φ}^T\textbf{t}$$

where $\textbf{Φ}$ is the design matrix, whose elements are given by $\textbf{Φ}_{nj} = x_n^j$, and $\textbf{t}$ is the vector of target values.

![Figure 1](fig/Figure1.png)

Figure 1 shows different value of polynomial order $M$ and the corresponding polynomial curve fitting. As we can see, the polynomial curve fitting becomes more flexible as the order $M$ increases. However, if we set $M$ too large, the model will overfit the data, which means it will perform well on the training data but poorly on the test data.