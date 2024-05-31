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

Another error measuring equation is RMS, which is the square root of the mean of the squares of the error function:

$$E_{RMS} = \sqrt{2E(\textbf{w}^*)/N}$$

![Figure2](fig/Figure2.png)

The figure above shows the RMS error as a function of the order $M$ of the polynomial. As we can see, for training data, the RMS error decreases as the order $M$ increases. However, the RMS error on the test data increases as the order $M$ increases, which means the model is overfitting the data. 

This is a little paradoxical, but it is a common phenomenon in machine learning. If $M=9$, there are 10 coefficients which means it should be able to fit 10 points exactly. However, as the figure shows, the RMS error on the test data is very large.

![alt text](fig/image3.png)

As the figure above shows, the coefficients of the polynomial model with $M=9$ are very large. So it only perform well on the training data. This a also a proof that we need add a regularization term to the error function to prevent overfitting.

What means regularization? It means we add a term to the error function to penalize the large coefficients. In fact, in the coefficient space, there are a lot of solutions that can minimize the error function. But some of them are too large, it will obviously perform poorly on the unknwon data. So we need to choose the solution that has the smallest coefficients that will have a better generalization ability.

So the regularized error function is:

$$E(\textbf{w}) = \frac{1}{2}\sum_{n=1}^{N}\{y(x_n, \textbf{w}) - t_n\}^2 + \frac{\lambda}{2}||\textbf{w}||^2$$

where $\lambda$ is the regularization coefficient, and $||\textbf{w}||^2$ is the square of the Euclidean norm of the weight vector $\textbf{w}$. From a high level aspect, we consider that we need to minimized the error function, incluing two terms. On the one hand, we need to minimized the original error function which is like MSE. On the other hand, we need to minimized the Euclidean norm of the weight vector. $\lambda$ is a trade off between the model complexity and the fitting ability.


![alt text](fig/image5.png)

And the figure above shows different $\lambda$ and the corresponding coefficients of the polynomial model with $M=9$. As we can see, the coefficients become smaller as $\lambda$ increases. And the RMS error on the test data decreases as $\lambda$ increases. So the regularization term can prevent overfitting.

![alt text](fig/image4.png)



Another methods to prevent overfitting is to add more training data. As the figure below shows, if we add more training data, the RMS error on the test data will decrease. Another way to say this is that the larger the training data, the better the generalization ability of the model.

## 1.2 Probability Theory

![alt text](fig/image6.png)

**Sum rule**: $$p(X = x_i) = \sum_{j}p(X = x_i, Y = y_j) = \frac{c_i}{N}$$

Conditional probability: $$p(X = x_i|Y = y_j) =  \frac{n_{i,j}}{c_i}=\frac{p(X = x_i, Y = y_j)}{p(Y = y_j)}$$

In conditional probability, $n_{i,j}$ is the number of times $X = x_i$ and $Y = y_j$ occur together, and $c_i$ is the number of times $X = x_i$ occurs. We can say that the core of conditional probability is shrink the sample space. The original sample space is $N$ while the new sample space is $c_i$. And based on it, we can give the equation of conditional probability as above. 

And we can change the equation of conditional probability to the form of joint probability:

$$p(X = x_i, Y = y_j) =\frac{n_{i,j}}{N}=\frac{n_{i,j}}{c_i} \times \frac{c_i}{N} =p(Y = y_j|X = x_i)p(Y = x_i)$$

The equation above is also called product rule. It is the core of probability theory. It is the foundation of the Bayes' theorem.

And we can also get the Bayes' theorem from the product rule:

$$p(Y = y_j|X = x_i) = \frac{p(X = x_i|Y = y_j)p(Y = y_j)}{p(X = x_i)}$$

### 1.2.1 Probability densities

We call $p(x)$ the probability density function of the continuous variable $x$. And the probability that $x$ lies in the range $[x, x + \delta x]$ is given by $p(x)\delta x$. And the probability that $x$ lies in the range $[x, x + \delta x]$ and $y$ lies in the range $[y, y + \delta y]$ is given by $p(x, y)\delta x \delta y$. So for the single variable, if we need to calculate the probability that $x$ lies in the range $[a, b]$, we can use the following equation:

$$p(a \leq x \leq b) = \int_{a}^{b}p(x)dx$$

And for the joint probability, we can use the following equation:

$$p(a \leq x \leq b, c \leq y \leq d) = \int_{a}^{b}\int_{c}^{d}p(x, y)dxdy$$

If we need to calculate the probability that $x$ lies in the range $[a, b]$, and y lies in the range $[-\infty, +\infty]$, we can use the following equation:

$$p(a \leq x \leq b) = \int_{a}^{b}\int_{-\infty}^{+\infty}p(x, y)dydx$$


And if the variable $x$ is non-linearly transformed by $y$,which means $x=g(y)$, we can get the following equation:

$$
p(y) = p(x)\left|\frac{dx}{dy}\right| = p(g(y))\left|\frac{dg(y)}{dy}\right|
$$

The equation above is the transformation of the probability density function. And the absolute value of the derivative of the transformation function is the Jacobian of the transformation. It will make sure the probability is conserved and non-negative.


This is the PDF(Probability Density Function) of the continuous variable. And the CDF(Cumulative Distribution Function) is the integral of the PDF. The CDF of the continuous variable $x$ is defined as:

$$P(x) = \int_{-\infty}^{x}p(x')dx'$$

Obviously, the CDF is a monotonically increasing function. And the PDF is the derivative of the CDF. So we can get the PDF from the CDF. And if we need to calculate the probability that $x$ lies in the range $[a, b]$, we can use the following equation:

$$p(a \leq x \leq b) = P(b) - P(a)$$
 
And it also have:

$$
\lim_{x \to -\infty}P(x) = 0\\
\lim_{x \to +\infty}P(x) = 1\\
$$

### 1.2.2 Expectations and Covariances

The expectation and variance of a function $f(x)$ with respect to the probability distribution $p(x)$ are defined as:

$$
E[f] = \int f(x)p(x)dx\\
E[f] \approx \frac{1}{N}\sum_{n=1}^{N}f(x_n)
$$

$$var[f] = E[(f(x) - E[f(x)])^2] = E[f(x)^2] - E[f(x)]^2$$

Expectation is the value to measure the center of the distribution. And variance is the value to measure the spread of the distribution. And the covariance between two functions $f(x)$ and $g(x)$ is defined as:

$$cov[f, g] = E[(f(x) - E[f(x)])(g(x) - E[g(x)])] = E[f(x)g(x)] - E[f(x)]E[g(x)]$$