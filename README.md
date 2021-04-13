# Column-and-constraint-generation-for-L1-SVM-and-cousins
This is a solver for L1-SVM and related problems using linear program with column and constraint generations. For details see https://arxiv.org/abs/1901.01585. 

## Problems solved

We consider a family of regularized linear Support Vectors Machines problem with hinge-loss and convex sparsity-inducing regularization. In particular we study the L1-SVM problem:

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda \theta_k^{(j)}" />

$$ \min \sum_{i=1}^n \max (0, 1 - y_i(x_i^T \beta +\beta_0)) + \lambda \| \beta \|_1 $$


```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \lambda \| \beta \|_1,
```
the group-SVM problem:
```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \lambda \sum_{g=1}^G \| \beta_g \|_\inf,
```
and the Slope-SVM problem:
```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \sum_{j=1}^p \lambda_j | \beta_(j) |.
```
