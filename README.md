# Column-and-constraint-generation-for-L1-SVM-and-cousins
This is a solver for L1-SVM and related problems using linear program with column and constraint generations. For details see https://arxiv.org/abs/1901.01585. 

## Problems solved

We consider a family of regularized linear Support Vectors Machines problem with hinge-loss and convex sparsity-inducing regularization. In particular we study the L1-SVM problem:

$$ \min \sum_{i=1}^n (1 - y_i(x_i^T \beta +\beta_0))_+ + \lambda \\| \beta \\|_1 $$

the group-SVM problem:

$$ \min \sum_{i=1}^n (1 - y_i (x_i^T \beta +\beta_0))_+  + \lambda \sum_{g=1}^G \\| \beta_g \\|_{\infty} $$

```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \lambda \sum_{g=1}^G \| \beta_g \|_\inf,
```
and the Slope-SVM problem:
```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \sum_{j=1}^p \lambda_j | \beta_(j) |.
```
