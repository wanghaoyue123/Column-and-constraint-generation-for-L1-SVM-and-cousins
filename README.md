# Column-and-constraint-generation-for-L1-SVM-and-cousins
This is a solver for L1-SVM and related problems using linear program with column and constraint generations. For details see https://arxiv.org/abs/1901.01585. 

## Problems solved

We consider a family of regularized linear Support Vectors Machines problem with hinge-loss and convex sparsity-inducing regularization. In particular we study 

$$\text{L1-SVM} ~~~~ \min_{\beta\in R^p, ~\beta_0 \in R} ~ \sum_{i=1}^n (1 - y_i(x_i^T \beta +\beta_0))_+ + \lambda \\| \beta \\|_1 $$

$$\text{Group-SVM} ~~~~ \min_{\beta\in R^p, ~\beta_0 \in R} ~ \sum_{i=1}^n (1 - y_i (x_i^T \beta +\beta_0))_+ + \sum\_{g=1}^G \\| \beta\_g \\|\_{\infty} $$

$$
\text{Slope-SVM} ~~~~ \min_{\beta\in R^p, ~\beta_0 \in R} ~ \sum_{i=1}^n (1 - y_i (x_i^T \beta +\beta_0))_+  + \sum\_{j=1}^p \lambda_j | \beta\_{(j)} |,
$$
where $| \beta\_{(1)} | \le  $

