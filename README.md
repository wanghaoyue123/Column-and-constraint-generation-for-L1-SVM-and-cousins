

# Column-and-constraint-generation-for-L1-SVM-and-cousins
This is a solver for L1-SVM and related problems using linear program with column and constraint generations. For details see https://arxiv.org/abs/1901.01585. 

## Problems solved

We consider a family of regularized linear Support Vectors Machines problem with hinge-loss and convex sparsity-inducing regularization. In particular we study 


<img width="500" height="200" src="https://github.com/wanghaoyue123/Column-and-constraint-generation-for-L1-SVM-and-cousins/blob/main/img-formula.png"/>



![image](https://github.com/wanghaoyue123/Column-and-constraint-generation-for-L1-SVM-and-cousins/blob/main/img-formula.png)

$$\text{L1-SVM} ~~~~ \min_{\beta\in R^p, ~\beta_0 \in R} ~ \sum_{i=1}^n (1 - y_i(x_i^T \beta +\beta_0))_+ + \lambda \\| \beta \\|_1 $$

$$\text{Group-SVM} ~~~~ \min_{\beta\in R^p, ~\beta_0 \in R} ~ \sum_{i=1}^n (1 - y_i (x_i^T \beta +\beta_0))_+ + \sum\_{g=1}^G \\| \beta\_g \\|\_{\infty} $$

$$
\text{Slope-SVM} ~~~~ \min_{\beta\in R^p, ~\beta_0 \in R} ~ \sum_{i=1}^n (1 - y_i (x_i^T \beta +\beta_0))_+  + \sum\_{j=1}^p \lambda_j | \beta\_{(j)} |
$$
where $| \beta\_{(1)} | \ge | \beta\_{(2)} | \ge \cdots \ge | \beta\_{(p)} |  $. 




## Examples

See the Main folder for a few examples:

1. example_L1_SVM_n_small_p_large.ipynb: This is an example of L1-SVM when n is small and p is large. 

2. example_L1_SVM_n_large_p_small.ipynb: This is an example of L1-SVM when n is large and p is small. 

3. example_L1_SVM_n_large_p_large.ipynb: This is an example of L1-SVM when both n and p are large. 

4. example_group_SVM_n_small_p_large.ipynb: This is an example of Group-SVM when n is small and p is large. 

5. example_slope_SVM_n_small_p_large.ipynb: This is an example of Slope-SVM when n is small and p is large. 





