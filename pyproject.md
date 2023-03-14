---
layout: page
title: Projects (Code in Python)
---

<br/>
**Cross-Validated High-dimensional Conditional Density Estimation** [[link]](/notes/amcv.html){:target="_blank" rel="noopener"}

- A unified framework for estimating conditional density with **high-dimensional** covariates

  - new representation: conditional density `~` *many* conditional means
  - allowing for any machine learners of conditional means (e.g `sklearn`)
  - The estimator is fully **data-driven**, achieved through *cross-validation* 
  - new metric/loss for cross-validation, easy to implement
  - theoretical guarantee for the optimality

<br/>
**Difference-in-Differences Models with Continuous Treatment** [[link]](/notes/Continuous_DiD.html){:target="_blank" rel="noopener"}

- Extending the diff-in-diff framework to **continuous treatment**

  - estimating average treatment effect on treated (ATT) at any continuous treatment intensity
  - under double/debiased machine learning (DML) framework: debiased score + crossfitting
  - can accommodate high-dimensional covariates
  - the estimator is asymptotically normal with explicit asymptotic variance 
  - bonus multiplier bootstrap confidence interval

<br/>
**Approximate Sparsity Class and Minimax Estimation** [[link]](/notes/minimax_series.html){:target="_blank" rel="noopener"}

- Proposing a new type of sparisty: **approximate sparsity**

  - complexity (metric entropy) and minimax rates are established
  - LASSO (as a selection mechanism) is still optimal
  - data-driven LASSO threshold based on a maximal (Talagrand's) inequality
  - simple code that implements the theoretical results
  - post-processing algorithm and additional Monte-Carlo simulations are provded

<br/>
**Consumer Preferences, Choices, and Counterfactuals** [[link]](/notes/Urban_Replication_Project.html){:target="_blank" rel="noopener"}

- Implementation of Bayer, Ferreira, and McMillan (2007) in Python

  - combination of **BLP** and intrumental variable
  - contraction mapping
  - counterfactual estimation

<br/>
**Introduction to Econometrics** [[link]](/notes/103_all_codes.html){:target="_blank" rel="noopener"}

- Jupyter Notebook for undergraduate econometrics course at UCLA

  - basic data manipulations (`pandas` and `numpy`)
  - linear regressions and hypothesis testing (`statsmodels`)
  - basic plotting (`matplotlib`)
  - special topic(s) (diff-in-diff)
   
