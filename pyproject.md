---
layout: page
title: Python Projects
---


- Cross-Validated High-dimensional Conditional Density Estimation [[link]](/notes/amcv.html){:target="_blank" rel="noopener"}

   > - code of the first part of my job market paper
   > 
   > - A unified framework for estimating conditional density with *high-dimensional* covariates
   >   - new representation: conditional density `~` *many* conditional means
   >   - allowing for any machine learners of conditional means (e.g `sklearn`)
   >   - The estimator is fully data-driven, achieved through *cross-validation* 
   >   - new metric/loss for cross-validation, easy to implement
   >   - theoretical guarantee for the optimality


<br><br>
- Difference-in-Differences Models with Continuous Treatment (*Coming Soon!*)



<br><br>
- Approximate Sparsity Class and Minimax Estimation [[link]](/notes/minimax_series.html){:target="_blank" rel="noopener"}

   > - code my PhD second year paper, which establish various theoretical results of a new type of sparisty
   >   - particularly, complexity (metric entropy) and minimax rates are established
   >   - we show LASSO (as a selection mechanism) the density subset is still optimal
   >   - data-driven LASSO threshold is based on a maximal inequality
   >   - the code implements the theoretical results
   >   - post-processing algorithms and additional Monte-Carlo simulations are provded



<br><br>
- Introduction to Econometrics [[link]](/notes/103_all_codes.html){:target="_blank" rel="noopener"}

   
   > Jupyter Notebook I created when teaching the undergrad econometrics course at UCLA:
   > 
   > - basic data manipulations (`pandas` and `numpy`)
   > - linear regressions and hypothesis testing (`statsmodels`)
   > - basic plotting (`matplotlib`)
   > - special topics
   

<br><br>
- Consumer Preferences, Choices, and Counterfactuals [[link]](/notes/Urban_Replication_Project.html){:target="_blank" rel="noopener"}
   
   > PhD applied labor course assignment, implementation of BLP type of estimations in Python from scratch
   >  - contraction mapping
   >  - intrumental variable
   >  - counterfactual estimation
