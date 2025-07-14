---
layout: page
title: Research
---
<br/>
**Working Papers**

* Continuous Difference-in-Differences with Double/Debiased Machine Learning. [[link]](https://arxiv.org/abs/2408.10509){:target="_blank" rel="noopener"} *(Accepted for Publication.)*

   **Abstract**: This paper extends difference-in-differences to settings with continuous treatments. Specifically, the average treatment effect on the treated (ATT) at any level of treatment intensity is identified under a conditional parallel trends assumption. Estimating the ATT in this framework requires first estimating infinite-dimensional nuisance parameters, particularly the conditional density of the continuous treatment, which can introduce substantial bias. To address this challenge, we propose estimators for the causal parameters under the double/debiased machine learning framework and establish their asymptotic normality. Additionally, we provide consistent variance estimators and construct uniform confidence bands based on a multiplier bootstrap procedure. To demonstrate the effectiveness of our approach, we apply our estimators to the 1983 Medicare Prospective Payment System (PPS) reform studied by [Acemoglu and Finkelstein (2008)](https://economics.mit.edu/sites/default/files/publications/Input%20and%20Technology%20Choices%20in%20Regulated%20Industri.pdf){:target="_blank" rel="noopener"}, reframing it as a DiD with continuous treatment and nonparametrically estimating its effects.
   
* Difference-in-Differences with Time-Varying Continuous Treatments Using Double/Debiased Machine Learning. [[link]](https://arxiv.org/abs/2410.21105){:target="_blank" rel="noopener"} Joint work with [Michel F. C. Haddad](https://www.qmul.ac.uk/sbm/staff/academic/profiles/haddadm.html){:target="_blank" rel="noopener"} and [Martin Huber](https://www.unifr.ch/directory/en/people/7260/c8d1a){:target="_blank" rel="noopener"}. 

  **Abstract**: We propose a diﬀerence-in-diﬀerences (DiD) method for a time-varying continuous treatment and multiple time periods. Our framework assesses the average treatment eﬀect on the treated (ATET) when comparing two non-zero treatment doses. The identification is based on a conditional parallel trend assumption imposed on the mean potential outcome under the lower dose, given observed covariates and past treatment histories. We employ kernel-based ATET estimators for repeated cross-sections and panel data adopting the double/debiased machine learning framework to control for covariates and past treatment histories in a data-adaptive manner. We also demonstrate the asymptotic normality of our estimation approach under specific regularity conditions. In a simulation study, we find a compelling finite sample performance of undersmoothed versions of our estimators in setups with several thousand observations.

* Approximate Sparsity Class and Minimax Estimation. [[link]](/notes/minimax_joe.pdf){:target="_blank" rel="noopener"}

   **Abstract**: Motivated by the orthogonal series density estimation in $L^2([0,1],\mu)$, in this project we consider a new class of functions that we call the approximate sparsity class. This new class is characterized by the rate of decay of the individual Fourier coefficients for a given orthonormal basis. We establish the $L^2([0,1],\mu)$ metric entropy of such class, with which we show the minimax rate of convergence. For the density subset in this class, we propose an adaptive density estimator based on a hard-thresholding procedure that achieves this minimax rate up to a $\log$ term.

<br/>
**Working in Progress**

* An Oracle for Data-Driven High-Dimensional Conditional Density Estimation. *(Forthcoming!)*
