---
title: Quantile Regression
tags: presentation
slideOptions:
  theme: white
  transition: 'fade'
---
<style>
.reveal section img { background:none; border:none; box-shadow:none; }
</style>

# Quantile Regression
A simple method to estimate uncertainty in Machine Learning

---

## Why estimate uncertainty?

* Get bounds for the data.
* Estimate the distribution of the output.
<span>
* **Reduce Risk**.<!-- .element: class="fragment" data-fragment-index="1" -->
</span>

---

## Problem

<img src="https://raw.githubusercontent.com/cgarciae/quantile-regression/master/main_files/main_1_0.png" height="500" />

---

## Problem
1. It is not normally distributed.
2. Noise it not symetric. 
3. Its variance is not constant.

---

## Solution
Estimate uncertainty by predicting the <br> quantiles of $y$ given $x$.


---

## Quantile Loss

$$
\begin{aligned}
    E &= y - f(x) \\
    L_q &= \begin{cases}
        q  E,     &    E \gt 0  \\
        (1 - q) (-E), &    E \lt 0
    \end{cases}
\end{aligned}
$$

---

## Quantile Loss

$$
\begin{aligned}
    E &= y - f(x) \\
    L_q &= \max \begin{cases}
        q  E   \\
        (q - 1) E
    \end{cases}
\end{aligned}
$$

---

## JAX Implementation
```python

    def quantile_loss(q, y_true, y_pred):
        e = y_true - y_pred
        return jnp.maximum(q * e, (q - 1.0) * e)

```

---

<img src="https://raw.githubusercontent.com/cgarciae/quantile-regression/master/main_files/main_5_1.png" height="550">
    
**Loss landscape** for a continous sequence of `y_true` values  between `[10, 20]`.
<!-- .element: style="font-size: 20px;" -->

---

<!-- ## Deep Quantile Regression -->
```python

    class QuantileRegression(elegy.Module):
        def __init__(self, n_quantiles: int):
            super().__init__()
            self.n_quantiles = n_quantiles

        def call(self, x):
            x = elegy.nn.Linear(128)(x)
            x = jax.nn.relu(x)
            x = elegy.nn.Linear(64)(x)
            x = jax.nn.relu(x)
            x = elegy.nn.Linear(self.n_quantiles)(x)
            
            return x
```

---

<img src="https://raw.githubusercontent.com/cgarciae/quantile-regression/master/main_files/main_13_0.png">

---

<img src="https://raw.githubusercontent.com/cgarciae/quantile-regression/master/main_files/main_15_0.png">

---

<img src="https://raw.githubusercontent.com/cgarciae/quantile-regression/master/main_files/main_19_0.png">
    

---

## Recap
* Quantile Regression: simple and effective.
* Use when risk management is needed.
* Neural Networks are an efficient way to predict multiple quantiles.
* With sufficient quantiles you can approximate the density function.
<!-- .element: style="font-size: 36px;" -->

---

## Next Steps
* Check out the blog and repo
    * Blog: BLOG_URL
    * Repo: [cgarciae/quantile-regression](https://github.com/cgarciae/quantile-regression)
* Take a look at Mixture Density Networks.
* Learn more about [jax]("https://github.com/google/jax) and [elegy]("https://github.com/poets-ai/elegy).
