# Deep Quantile Regression

![](log-data.png)

$$
\begin{aligned}

    E &= \left\|y - f(x)\right\| \\

    L_q &= \begin{cases}
        q  E ,     &    e \gt 0  \\
        (1 - q) E, &    e \lt 0
    \end{cases}

\end{aligned}
$$

$$
\begin{aligned}

    E &= y - f(x) \\

    L_q &= \max \begin{cases}
        q  E   \\
        (q - 1) E
    \end{cases}

\end{aligned}
$$

```python
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    loss = np.maximum(q * e, (q - 1.0) * e)

    return loss
```

```python
Linear(64)
gelu
Linear(3) # 3 output quantiles: (0.1, 0.5, 0.9)
```

![](quantiles-plot.png)