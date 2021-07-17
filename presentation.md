# Quantile Regression
Estimating uncertainty in Machine Learning

---

## Why estimate uncertainty?

* Get bounds for the data.
* Estimate the distribution of the output.
<span>
* **Reduce Risk**.<!-- .element: class="fragment" data-fragment-index="1" -->
</span>

---

## Problem
![png](https://raw.githubusercontent.com/cgarciae/quantile-regression/master/README_files/README_1_0.png)
    

---

## Properties
1. It is not normally distributed.
2. Noise it not symetric. 
3. Its variance is not constant.

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

## Loss Landscape
![png](https://raw.githubusercontent.com/cgarciae/quantile-regression/master/README_files/README_5_1.png)
    
For a sequence of values of `y_true` between `[10, 20]`.

## Deep Quantile Regression

Generally you would have to create a model per quantile, however if we use a neural network we can have it output the predictions for all the quantiles at the same time. Here will use `elegy` to create a neural network with 2 hidden layers with `relu` activations and a linear layers with `n_quantiles` output units.


```python
import elegy


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

Now we are going to properly define a `QuantileLoss` class that is parameterized by
a set of user defined `quantiles`.


```python


class QuantileLoss(elegy.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = np.array(quantiles)

    def call(self, y_true, y_pred):
        loss = jax.vmap(quantile_loss, in_axes=(0, None, -1), out_axes=1)(
            self.quantiles, y_true[:, 0], y_pred
        )
        return jnp.sum(loss, axis=-1)
```

Notice that we use the same `quantile_loss` that we created previously along with some `jax.vmap` magic to properly vectorize the function. Finally we are going to create a simple function that creates and trains our model for a set of quantiles using `elegy`.


```python
import optax


def train_model(quantiles, epochs: int, lr: float, eager: bool):
    model = elegy.Model(
        QuantileRegression(n_quantiles=len(quantiles)),
        loss=QuantileLoss(quantiles),
        optimizer=optax.adamw(lr),
        run_eagerly=eager,
    )
    model.init(x, y)
    model.summary(x)

    model.fit(x, y, epochs=epochs, batch_size=64, verbose=0)

    return model


if not multimodal:
    quantiles = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
else:
    quantiles = np.linspace(0.05, 0.95, 9)

model = train_model(quantiles=quantiles, epochs=3001, lr=1e-4, eager=False)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer                        </span>┃<span style="font-weight: bold"> Outputs Shape        </span>┃<span style="font-weight: bold"> Trainable        </span>┃<span style="font-weight: bold"> Non-trainable </span>┃
┃                              ┃                      ┃<span style="font-weight: bold"> Parameters       </span>┃<span style="font-weight: bold"> Parameters    </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Inputs                       │ (1000, 1)    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">float64</span> │                  │               │
├──────────────────────────────┼──────────────────────┼──────────────────┼───────────────┤
│ linear    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Linear</span>             │ (1000, 128)  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">float32</span> │ <span style="color: #008000; text-decoration-color: #008000">256</span>      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1.0 KB</span>  │               │
├──────────────────────────────┼──────────────────────┼──────────────────┼───────────────┤
│ linear_1  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Linear</span>             │ (1000, 64)   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">float32</span> │ <span style="color: #008000; text-decoration-color: #008000">8,256</span>    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">33.0 KB</span> │               │
├──────────────────────────────┼──────────────────────┼──────────────────┼───────────────┤
│ linear_2  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Linear</span>             │ (1000, 7)    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">float32</span> │ <span style="color: #008000; text-decoration-color: #008000">455</span>      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1.8 KB</span>  │               │
├──────────────────────────────┼──────────────────────┼──────────────────┼───────────────┤
│ *         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">QuantileRegression</span> │ (1000, 7)    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">float32</span> │                  │               │
├──────────────────────────────┼──────────────────────┼──────────────────┼───────────────┤
│<span style="font-weight: bold">                              </span>│<span style="font-weight: bold">                Total </span>│<span style="font-weight: bold"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">8,967</span><span style="font-weight: bold">    </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">35.9 KB</span><span style="font-weight: bold"> </span>│<span style="font-weight: bold">               </span>│
└──────────────────────────────┴──────────────────────┴──────────────────┴───────────────┘
<span style="font-weight: bold">                                                                                          </span>
<span style="font-weight: bold">                            Total Parameters: </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">8,967</span><span style="font-weight: bold">   </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">35.9 KB</span><span style="font-weight: bold">                             </span>
</pre>



    
    


Now that we have a model lets generate some test data that spans the entire domain and compute the predicted quantiles.


```python
x_test = np.linspace(x.min(), x.max(), 100)
y_pred = model.predict(x_test[..., None])

plt.scatter(x, y, s=20, facecolors="none", edgecolors="k")

for i, q_values in enumerate(np.split(y_pred, len(quantiles), axis=-1)):
    plt.plot(x_test, q_values[:, 0], linewidth=2, label=f"Q({quantiles[i]:.2f})")

plt.legend()
plt.show()
```


    
![png](https://raw.githubusercontent.com/cgarciae/quantile-regression/master/README_files/README_13_0.png)
    


Amazing! Notice how the first few quantiles are tightly packed together while the last ones spread out capturing the behavior of the exponential distribution. 

Having the quantile values also allows you to estimate the density of the data, since the difference between two adjacent quantiles represent the probability that a point lies between them, we can construct a piecewise function that approximates the density of the data.


```python
def get_pdf(quantiles, q_values):
    densities = []

    for i in range(len(quantiles) - 1):
        area = quantiles[i + 1] - quantiles[i]
        b = q_values[i + 1] - q_values[i]
        a = area / b

        densities.append(a)

    return densities


def piecewise(xs):
    return [xs[i + j] for i in range(len(xs) - 1) for j in range(2)]


def doubled(xs):
    return [np.clip(xs[i], 0, 3) for i in range(len(xs)) for _ in range(2)]
```

Now for a given `x` we can compute the quantile values and then use these to compute the conditional piecewise density function of `y` given `x`.


```python
xi = 7.0

q_values = model.predict(np.array([[xi]]))[0].tolist()

densities = get_pdf(quantiles, q_values)

plt.title(f"x = {xi}")
plt.fill_between(piecewise(q_values), 0, doubled(densities))
# plt.fill_between(q_values, 0, densities + [0])
# plt.plot(q_values, densities + [0], color="k")
plt.xlim(0, y.max())
plt.gca().set_xlabel("y")
plt.gca().set_ylabel("p(y)")
plt.show()
```


    
![png](https://raw.githubusercontent.com/cgarciae/quantile-regression/master/README_files/README_17_0.png)
    


One of the nice properties of Quantile Regression is that we did not need to know a priori the output distribution and training is easy in comparison to other methods.

## Recap
* Quantile Regression is a simple and effective method for learning some statistics
about the output distribution.
* It is specially useful to stablish bounds on the predictions of a model when risk management is desired.
* The Quantile Loss function is simple and easy to implement.
* Quantile Regression can be efficiently implemented in using Neural Networks since a single model can be used to predict all the quantiles.
* The quantiles can be used to estimate the conditional density of the data.

## Next Steps
* Try running this notebook with `multimodal = True`.
* Take a look at Mixture Density Networks.
* Learn more about [jax](https://github.com/google/jax) and [elegy](https://github.com/poets-ai/elegy).
