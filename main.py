import typing as tp

import elegy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
import optax


class QuantileNet(elegy.Module):
    def call(self, x):

        x = elegy.nn.Linear(64)(x)
        x = jax.nn.gelu(x)
        x = elegy.nn.Linear(3)(x)

        return x


class QuantileLoss(elegy.Loss):
    def __init__(self, quantiles: tp.Sequence[float]):
        super().__init__()
        self.quantiles = np.array(quantiles)

    def call(self, y_true, y_pred):
        loss = jax.vmap(self.loss_fn, in_axes=(0, None, -1), out_axes=1)(
            self.quantiles, y_true[:, 0], y_pred
        )
        return jnp.sum(loss, axis=-1)

    def loss_fn(self, q, y_true, y_pred):
        e = y_true - y_pred
        loss = jnp.maximum(q * e, (q - 1.0) * e)

        return loss


def main(visualize: bool = False, eager: bool = False):

    x = np.random.uniform(0.3, 10, 1000)
    y = np.log(x) + np.random.exponential(0.1 + x / 20.0)

    if visualize:
        plt.scatter(x, y, s=20, facecolors="none", edgecolors="k")
        plt.show()

    x = x[..., None]
    y = y[..., None]

    quantiles = [0.1, 0.5, 0.90]
    model = elegy.Model(
        QuantileNet(),
        loss=QuantileLoss(quantiles),
        optimizer=optax.adamw(1e-3),
        run_eagerly=eager,
    )
    model.init(x, y)
    model.summary(x)

    model.fit(x, y, epochs=1000)

    x_test = np.linspace(x.min(), x.max(), 100)

    y_pred = model.predict(x_test[..., None])
    plt.scatter(x, y, s=20, facecolors="none", edgecolors="k")

    for i, q_values in enumerate(np.split(y_pred, 3, axis=-1)):
        plt.plot(x_test, q_values[:, 0], linewidth=2, label=f"Q({quantiles[i]})")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    typer.run(main)
