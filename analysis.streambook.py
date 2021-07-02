
import streamlit as __st
import streambook
__toc = streambook.TOCSidebar()
__st.markdown("""# Quantile Regression
![](log-data.png)""")
with __st.echo(), streambook.st_stdout('info'):
    import typing as tp

    import elegy
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax
    import typer
__toc.title('Quantile Regression')

__toc.generate()
