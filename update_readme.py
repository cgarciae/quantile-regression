from pathlib import Path

path = Path("README.md")
text = path.read_text()

text = text.replace("![png](", "![png](https://raw.githubusercontent.com/cgarciae/quantile-regression/master/")

path.write_text(text)