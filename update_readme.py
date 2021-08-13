from pathlib import Path

path = Path("main.md")
text = path.read_text()

text = text.replace("![png](", "![png](https://raw.githubusercontent.com/cgarciae/quantile-regression/master/")

Path("README.md").write_text(text)