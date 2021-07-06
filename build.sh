python -m streambook main.py --nowatch
jupytext --to notebook --execute main.notebook.py --output main.notebook.ipynb
jupyter nbconvert --to markdown --output README.md main.notebook.ipynb
poetry export -f requirements.txt --output requirements.txt