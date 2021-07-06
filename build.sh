python -m streambook main.py --nowatch
jupytext --to notebook --execute main.notebook.py --output main.ipynb
jupytext --to markdown --execute main.notebook.py --output README.md # use nbconvert instead to save computation
poetry export -f requirements.txt --output requirements.txt