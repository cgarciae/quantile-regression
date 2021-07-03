jupytext --to notebook --execute main.notebook.py
jupytext --to markdown --execute main.notebook.py --output README.md
poetry export -f requirements.txt --output requirements.txt