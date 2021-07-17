python -m streambook main.py --nowatch
jupytext --to notebook --execute main.notebook.py --output main.ipynb
jupyter nbconvert --to markdown --output main.md main.ipynb
cp main.md README.md
poetry export -f requirements.txt --output requirements.txt