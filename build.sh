#!/bin/env bash
streambook export main.py
jupytext --to notebook --execute main.notebook.py --output main.ipynb
rm -fr main_files
jupyter nbconvert --to markdown --output README.md main.ipynb
python update_readme.py
poetry export --without-hashes -f requirements.txt --output requirements.txt