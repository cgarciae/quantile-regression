#!/bin/env bash
streambook export main.py
jupytext --to notebook --execute main.notebook.py --output main.ipynb
rm -fr main_files
jupyter nbconvert --to markdown --output main.md main.ipynb
poetry export --without-hashes -f requirements.txt --output requirements.txt