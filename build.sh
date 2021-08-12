#!/bin/env bash
python -m streambook main.py --nowatch
jupytext --to notebook --execute main.notebook.py --output main.ipynb
jupyter nbconvert --to markdown --output main.md main.ipynb
poetry export --without-hashes -f requirements.txt --output requirements.txt