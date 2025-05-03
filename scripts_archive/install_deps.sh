#!/bin/bash
set -e

# Install core packages first
pip install jupyter-book>=0.15.1
pip install jupytext>=1.14.0
pip install matplotlib
pip install numpy pandas scipy
pip install nbformat ipywidgets

# Install interactive book features
pip install sphinx-togglebutton sphinx-copybutton sphinx-comments sphinx-thebe
pip install sphinxcontrib-bibtex
pip install notebook

echo "Basic dependencies installed. You can now try building the book!"