# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build JupyterBook: `jb build book`
- Build specific chapter: `jupyter-book build book/path/to/chapter.md --builder pdfhtml`
- Preview locally: `python -m http.server -d book/_build/html`
- Install dependencies: `pip install -r book/requirements.txt`
- Initialize environment: `python init_neuroai.py --all`
- Export PDFs: `bash export_pdfs.sh`

## Project Structure
- Content in `book/` directory
- Configure book settings in `book/_config.yml`
- Manage table of contents in `book/_toc.yml`
- Python code examples in `book/part1/` directory
- SVG figures in `book/figures/` organized by chapter
- Reference management with `book/references.bib`

## Code Style Guidelines
- Python: Follow PEP 8 style guidelines
- Type hints for all function parameters and return values
- Docstrings: NumPy style with Parameters/Returns sections
- Error handling: Use try/except with specific exceptions
- Imports order: standard library → third-party → local modules
- Naming: snake_case for functions/variables, CamelCase for classes
- Jupyter notebooks: Clear outputs before committing
- Markdown: ATX-style headers (# for headings)
- Citations: Use bibtex format in references.bib