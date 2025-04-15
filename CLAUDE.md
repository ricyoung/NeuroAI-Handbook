# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build JupyterBook: `jb build book`
- Preview locally: `python -m http.server -d book/_build/html`
- Install dependencies: `pip install -r book/requirements.txt`

## Project Structure
- Content in `book/` directory
- Configure book settings in `book/_config.yml`
- Manage table of contents in `book/_toc.yml`
- Python code in `book/part1/` for demonstrations
- Reference management with `book/references.bib`

## Code Style Guidelines
- Python: Follow PEP 8 style guidelines
- Jupyter notebooks: Clear outputs before committing
- Use type hints for Python functions
- Markdown files: Use ATX-style headers (# for titles)
- Imports: Standard library first, then third-party, then local
- Naming: snake_case for functions/variables, CamelCase for classes
- Citations: Use bibtex format in references.bib