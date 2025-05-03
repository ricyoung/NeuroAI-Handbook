# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build JupyterBook HTML: `jb build book`
- Build specific chapter PDF: `jupyter-book build book/path/to/chapter.md --builder pdfhtml`
- Preview locally: `python -m http.server -d book/_build/html`
- Install dependencies: `pip install -r book/requirements.txt`
- Initialize environment: `python init_neuroai.py --all`
- Build complete PDF handbook: `python build_neuroai_handbook.py`
- Open the built PDF: `python open_pdf.py`

## PDF Build Process
The PDF build process uses these main components:
1. `01_build_chapters.py`: Builds individual chapter PDFs
2. `export_frontmatter.sh`: Exports frontmatter pages from JupyterBook (copyright, acknowledgments, about)
3. `02_build_supporting.py`: Creates supporting pages (cover, TOC, dividers)
4. `03_merge_final.py`: Combines everything into a single PDF

Custom assets can be placed in the `_assets` directory:
- For a custom cover page, place a `cover.pdf` file in the `_assets` directory
- For frontmatter content, edit the markdown files in `book/frontmatter/`

## Project Structure
- Content in `book/` directory
- Configure book settings in `book/_config.yml`
- Manage table of contents in `book/_toc.yml`
- Frontmatter content in `book/frontmatter/` directory
- Python code examples in `book/part1/` directory
- SVG figures in `book/figures/` organized by chapter
- Reference management with `book/references.bib`
- PDF build scripts in the root directory
- Custom assets in `_assets/` directory
- PDF outputs in `pdf_exports/` directory
- Build logs in `logs/` directory

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