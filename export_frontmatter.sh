#!/bin/bash
# Script to export frontmatter pages from JupyterBook to PDF

# Create output directories
mkdir -p pdf_exports/frontmatter

echo "Exporting frontmatter pages..."

# Export the frontmatter pages
jupyter-book build book/frontmatter/copyright.md --builder pdfhtml
jupyter-book build book/frontmatter/acknowledgments.md --builder pdfhtml
jupyter-book build book/frontmatter/about.md --builder pdfhtml

# Move the PDFs to the appropriate location
echo "Moving PDFs to pdf_exports/frontmatter directory..."
find book/_build/_page/frontmatter -name "*.pdf" -exec cp {} pdf_exports/frontmatter/ \;

# Copy to pdf_components for merging
mkdir -p pdf_components/frontmatter
cp pdf_exports/frontmatter/copyright.pdf pdf_components/supporting/01_copyright.pdf
cp pdf_exports/frontmatter/acknowledgments.pdf pdf_components/supporting/acknowledgments.pdf
cp pdf_exports/frontmatter/about.pdf pdf_components/supporting/about.pdf

echo "Frontmatter export complete. PDFs are in pdf_exports/frontmatter and pdf_components/supporting directories."