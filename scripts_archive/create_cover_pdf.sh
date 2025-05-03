#!/bin/bash
# Script to create a cover page PDF

# Check if wkhtmltopdf is installed
if ! command -v wkhtmltopdf &> /dev/null; then
    echo "wkhtmltopdf is not installed. Please install it to create cover pages."
    echo "On macOS, run: brew install wkhtmltopdf"
    echo "On Ubuntu/Debian, run: sudo apt-get install wkhtmltopdf"
    exit 1
fi

# Create the cover page PDF
echo "Creating cover page PDF..."
wkhtmltopdf --enable-local-file-access book/cover.html pdf_exports/cover.pdf

echo "Cover page PDF created at pdf_exports/cover.pdf"