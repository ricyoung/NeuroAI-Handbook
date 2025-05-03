#!/bin/bash
# Script to add cover pages to chapter PDFs

# Check if pdftk is installed
if ! command -v pdftk &> /dev/null; then
    echo "pdftk is not installed. Please install it to combine PDFs."
    echo "On macOS, run: brew install pdftk-java"
    echo "On Ubuntu/Debian, run: sudo apt-get install pdftk"
    exit 1
fi

# Create output directory for PDFs with covers
mkdir -p pdf_exports/with_covers

# First, run the regular PDF export if PDFs don't exist
if [ ! -d "pdf_exports" ] || [ -z "$(ls -A pdf_exports)" ]; then
    echo "No PDFs found. Running export_pdfs.sh first..."
    bash export_pdfs.sh
fi

# Create cover page if it doesn't exist
if [ ! -f "pdf_exports/cover.pdf" ]; then
    echo "No cover page found. Creating one..."
    bash create_cover_pdf.sh
fi

# For each PDF in the pdf_exports directory
for pdf in pdf_exports/*.pdf; do
    if [[ "$pdf" == "pdf_exports/cover.pdf" ]] || [[ "$pdf" == "pdf_exports/with_covers/"* ]]; then
        continue
    fi
    
    filename=$(basename "$pdf")
    echo "Adding cover to $filename..."
    
    # Combine cover with chapter PDF
    pdftk pdf_exports/cover.pdf "$pdf" cat output "pdf_exports/with_covers/$filename"
    
    echo "Created pdf_exports/with_covers/$filename with cover page"
done

echo "All PDFs have been processed. PDFs with covers are in pdf_exports/with_covers/"