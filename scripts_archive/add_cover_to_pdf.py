import os
import sys
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import glob

def add_page_number(page, page_number):
    """Add page number to the PDF page."""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    # Add page number at the bottom
    can.setFont('Helvetica', 10)
    can.drawCentredString(300, 30, str(page_number))
    can.save()
    
    packet.seek(0)
    number_pdf = PdfReader(packet)
    page.merge_page(number_pdf.pages[0])
    return page

def add_cover_to_pdf(cover_path, chapter_path, output_path):
    """Combine cover PDF with chapter PDF and add page numbers."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the PDFs
    try:
        cover_pdf = PdfReader(cover_path)
        chapter_pdf = PdfReader(chapter_path)
        
        # Create a PDF writer
        output = PdfWriter()
        
        # Add cover page (no page number on cover)
        output.add_page(cover_pdf.pages[0])
        
        # Add all pages from chapter with page numbers
        for i, page in enumerate(chapter_pdf.pages):
            # Page number starts from 1 after the cover
            page_with_number = add_page_number(page, i+1)
            output.add_page(page_with_number)
        
        # Write the combined PDF
        with open(output_path, "wb") as output_file:
            output.write(output_file)
            
        print(f"Created {output_path} with page numbers")
        return True
    except Exception as e:
        print(f"Error combining PDFs: {e}")
        return False

def process_all_pdfs(cover_path, pdf_dir, output_dir):
    """Process all PDFs in a directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDFs in the directory
    pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    success_count = 0
    
    for pdf_path in pdfs:
        # Skip the cover PDF if it's in the same directory
        if os.path.basename(pdf_path) == os.path.basename(cover_path):
            continue
            
        # Create output path
        output_path = os.path.join(output_dir, os.path.basename(pdf_path))
        
        # Add cover
        if add_cover_to_pdf(cover_path, pdf_path, output_path):
            success_count += 1
    
    print(f"Added cover to {success_count} PDFs. Results saved in {output_dir}")

if __name__ == "__main__":
    cover_path = "pdf_exports/cover.pdf"
    pdf_dir = "pdf_exports"
    output_dir = "pdf_exports/with_covers"
    
    process_all_pdfs(cover_path, pdf_dir, output_dir)