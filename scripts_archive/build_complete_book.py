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

def create_complete_book():
    """Build a complete book from all chapter PDFs."""
    # Ensure output directory exists
    os.makedirs("pdf_exports", exist_ok=True)
    
    # Path to the cover page
    cover_path = "pdf_exports/cover.pdf"
    output_path = "pdf_exports/neuroai_handbook_complete.pdf"
    
    # Get all chapter PDFs
    pdf_dir = "pdf_exports"
    pdf_files = glob.glob(os.path.join(pdf_dir, "ch*.pdf"))
    
    # Print all available chapter PDFs for debugging
    print("Available chapter PDFs:")
    for pdf in sorted(pdf_files):
        print(f"  - {os.path.basename(pdf)}")
    
    # Sort files in correct order
    sorted_files = []
    
    # First get introduction
    intro_file = next((f for f in pdf_files if "ch01_intro" in f), None)
    if intro_file:
        sorted_files.append(intro_file)
        pdf_files.remove(intro_file)
    
    # Get chapter number for sorting
    def get_chapter_num(filename):
        basename = os.path.basename(filename)
        # Extract the chapter number (e.g., "ch01" -> 1, "ch15" -> 15)
        chapter_part = basename.split('_')[0]  # Get "ch01", "ch15", etc.
        if chapter_part.startswith('ch'):
            try:
                return int(chapter_part[2:])  # Extract number from "ch01" -> 1
            except ValueError:
                return 999  # If not a valid number, put at end
        return 999  # Default to end if not matching pattern
    
    # Sort all remaining PDFs by chapter number
    remaining_files = sorted(pdf_files, key=get_chapter_num)
    sorted_files.extend(remaining_files)
    
    print("\nChapters in order:")
    for pdf in sorted_files:
        print(f"  - {os.path.basename(pdf)}")
    
    # Now we have all the files in the right order
    try:
        # Get the cover
        cover_pdf = PdfReader(cover_path)
        
        # Create PDF writer for output
        output = PdfWriter()
        
        # Add cover (no page number)
        output.add_page(cover_pdf.pages[0])
        
        # Track the current page number
        current_page = 1
        
        # Add all PDFs in correct order
        for pdf_path in sorted_files:
            print(f"Adding {os.path.basename(pdf_path)} to the complete book")
            pdf = PdfReader(pdf_path)
            
            for page in pdf.pages:
                # Add page number
                page_with_number = add_page_number(page, current_page)
                output.add_page(page_with_number)
                current_page += 1
        
        # Write the final PDF
        with open(output_path, "wb") as output_file:
            output.write(output_file)
        
        print(f"\nComplete book created at {output_path}")
        print(f"Total pages: {current_page}")
        return True
        
    except Exception as e:
        print(f"Error creating complete book: {e}")
        return False

if __name__ == "__main__":
    create_complete_book()