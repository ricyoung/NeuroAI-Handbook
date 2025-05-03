import os
import yaml
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import io

def create_toc_pdf(toc_yml_path, output_path):
    """Create a table of contents PDF from a _toc.yml file."""
    # Load TOC structure
    with open(toc_yml_path, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Create a new PDF with ReportLab
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    width, height = letter
    
    # Background color (light gray)
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(0, 0, width, height, fill=True)
    
    # Title
    c.setFillColorRGB(0.1, 0.1, 0.6)  # Dark blue
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-1.5*inch, "Table of Contents")
    
    # Get the parts
    parts = toc_data.get('parts', [])
    
    # Starting position
    y_pos = height - 2.5*inch
    part_num = 1
    
    # Draw each part and its chapters
    for part in parts:
        # Skip the cover part
        if part.get('caption') == 'Cover':
            continue
            
        # Part title
        c.setFillColorRGB(0.1, 0.1, 0.6)  # Dark blue for part title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, y_pos, f"Part {part_num}: {part.get('caption', '')}")
        y_pos -= 0.5*inch
        
        # Chapters
        for i, chapter in enumerate(part.get('chapters', [])):
            c.setFillColorRGB(0.3, 0.3, 0.3)  # Dark gray for chapters
            c.setFont("Helvetica", 12)
            
            # Get the chapter title from the file name if not specified
            file_path = chapter.get('file', '')
            title = chapter.get('title', '')
            if not title and file_path:
                # Extract chapter name from file path (e.g., "ch01_intro" -> "Introduction")
                base_name = os.path.basename(file_path)
                # Remove the chapter number prefix (ch01_, etc.)
                if base_name.startswith('ch') and '_' in base_name:
                    title = base_name.split('_', 1)[1].replace('_', ' ').title()
                else:
                    title = base_name.replace('_', ' ').title()
            
            c.drawString(1.5*inch, y_pos, title)
            y_pos -= 0.3*inch
            
            # Check if we need to start a new page
            if y_pos < 1*inch:
                c.showPage()
                # Reset the position
                y_pos = height - 1*inch
                
                # Background color (light gray)
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(0, 0, width, height, fill=True)
        
        # Move to next part
        part_num += 1
        y_pos -= 0.5*inch
        
        # Check if we need to start a new page
        if y_pos < 1*inch:
            c.showPage()
            # Reset the position
            y_pos = height - 1*inch
            
            # Background color (light gray)
            c.setFillColorRGB(0.95, 0.95, 0.95)
            c.rect(0, 0, width, height, fill=True)
    
    # Save the PDF
    c.save()
    
    # Move to the beginning of the StringIO object
    packet.seek(0)
    
    # Create a new PDF with the TOC
    toc_pdf = PdfReader(packet)
    
    # Write the TOC PDF
    output = PdfWriter()
    for page in toc_pdf.pages:
        output.add_page(page)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        output.write(f)
    
    print(f"Table of Contents PDF created at {output_path}")
    return output_path

def insert_toc_into_book(toc_path, book_path, output_path):
    """Insert the TOC after the cover page in the complete book."""
    try:
        # Read the book PDF
        book_pdf = PdfReader(book_path)
        
        # Read the TOC PDF
        toc_pdf = PdfReader(toc_path)
        
        # Create a new PDF
        output = PdfWriter()
        
        # Add the cover page (first page)
        output.add_page(book_pdf.pages[0])
        
        # Add the TOC pages
        for page in toc_pdf.pages:
            output.add_page(page)
        
        # Add the rest of the book
        for i in range(1, len(book_pdf.pages)):
            output.add_page(book_pdf.pages[i])
        
        # Write the output PDF
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            output.write(f)
        
        print(f"Complete book with TOC created at {output_path}")
        return True
    except Exception as e:
        print(f"Error inserting TOC: {e}")
        return False

if __name__ == "__main__":
    # Create TOC PDF
    toc_yml_path = "book/_toc.yml"
    toc_pdf_path = "pdf_exports/toc.pdf"
    create_toc_pdf(toc_yml_path, toc_pdf_path)
    
    # Insert TOC into complete book
    book_path = "pdf_exports/with_covers/neuroai_handbook_complete.pdf"
    output_path = "pdf_exports/with_covers/neuroai_handbook_complete_with_toc.pdf"
    insert_toc_into_book(toc_pdf_path, book_path, output_path)