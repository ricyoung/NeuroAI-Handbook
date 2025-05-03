import os
import sys
import yaml
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
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
        part_caption = part.get('caption', '').replace(' · ', ': ')
        c.drawString(1*inch, y_pos, f"Part {part_num}: {part_caption}")
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
            
            c.drawString(1.5*inch, y_pos, title or os.path.basename(file_path))
            y_pos -= 0.3*inch
            
            # Add any sections
            if 'sections' in chapter:
                for section in chapter.get('sections', []):
                    c.setFont("Helvetica-Oblique", 10)
                    section_title = section.get('title', os.path.basename(section.get('file', '')))
                    c.drawString(2*inch, y_pos, section_title)
                    y_pos -= 0.3*inch
                    
                    # Check if we need to start a new page
                    if y_pos < 1*inch:
                        c.showPage()
                        # Reset the position
                        y_pos = height - 1*inch
                        # Background color (light gray)
                        c.setFillColorRGB(0.95, 0.95, 0.95)
                        c.rect(0, 0, width, height, fill=True)
            
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

def create_complete_book_in_toc_order():
    """Build a complete book from chapter PDFs in the exact order specified in _toc.yml."""
    # Ensure output directory exists
    os.makedirs("pdf_exports", exist_ok=True)
    os.makedirs("pdf_exports/with_covers", exist_ok=True)
    
    # Path to the cover page and files
    cover_path = "pdf_exports/cover.pdf"
    toc_yml_path = "book/_toc.yml"
    output_path = "pdf_exports/neuroai_handbook_complete_ordered.pdf"
    toc_path = "pdf_exports/toc.pdf"
    final_output = "pdf_exports/with_covers/neuroai_handbook_complete_final.pdf"
    
    # Create TOC PDF
    create_toc_pdf(toc_yml_path, toc_path)
    
    # Load the TOC structure to get the exact order
    with open(toc_yml_path, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Extract all chapter files in order from TOC
    chapter_files = []
    for part in toc_data.get('parts', []):
        for chapter in part.get('chapters', []):
            # Skip the cover
            if chapter.get('file') == 'cover':
                continue
                
            file_path = chapter.get('file', '')
            if file_path:
                # Convert path from TOC format to PDF filename format
                base_name = os.path.basename(file_path)
                pdf_path = f"pdf_exports/{base_name}.pdf"
                chapter_files.append((file_path, pdf_path))
            
            # Process sections if any
            if 'sections' in chapter:
                for section in chapter.get('sections', []):
                    section_path = section.get('file', '')
                    if section_path:
                        # Convert path from TOC format to PDF filename format
                        base_name = os.path.basename(section_path)
                        pdf_path = f"pdf_exports/{base_name}.pdf"
                        chapter_files.append((section_path, pdf_path))
    
    # Print the order for verification
    print("Chapters in TOC order:")
    for i, (toc_path, pdf_path) in enumerate(chapter_files):
        print(f"  {i+1}. {toc_path} -> {pdf_path}")
    
    # Now build the PDF in the correct order
    try:
        # Get the cover
        cover_pdf = PdfReader(cover_path)
        
        # Create PDF writer for output
        output = PdfWriter()
        
        # Add cover (no page number)
        output.add_page(cover_pdf.pages[0])
        
        # Add TOC pages
        toc_pdf = PdfReader(toc_path)
        for page in toc_pdf.pages:
            output.add_page(page)
        
        # Track the current page number
        current_page = 1
        
        # Add all PDFs in the order specified in TOC
        for i, (toc_path, pdf_path) in enumerate(chapter_files):
            try:
                # Check if the file exists and handle appendices files differently
                if 'appendices/' in toc_path:
                    # Get just the base name without the directory
                    base_name = os.path.basename(toc_path)
                    pdf_path = f"pdf_exports/{base_name}.pdf"
                
                if os.path.exists(pdf_path):
                    print(f"Adding {pdf_path} to the complete book")
                    pdf = PdfReader(pdf_path)
                    
                    for page in pdf.pages:
                        # Add page number
                        page_with_number = add_page_number(page, current_page)
                        output.add_page(page_with_number)
                        current_page += 1
                else:
                    print(f"Warning: PDF file not found: {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        
        # Write the final PDF
        with open(output_path, "wb") as output_file:
            output.write(output_file)
        
        print(f"\nComplete book created at {output_path}")
        print(f"Total pages: {current_page}")
        
        # Now add cover to the final PDF
        with open(output_path, "rb") as book_file, open(cover_path, "rb") as cover_file:
            book_pdf = PdfReader(book_file)
            cover_pdf = PdfReader(cover_file)
            
            final_output_writer = PdfWriter()
            
            # Add the cover page first
            final_output_writer.add_page(cover_pdf.pages[0])
            
            # Add all pages from the book
            for page in book_pdf.pages:
                final_output_writer.add_page(page)
            
            # Write the final PDF
            with open(final_output, "wb") as final_file:
                final_output_writer.write(final_file)
            
            print(f"Final complete book with cover created at {final_output}")
        
        return True
        
    except Exception as e:
        print(f"Error creating complete book: {e}")
        return False

if __name__ == "__main__":
    create_complete_book_in_toc_order()