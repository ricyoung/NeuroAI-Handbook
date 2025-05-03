#!/usr/bin/env python3
import os
import yaml
import glob
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# Configure paths
PDF_DIR = "pdf_exports"
COVER_PDF = os.path.join(PDF_DIR, "cover.pdf")
OUTPUT_DIR = os.path.join(PDF_DIR, "with_covers")
FINAL_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_final.pdf")
TOC_CONFIG = "book/_toc.yml"

def read_toc_structure():
    """Read the table of contents from the _toc.yml file."""
    with open(TOC_CONFIG, 'r') as file:
        toc_data = yaml.safe_load(file)
        
    # Get the correct chapter order from the TOC
    chapters = []
    
    for part in toc_data.get('parts', []):
        for chapter in part.get('chapters', []):
            # Get just the filename without path or extension
            if isinstance(chapter, dict) and 'file' in chapter:
                filename = os.path.basename(chapter['file'])
                chapters.append(filename)
            elif isinstance(chapter, str):
                filename = os.path.basename(chapter)
                chapters.append(filename)
    
    return chapters

def create_divider_page(title, part_number):
    """Create a PDF divider page with the part title."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    
    # Set font
    c.setFont("Helvetica-Bold", 24)
    
    # Draw part number and title centered
    c.drawCentredString(letter[0]/2, letter[1]/2, f"Part {part_number}")
    c.drawCentredString(letter[0]/2, letter[1]/2 - 40, title)
    
    c.save()
    packet.seek(0)
    return PdfReader(packet)

def add_page_numbers(pdf_writer):
    """Add page numbers to all pages except the cover."""
    for i in range(1, len(pdf_writer.pages)):  # Skip the cover page (index 0)
        page = pdf_writer.pages[i]
        
        # Create a canvas with a page number
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=letter)
        c.setFont("Helvetica", 10)
        c.drawCentredString(letter[0]/2, 30, str(i))
        c.save()
        
        # Add the page number to the page
        packet.seek(0)
        number_pdf = PdfReader(packet)
        page.merge_page(number_pdf.pages[0])

def create_toc_pages():
    """Create a table of contents PDF."""
    packet = io.BytesIO()
    doc = SimpleDocTemplate(packet, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Get the chapter structure from _toc.yml
    with open(TOC_CONFIG, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Create the document structure
    elements = []
    
    # Add title
    title_style = styles['Title']
    elements.append(Paragraph("Table of Contents", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Add each part and its chapters
    part_num = 1
    page_tracker = 2  # Start at page 2 (after cover)
    
    for part in toc_data.get('parts', []):
        # Add part title
        part_title = part.get('caption', f"Part {part_num}")
        elements.append(Paragraph(f"Part {part_num}: {part_title}", styles['Heading1']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Find all PDFs for this part
        for chapter in part.get('chapters', []):
            # Get the chapter file
            if isinstance(chapter, dict) and 'file' in chapter:
                chapter_file = os.path.basename(chapter['file'])
                chapter_title = chapter.get('title', chapter_file)
            elif isinstance(chapter, str):
                chapter_file = os.path.basename(chapter)
                chapter_title = chapter_file.replace("_", " ").replace(".md", "")
            
            # Get corresponding PDF
            pdf_path = os.path.join(PDF_DIR, f"{chapter_file.replace('.md', '')}.pdf")
            
            if os.path.exists(pdf_path):
                # Get page count from this PDF to track TOC numbering
                with open(pdf_path, 'rb') as f:
                    pdf = PdfReader(f)
                    page_count = len(pdf.pages)
                
                # Add chapter entry with page number
                elements.append(Paragraph(
                    f"{chapter_title} ..... {page_tracker}",
                    styles['Normal']
                ))
                elements.append(Spacer(1, 0.1*inch))
                
                # Update page tracker for next chapter
                page_tracker += page_count
            
        part_num += 1
        elements.append(Spacer(1, 0.2*inch))
    
    # Build the PDF
    doc.build(elements)
    packet.seek(0)
    
    toc_pdf_path = os.path.join(PDF_DIR, "toc.pdf")
    with open(toc_pdf_path, 'wb') as f:
        f.write(packet.getvalue())
    
    return toc_pdf_path

def create_final_book():
    """Create the final book with cover, TOC, and all chapters."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a PDF writer
    output = PdfWriter()
    
    # Add cover page
    print("Adding cover page...")
    cover_pdf = PdfReader(COVER_PDF)
    output.add_page(cover_pdf.pages[0])
    
    # Create and add TOC
    print("Creating table of contents...")
    toc_path = create_toc_pages()
    toc_pdf = PdfReader(toc_path)
    for page in toc_pdf.pages:
        output.add_page(page)
    
    # Get chapter order from TOC
    chapter_files = read_toc_structure()
    print(f"Reading chapters in order from TOC structure...")
    
    # Add chapters in correct order
    part_num = 1
    with open(TOC_CONFIG, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    for part in toc_data.get('parts', []):
        # Add part divider
        part_title = part.get('caption', f"Part {part_num}")
        print(f"Adding divider for Part {part_num}: {part_title}")
        divider_pdf = create_divider_page(part_title, part_num)
        output.add_page(divider_pdf.pages[0])
        
        # Add chapters in this part
        for chapter in part.get('chapters', []):
            # Get chapter file
            if isinstance(chapter, dict) and 'file' in chapter:
                chapter_file = os.path.basename(chapter['file'])
            else:
                chapter_file = os.path.basename(chapter)
            
            # Strip .md extension if present
            chapter_base = chapter_file.replace(".md", "")
            pdf_path = os.path.join(PDF_DIR, f"{chapter_base}.pdf")
            
            print(f"  Adding chapter: {chapter_base}")
            try:
                chapter_pdf = PdfReader(pdf_path)
                for page in chapter_pdf.pages:
                    output.add_page(page)
            except Exception as e:
                print(f"    Error adding {pdf_path}: {e}")
        
        part_num += 1
    
    # Add page numbers
    print("Adding page numbers...")
    add_page_numbers(output)
    
    # Write the final PDF
    print(f"Writing final book to {FINAL_PDF}...")
    with open(FINAL_PDF, "wb") as output_file:
        output.write(output_file)
    
    print(f"Final book created with {len(output.pages)} pages.")

if __name__ == "__main__":
    create_final_book()