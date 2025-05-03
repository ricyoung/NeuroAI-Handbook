#!/usr/bin/env python
import os
import yaml
import re
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io

def add_page_number(page, page_number):
    """Add page number to the PDF page."""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    width, height = letter
    
    # Add page number at the bottom
    can.setFont('Helvetica', 10)
    can.setFillColorRGB(0.5, 0.5, 0.5)
    can.drawCentredString(width/2, 0.5*inch, str(page_number))
    
    can.save()
    packet.seek(0)
    number_pdf = PdfReader(packet)
    
    # Merge the page number into the existing page
    page.merge_page(number_pdf.pages[0])
    return page

def get_chapter_title(filename):
    """Extract a formatted title from the chapter filename."""
    # Remove file extension
    base_name = os.path.basename(filename)
    base_name = os.path.splitext(base_name)[0]
    
    # Extract chapter number 
    chapter_num = ""
    if base_name.startswith("ch"):
        match = re.match(r'ch(\d+)_(.+)', base_name)
        if match:
            chapter_num = match.group(1)
            chapter_text = match.group(2)
            # Replace underscores with spaces and title case
            chapter_text = chapter_text.replace('_', ' ').title()
            return f"Chapter {chapter_num}", chapter_text
    
    # Handle appendices
    if base_name in ["glossary", "math_python_refresher", "dataset_catalogue", "colab_setup"]:
        appendix_titles = {
            "glossary": "Glossary",
            "math_python_refresher": "Math & Python Refresher",
            "dataset_catalogue": "Dataset Catalogue", 
            "colab_setup": "Colab Setup Guide"
        }
        return "Appendix", appendix_titles.get(base_name, base_name.replace('_', ' ').title())
    
    # Default formatting
    return "", base_name.replace('_', ' ').title()

def create_divider_page(chapter_num, chapter_title):
    """Create a PDF divider page for a chapter."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    width, height = letter
    
    # Light background color
    c.setFillColorRGB(0.97, 0.97, 0.97)
    c.rect(0, 0, width, height, fill=True)
    
    # Add decorative element
    c.setStrokeColorRGB(0.8, 0.8, 0.9)
    c.setLineWidth(2)
    c.line(width/4, height/2, 3*width/4, height/2)
    
    # Add chapter number
    c.setFillColorRGB(0.4, 0.4, 0.7)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height/2 + 60, chapter_num)
    
    # Add chapter title
    c.setFillColorRGB(0.2, 0.2, 0.4)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width/2, height/2, chapter_title)
    
    c.save()
    packet.seek(0)
    
    return PdfReader(packet)

def create_toc(chapter_info):
    """Create a simple table of contents page."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    width, height = letter
    
    # Background
    c.setFillColorRGB(0.97, 0.97, 0.97)
    c.rect(0, 0, width, height, fill=True)
    
    # Title
    c.setFillColorRGB(0.2, 0.2, 0.4)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 1.2*inch, "Table of Contents")
    
    # Decorative line
    c.setStrokeColorRGB(0.4, 0.4, 0.7)
    c.setLineWidth(1.5)
    c.line(width/4, height - 1.5*inch, 3*width/4, height - 1.5*inch)
    
    # Start position for entries
    y_pos = height - 2*inch
    
    # Group chapters by part
    parts = {}
    for chapter_prefix, chapter_title, page_num, part in chapter_info:
        if part not in parts:
            parts[part] = []
        parts[part].append((chapter_prefix, chapter_title, page_num))
    
    # Draw TOC entries
    for part_name, chapters in parts.items():
        # Skip empty parts
        if not part_name:
            continue
            
        # Part heading
        c.setFillColorRGB(0.3, 0.3, 0.6)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y_pos, part_name)
        y_pos -= 0.4*inch
        
        # Chapters in this part
        for chapter_prefix, chapter_title, page_num in chapters:
            c.setFillColorRGB(0.1, 0.1, 0.1)
            c.setFont("Helvetica", 12)
            
            # Draw chapter info
            title_text = f"{chapter_prefix} {chapter_title}"
            c.drawString(1.2*inch, y_pos, title_text)
            
            # Draw dots
            text_width = c.stringWidth(title_text, "Helvetica", 12)
            dot_spacing = 6
            start_x = 1.2*inch + text_width + 10
            end_x = width - inch - 20
            
            c.setFillColorRGB(0.5, 0.5, 0.5)
            for x in range(int(start_x), int(end_x), dot_spacing):
                c.circle(x, y_pos + 4, 0.5, fill=1)
            
            # Page number
            c.setFont("Helvetica", 12)
            c.drawRightString(width - inch, y_pos, str(page_num))
            
            y_pos -= 0.3*inch
            
            # Start a new page if needed
            if y_pos < inch:
                c.showPage()
                y_pos = height - inch
                # Reset background for new page
                c.setFillColorRGB(0.97, 0.97, 0.97)
                c.rect(0, 0, width, height, fill=True)
        
        # Add space between parts
        y_pos -= 0.2*inch
        
        # Start a new page if needed
        if y_pos < inch:
            c.showPage()
            y_pos = height - inch
            # Reset background for new page
            c.setFillColorRGB(0.97, 0.97, 0.97)
            c.rect(0, 0, width, height, fill=True)
    
    c.save()
    packet.seek(0)
    return PdfReader(packet)

def create_simple_final_book():
    """Create a final book with TOC, dividers, and page numbers."""
    # Paths
    toc_yml_path = "book/_toc.yml"
    cover_pdf = "pdf_exports/cover.pdf"
    output_pdf = "pdf_exports/with_covers/neuroai_handbook_final.pdf"
    
    # Load TOC structure
    with open(toc_yml_path, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Get parts
    parts = toc_data.get('parts', [])
    
    # Create ordered list of file paths
    ordered_files = []
    part_index = 0
    
    for part in parts:
        # Skip cover part
        if part.get('caption') == 'Cover':
            continue
        
        part_title = f"Part {part_index+1}: {part.get('caption', '').split(' · ')[-1]}"
        part_index += 1
        
        for chapter in part.get('chapters', []):
            # Get file base name
            file_path = chapter.get('file', '')
            if file_path:
                # Get title from TOC
                title = chapter.get('title', '')
                
                # Get base name from path
                base_name = os.path.basename(file_path)
                pdf_path = f"pdf_exports/{base_name}.pdf"
                
                # Only add if PDF exists
                if os.path.exists(pdf_path):
                    ordered_files.append((pdf_path, title, part_title))
                else:
                    print(f"Warning: PDF not found: {pdf_path}")
    
    # Print order
    print("PDFs in order:")
    for i, (path, title, part) in enumerate(ordered_files):
        print(f"  {i+1}. {path} - {title or '(No title)'} - Part: {part}")
    
    # Create output PDF
    output = PdfWriter()
    
    # Add cover
    if os.path.exists(cover_pdf):
        cover_reader = PdfReader(cover_pdf)
        for page in cover_reader.pages:
            output.add_page(page)
        print(f"Added cover: {cover_pdf}")
    
    # Calculate TOC info
    chapter_info = []
    page_number = 2  # Start after cover page
    
    # First pass to collect page numbers
    for pdf_path, title, part in ordered_files:
        # Get chapter info
        chapter_prefix, chapter_title = get_chapter_title(pdf_path)
        
        # Use title from TOC if available
        if title:
            chapter_title = title
        
        # Only add to TOC if it's a chapter
        if chapter_prefix:
            chapter_info.append((chapter_prefix, chapter_title, page_number, part))
        
        # Count pages
        pdf = PdfReader(pdf_path)
        # +1 for divider page if it's a chapter
        page_number += len(pdf.pages) + (1 if chapter_prefix else 0)
    
    # Create and add TOC
    toc_reader = create_toc(chapter_info)
    for page_idx in range(len(toc_reader.pages)):
        output.add_page(toc_reader.pages[page_idx])
    print(f"Added TOC with {len(toc_reader.pages)} pages")
    
    # Adjust page count for content
    current_page = 1 + len(toc_reader.pages)  # Start after cover and TOC
    
    # Second pass to add content with page numbers
    for pdf_path, title, part in ordered_files:
        # Get chapter info
        chapter_prefix, chapter_title = get_chapter_title(pdf_path)
        
        # Use title from TOC if available
        if title:
            chapter_title = title
        
        # Add divider for chapters
        if chapter_prefix:
            # Create divider page
            divider_reader = create_divider_page(chapter_prefix, chapter_title)
            divider_page = divider_reader.pages[0]
            
            # Add page number
            numbered_divider = add_page_number(divider_page, current_page)
            output.add_page(numbered_divider)
            
            print(f"Added divider for: {chapter_prefix} - {chapter_title} (page {current_page})")
            current_page += 1
        
        # Add content pages with numbers
        try:
            pdf = PdfReader(pdf_path)
            for page_idx in range(len(pdf.pages)):
                page = pdf.pages[page_idx]
                # Add page number
                numbered_page = add_page_number(page, current_page)
                output.add_page(numbered_page)
                current_page += 1
            
            print(f"Added: {pdf_path} (ending at page {current_page-1})")
        except Exception as e:
            print(f"Error adding {pdf_path}: {e}")
    
    # Write final PDF
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, "wb") as f:
        output.write(f)
    
    print(f"\nCreated final PDF at: {output_pdf}")
    print(f"Total pages: {current_page-1}")

if __name__ == "__main__":
    create_simple_final_book()