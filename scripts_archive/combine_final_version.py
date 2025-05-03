#!/usr/bin/env python
import os
import yaml
import re
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io

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

def create_toc_pages(chapter_info):
    """Create table of contents pages."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    width, height = letter
    
    # Variables for positioning
    y_pos = height - 2*inch
    items_per_page = 25
    item_count = 0
    current_part = None
    
    # Add title to first page
    c.setFillColorRGB(0.2, 0.2, 0.4)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 1.2*inch, "Table of Contents")
    
    # Draw a decorative line under the title
    c.setStrokeColorRGB(0.4, 0.4, 0.7)
    c.setLineWidth(1.5)
    c.line(width/4, height - 1.5*inch, 3*width/4, height - 1.5*inch)
    
    # Process each chapter entry
    for i, (chapter_prefix, chapter_title, page_num, part) in enumerate(chapter_info):
        # Check if we need to start a new page
        if item_count >= items_per_page:
            c.showPage()
            y_pos = height - inch
            item_count = 0
            
            # Reset background color for new page
            c.setFillColorRGB(0.97, 0.97, 0.97)
            c.rect(0, 0, width, height, fill=True)
        
        # If this is a new part, add the part title
        if part and part != current_part:
            # Add some space before new part
            if current_part is not None:
                y_pos -= 0.3*inch
                item_count += 1
                
                # Check if we need to start a new page
                if item_count >= items_per_page:
                    c.showPage()
                    y_pos = height - inch
                    item_count = 0
                    
                    # Reset background color for new page
                    c.setFillColorRGB(0.97, 0.97, 0.97)
                    c.rect(0, 0, width, height, fill=True)
            
            # Add part heading
            c.setFillColorRGB(0.3, 0.3, 0.6)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(inch, y_pos, part)
            y_pos -= 0.4*inch
            item_count += 1
            current_part = part
        
        # Draw chapter entry
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica", 12)
        
        # Chapter prefix and title
        title_text = f"{chapter_prefix} {chapter_title}"
        c.drawString(1.2*inch, y_pos, title_text)
        
        # Add dotted line - using dots for TOC leader instead of dashed line
        c.setStrokeColorRGB(0.5, 0.5, 0.5)
        text_width = c.stringWidth(title_text, "Helvetica", 12)
        
        # Draw dots manually instead of using setLineDash
        dot_spacing = 6
        start_x = 1.2*inch + text_width + 10
        end_x = width - inch - 20
        for x in range(int(start_x), int(end_x), dot_spacing):
            c.circle(x, y_pos + 6, 0.5, fill=1)
        
        #c.line(1.2*inch + text_width + 10, y_pos + 6, width - inch - 20, y_pos + 6)
        
        # Page number
        c.setFont("Helvetica", 12)
        c.drawRightString(width - inch, y_pos, str(page_num))
        
        # Move to next line
        y_pos -= 0.3*inch
        item_count += 1
    
    # Save the final page
    c.save()
    packet.seek(0)
    
    return PdfReader(packet)

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

def get_part_title(i, part_list):
    """Get the part title based on index."""
    if i < 0 or i >= len(part_list):
        return "Unknown Part"
    
    part_caption = part_list[i].get('caption', '')
    # Remove the "Part X · " prefix if present
    if " · " in part_caption:
        part_caption = part_caption.split(" · ", 1)[1]
    
    return f"Part {i+1}: {part_caption}"

def create_final_book():
    """Create final book with TOC, dividers, and page numbers."""
    # Paths
    toc_yml_path = "book/_toc.yml"
    cover_pdf = "pdf_exports/cover.pdf"
    output_pdf = "pdf_exports/with_covers/neuroai_handbook_final.pdf"
    
    # Load TOC structure to get chapter order and titles
    with open(toc_yml_path, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Get list of parts for organizing TOC
    parts = toc_data.get('parts', [])
    
    # Create ordered list of file paths with titles
    ordered_files = []
    part_index = -1
    
    for part in parts:
        part_index += 1
        # Skip the cover part
        if part.get('caption') == 'Cover':
            continue
            
        part_title = get_part_title(part_index, parts)
        
        for chapter in part.get('chapters', []):
            # Get file base name
            file_path = chapter.get('file', '')
            if file_path:
                # Extract title from TOC if available, otherwise from filename
                title = chapter.get('title', '')
                
                # Get base name from path
                base_name = os.path.basename(file_path)
                pdf_path = f"pdf_exports/{base_name}.pdf"
                
                # Only add if the PDF exists
                if os.path.exists(pdf_path):
                    ordered_files.append((pdf_path, title, part_title))
                else:
                    print(f"Warning: PDF not found: {pdf_path}")
            
            # Add sections if any (these don't get dividers)
            if 'sections' in chapter:
                for section in chapter.get('sections', []):
                    section_path = section.get('file', '')
                    if section_path:
                        # Get base name
                        base_name = os.path.basename(section_path)
                        pdf_path = f"pdf_exports/{base_name}.pdf"
                        
                        # Only add if the PDF exists 
                        if os.path.exists(pdf_path):
                            # Sections don't get divider pages, mark with empty string
                            ordered_files.append((pdf_path, section.get('title', ''), ""))
                        else:
                            print(f"Warning: PDF not found: {pdf_path}")
    
    # Print order
    print("PDFs in order:")
    for i, (path, title, part) in enumerate(ordered_files):
        print(f"  {i+1}. {path} - {title or '(No title)'} - Part: {part}")
    
    # Create merged PDF
    merger = PdfWriter()
    
    # First add the cover
    cover_reader = None
    if os.path.exists(cover_pdf):
        cover_reader = PdfReader(cover_pdf)
        merger.append(cover_reader)
        print(f"Added cover: {cover_pdf}")
    else:
        print(f"Warning: Cover PDF not found: {cover_pdf}")
    
    # Tracking info for TOC
    chapter_info = []
    current_page = 1  # Start page numbering after cover and TOC
    
    # First pass to collect all chapter info for TOC
    for pdf_path, title, part in ordered_files:
        try:
            # Get chapter number and title
            chapter_prefix, chapter_title = get_chapter_title(pdf_path)
            
            # Use title from TOC if available
            if title:
                chapter_title = title
            
            # Only add TOC entry for chapter files (not sections)
            if chapter_prefix:
                # Chapter info: (prefix, title, start page, part title)
                chapter_info.append((chapter_prefix, chapter_title, current_page + 2, part))  # +2 for cover and TOC page
                
            # Count pages in the PDF
            pdf = PdfReader(pdf_path)
            current_page += len(pdf.pages) + (1 if chapter_prefix else 0)  # +1 for divider if it's a chapter
            
        except Exception as e:
            print(f"Error processing chapter info for {pdf_path}: {e}")
    
    # Create and add TOC
    toc_reader = create_toc_pages(chapter_info)
    for page in toc_reader.pages:
        merger.append(page)
    print("Added table of contents")
    
    # Calculate TOC pages
    toc_page_count = len(toc_reader.pages)
    
    # Reset page counter for actual content
    current_page = 1 + toc_page_count  # Start after cover + TOC
    
    # Second pass to add all content with page numbers
    for pdf_path, title, part in ordered_files:
        try:
            # Get chapter number and title
            chapter_prefix, chapter_title = get_chapter_title(pdf_path)
            
            # Use title from TOC if available
            if title:
                chapter_title = title
            
            # Only add divider for chapter files (not sections)
            if chapter_prefix:
                # Create and add divider page with page number
                divider_reader = create_divider_page(chapter_prefix, chapter_title)
                divider_page = divider_reader.pages[0]
                divider_with_number = add_page_number(divider_page, current_page)
                merger.add_page(divider_with_number)
                print(f"Added divider for: {chapter_prefix} - {chapter_title} (page {current_page})")
                current_page += 1
            
            # Add the actual chapter with page numbers
            pdf = PdfReader(pdf_path)
            for page in pdf.pages:
                # Add page number
                page_with_number = add_page_number(page, current_page)
                merger.add_page(page_with_number)
                current_page += 1
            
            print(f"Added: {pdf_path} (ending at page {current_page-1})")
        except Exception as e:
            print(f"Error adding {pdf_path}: {e}")
    
    # Write to output
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, "wb") as f:
        merger.write(f)
    
    print(f"\nCreated final PDF at: {output_pdf}")
    print(f"Total pages: {current_page-1}")

if __name__ == "__main__":
    create_final_book()