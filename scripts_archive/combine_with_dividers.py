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

def combine_chapters_with_dividers():
    """Combine PDFs with divider pages between chapters."""
    # Paths
    toc_yml_path = "book/_toc.yml"
    cover_pdf = "pdf_exports/cover.pdf"
    output_pdf = "pdf_exports/with_covers/neuroai_handbook_with_dividers.pdf"
    
    # Load TOC structure to get chapter order and titles
    with open(toc_yml_path, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Create ordered list of file paths with titles
    ordered_files = []
    for part in toc_data.get('parts', []):
        for chapter in part.get('chapters', []):
            # Skip the cover
            if chapter.get('file') == 'cover':
                continue
                
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
                    ordered_files.append((pdf_path, title))
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
                            ordered_files.append((pdf_path, ""))
                        else:
                            print(f"Warning: PDF not found: {pdf_path}")
    
    # Print order
    print("PDFs in order:")
    for i, (path, title) in enumerate(ordered_files):
        print(f"  {i+1}. {path} - {title or '(No title)'}")
    
    # Create merged PDF
    merger = PdfWriter()
    
    # First add the cover
    if os.path.exists(cover_pdf):
        merger.append(cover_pdf)
        print(f"Added cover: {cover_pdf}")
    else:
        print(f"Warning: Cover PDF not found: {cover_pdf}")
    
    # Add all chapter PDFs in order, with dividers
    for pdf_path, title in ordered_files:
        try:
            # Get chapter number and title
            chapter_num, chapter_title = get_chapter_title(pdf_path)
            
            # Use title from TOC if available
            if title:
                chapter_title = title
            
            # Only add divider for chapter files (not sections)
            if chapter_num:
                # Create and add divider page
                divider = create_divider_page(chapter_num, chapter_title)
                merger.append(divider)
                print(f"Added divider for: {chapter_num} - {chapter_title}")
            
            # Add the actual chapter
            merger.append(pdf_path)
            print(f"Added: {pdf_path}")
        except Exception as e:
            print(f"Error adding {pdf_path}: {e}")
    
    # Write to output
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, "wb") as f:
        merger.write(f)
    
    print(f"\nCreated combined PDF at: {output_pdf}")
    print(f"Total PDFs combined: {len(ordered_files) + 1} with dividers")

if __name__ == "__main__":
    combine_chapters_with_dividers()