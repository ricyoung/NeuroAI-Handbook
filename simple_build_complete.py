#!/usr/bin/env python3
"""
Simple script to build the complete NeuroAI Handbook by combining PDFs directly with PyPDF2.
"""

import os
import glob
import yaml
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Configuration
PDF_DIR = "pdf_exports"
OUTPUT_PDF = os.path.join(PDF_DIR, "neuroai_handbook_simple.pdf")
TOC_FILE = "book/_toc.yml"

def get_chapter_order():
    """Get the chapter order from the _toc.yml file."""
    with open(TOC_FILE, 'r') as f:
        toc_data = yaml.safe_load(f)
    
    chapter_order = []
    for part in toc_data.get('parts', []):
        for chapter in part.get('chapters', []):
            if isinstance(chapter, dict) and 'file' in chapter:
                chapter_file = os.path.basename(chapter['file'])
            else:
                chapter_file = os.path.basename(chapter)
            
            chapter_base = chapter_file.replace('.md', '')
            chapter_order.append(chapter_base)
    
    return chapter_order

def add_page_numbers(pdf_writer):
    """Add page numbers to all pages."""
    for i in range(len(pdf_writer.pages)):
        page = pdf_writer.pages[i]
        
        # Create a PDF with just the page number
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=letter)
        c.setFont("Helvetica", 10)
        c.drawCentredString(letter[0]/2, 30, str(i + 1))
        c.save()
        
        # Add the page number to the page
        packet.seek(0)
        number_pdf = PdfReader(packet)
        page.merge_page(number_pdf.pages[0])

def merge_pdfs():
    """Merge all chapter PDFs in the correct order."""
    ordered_chapters = get_chapter_order()
    print("Chapters in order:")
    for chapter in ordered_chapters:
        print(f"  - {chapter}")
    
    # Create output PDF writer
    output = PdfWriter()
    
    # Add cover page if it exists
    cover_path = os.path.join(PDF_DIR, "cover.pdf")
    if os.path.exists(cover_path):
        print(f"Adding cover from {cover_path}")
        with open(cover_path, 'rb') as f:
            cover_pdf = PdfReader(f)
            output.add_page(cover_pdf.pages[0])
    
    # Add each chapter in order
    for chapter in ordered_chapters:
        pdf_path = os.path.join(PDF_DIR, f"{chapter}.pdf")
        if os.path.exists(pdf_path):
            print(f"Adding chapter: {chapter}")
            with open(pdf_path, 'rb') as f:
                pdf = PdfReader(f)
                for page in pdf.pages:
                    output.add_page(page)
        else:
            print(f"Warning: {pdf_path} not found")
    
    # Add page numbers
    print("Adding page numbers...")
    add_page_numbers(output)
    
    # Write the final PDF
    print(f"Writing final PDF to {OUTPUT_PDF}")
    with open(OUTPUT_PDF, 'wb') as output_file:
        output.write(output_file)
    
    print(f"Created PDF with {len(output.pages)} pages")

if __name__ == "__main__":
    merge_pdfs()