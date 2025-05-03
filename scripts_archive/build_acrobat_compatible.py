#!/usr/bin/env python3
"""
Build an Acrobat-compatible version of the NeuroAI Handbook PDF.
Uses a simpler approach to avoid PDF compatibility issues.
"""

import os
import sys
import subprocess
import tempfile
from PyPDF2 import PdfWriter, PdfReader
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

# Configuration
PDF_DIR = "pdf_exports"
COVER_PDF = os.path.join(PDF_DIR, "cover.pdf")
OUTPUT_DIR = os.path.join(PDF_DIR, "with_covers")
FINAL_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_acrobat.pdf")

# Order of chapters with part information
PARTS = [
    {
        "title": "Part I · Brains & Inspiration",
        "chapters": ["ch01_intro", "ch02_neuro_foundations", "ch03_spatial_navigation", "ch04_perception_pipeline"]
    },
    {
        "title": "Part II · Brains Meet Math & Data",
        "chapters": ["ch05_brain_networks", "ch06_neurostimulation", "ch07_information_theory", "ch08_data_science_pipeline"]
    },
    {
        "title": "Part III · Learning Machines",
        "chapters": ["ch09_ml_foundations", "ch10_deep_learning", "ch11_sequence_models"]
    },
    {
        "title": "Part IV · Frontier Models",
        "chapters": ["ch12_large_language_models", "ch13_multimodal_models"]
    },
    {
        "title": "Part V · Ethics & Futures",
        "chapters": ["ch15_ethical_ai", "ch16_future_directions"]
    },
    {
        "title": "Part VI · Advanced Applications",
        "chapters": ["ch17_bci_human_ai_interfaces", "ch18_neuromorphic_computing", 
                     "ch19_cognitive_neuro_dl", "ch20_case_studies", "ch21_ai_for_neuro_discovery", 
                     "ch22_embodied_ai_robotics", "ch24_quantum_computing_neuroai", "ch23_lifelong_learning"]
    },
    {
        "title": "Appendices",
        "chapters": ["math_python_refresher", "dataset_catalogue", "colab_setup"]
    }
]

def print_status(message):
    """Print a status message."""
    print(f"[INFO] {message}")

def create_toc_pdf(output_path):
    """Create a table of contents PDF."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create a centered title style
    title_style = ParagraphStyle(
        name='CenteredTitle',
        parent=styles['Title'],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=30
    )
    
    # Create heading styles for parts and chapters
    part_style = ParagraphStyle(
        name='PartHeading',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        spaceBefore=24
    )
    
    chapter_style = ParagraphStyle(
        name='ChapterItem',
        parent=styles['Normal'],
        fontSize=12,
        leftIndent=20,
        spaceAfter=6
    )
    
    # Create the TOC elements
    elements = []
    
    # Add TOC title
    elements.append(Paragraph("Table of Contents", title_style))
    elements.append(Spacer(1, 0.5*72))  # 0.5 inch in points
    
    # Add each part and its chapters (without page numbers for now)
    for part_num, part in enumerate(PARTS, 1):
        # Add part title
        part_title = part.get('title', f"Part {part_num}")
        elements.append(Paragraph(f"{part_title}", part_style))
        
        # Add chapters in this part
        for chapter in part.get('chapters', []):
            # Get friendly chapter name
            chapter_name = chapter.replace("_", " ")
            if chapter_name.startswith("ch"):
                # Extract chapter number if available
                chapter_number = chapter_name[2:4]
                if chapter_number.isdigit():
                    chapter_name = f"Chapter {int(chapter_number)}: {chapter_name[5:]}"
                else:
                    chapter_name = chapter_name[3:]  # Remove "ch" prefix
            
            # Capitalize each word
            chapter_name = ' '.join(word.capitalize() for word in chapter_name.split())
            
            # Add chapter to TOC (without page number)
            elements.append(Paragraph(
                f"{chapter_name}",
                chapter_style
            ))
    
    # Build the TOC PDF
    doc.build(elements)
    print_status(f"Created TOC PDF at {output_path}")

def create_divider_pdf(title, part_number, output_path):
    """Create a PDF divider page with the part title."""
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Set font
    c.setFont("Helvetica-Bold", 24)
    
    # Draw part number and title centered
    part_title = f"Part {part_number}"
    title_y = letter[1]/2 + 40
    
    c.drawCentredString(letter[0]/2, title_y, part_title)
    
    # Draw a decorative line
    c.setStrokeColor(colors.grey)
    c.setLineWidth(2)
    c.line(letter[0]/4, title_y - 20, 3*letter[0]/4, title_y - 20)
    
    # Draw the part title below the line
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(letter[0]/2, title_y - 60, title)
    
    c.save()
    print_status(f"Created divider PDF for {title} at {output_path}")

def combine_pdfs():
    """Combine all PDFs into the final book."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        print_status(f"Using temporary directory: {temp_dir}")
        
        # List of PDFs to combine
        pdfs_to_combine = []
        
        # Add cover
        if os.path.exists(COVER_PDF):
            pdfs_to_combine.append(COVER_PDF)
            print_status(f"Added cover: {COVER_PDF}")
        else:
            print_status(f"Warning: Cover not found at {COVER_PDF}")
        
        # Create TOC
        toc_path = os.path.join(temp_dir, "toc.pdf")
        create_toc_pdf(toc_path)
        pdfs_to_combine.append(toc_path)
        
        # Add parts and chapters
        for part_num, part in enumerate(PARTS, 1):
            # Create divider
            divider_path = os.path.join(temp_dir, f"divider_{part_num}.pdf")
            create_divider_pdf(part.get('title', f"Part {part_num}"), part_num, divider_path)
            pdfs_to_combine.append(divider_path)
            
            # Add chapters
            for chapter in part.get('chapters', []):
                chapter_path = os.path.join(PDF_DIR, f"{chapter}.pdf")
                if os.path.exists(chapter_path):
                    pdfs_to_combine.append(chapter_path)
                    print_status(f"Added chapter: {chapter}")
                else:
                    print_status(f"Warning: Chapter not found: {chapter_path}")
        
        # Use PyPDF2 to combine all PDFs
        print_status("Combining all PDFs...")
        output = PdfWriter()
        
        for pdf_path in pdfs_to_combine:
            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        output.add_page(page)
        
        print_status(f"Writing final PDF to {FINAL_PDF} ({len(output.pages)} pages)")
        with open(FINAL_PDF, 'wb') as f:
            output.write(f)
        
        print_status(f"Successfully created {FINAL_PDF}")

if __name__ == "__main__":
    print_status("Building Acrobat-compatible NeuroAI Handbook PDF")
    combine_pdfs()
    print_status("Build complete!")