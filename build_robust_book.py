#!/usr/bin/env python3
"""
Robust script to build the NeuroAI Handbook PDF with validation at every step.
Includes:
- Cover page
- Table of contents with proper numbering
- Chapter dividers for each part
- All chapters in correct order
- Page numbers on all content pages
- Error handling and validation throughout
"""

import os
import sys
import yaml
import glob
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
import io

# Configuration
PDF_DIR = "pdf_exports"
COVER_PDF = os.path.join(PDF_DIR, "cover.pdf")
OUTPUT_DIR = os.path.join(PDF_DIR, "with_covers")
FINAL_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_complete.pdf")
TOC_CONFIG = "book/_toc.yml"
VALIDATION_THRESHOLD = 5  # Minimum number of pages for a valid chapter PDF

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

def print_header(title):
    """Print a header with emphasis."""
    width = 80
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def validate_cover():
    """Validate the cover page exists and is a valid PDF."""
    print_header("VALIDATING COVER")
    
    if not os.path.exists(COVER_PDF):
        print_error(f"Cover PDF not found at: {COVER_PDF}")
        return False
    
    try:
        with open(COVER_PDF, 'rb') as f:
            cover_pdf = PdfReader(f)
            num_pages = len(cover_pdf.pages)
            if num_pages < 1:
                print_error(f"Cover PDF has {num_pages} pages (expected at least 1)")
                return False
        print_success(f"Cover validated: {COVER_PDF} ({num_pages} pages)")
        return True
    except Exception as e:
        print_error(f"Error validating cover PDF: {e}")
        return False

def validate_chapters():
    """Validate all chapter PDFs exist and are valid."""
    print_header("VALIDATING CHAPTERS")
    
    all_valid = True
    chapter_stats = {}
    
    for part in PARTS:
        for chapter in part["chapters"]:
            chapter_pdf = os.path.join(PDF_DIR, f"{chapter}.pdf")
            
            if not os.path.exists(chapter_pdf):
                print_error(f"Chapter PDF not found: {chapter_pdf}")
                all_valid = False
                continue
            
            try:
                with open(chapter_pdf, 'rb') as f:
                    pdf = PdfReader(f)
                    num_pages = len(pdf.pages)
                    
                    if num_pages < VALIDATION_THRESHOLD:
                        print_warning(f"Chapter {chapter} has only {num_pages} pages (might be incorrect)")
                    else:
                        print_success(f"Chapter {chapter}: {num_pages} pages")
                    
                    chapter_stats[chapter] = num_pages
            except Exception as e:
                print_error(f"Error validating chapter {chapter}: {e}")
                all_valid = False
    
    if all_valid:
        print_success(f"All chapters validated - Total chapters: {len(chapter_stats)}")
    else:
        print_error("Some chapters failed validation")
    
    return all_valid, chapter_stats

def create_toc_pdf(chapter_stats):
    """Create a table of contents PDF with accurate page numbers."""
    print_header("GENERATING TABLE OF CONTENTS")
    
    packet = io.BytesIO()
    doc = SimpleDocTemplate(packet, pagesize=letter)
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
    elements.append(Spacer(1, 0.5*inch))
    
    # Start page counter after cover and TOC (estimated 2 pages)
    page_counter = 3
    
    # Add each part and its chapters
    for part_num, part in enumerate(PARTS, 1):
        # Add part title (with page number where it starts)
        part_title = part.get('title', f"Part {part_num}")
        elements.append(Paragraph(f"{part_title}", part_style))
        
        # Add one page for part divider
        page_counter += 1
        
        # Add chapters in this part
        for chapter in part.get('chapters', []):
            # Get friendly chapter name (replace underscores, remove prefix)
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
                
            # Get page count from stats
            num_pages = chapter_stats.get(chapter, 0)
            if num_pages > 0:
                # Add chapter to TOC with page number
                elements.append(Paragraph(
                    f"{chapter_name} ................... {page_counter}",
                    chapter_style
                ))
                
                # Update page counter for next chapter
                page_counter += num_pages
            else:
                print_warning(f"Skipping {chapter} in TOC (no page count available)")
    
    # Build the TOC PDF
    doc.build(elements)
    packet.seek(0)
    
    print_success(f"Table of contents generated with {page_counter} total pages")
    return PdfReader(packet)

def create_divider_page(title, part_number):
    """Create a PDF divider page with the part title."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    
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
    packet.seek(0)
    return PdfReader(packet)

def add_page_numbers(pdf_writer):
    """Add page numbers to all pages except the cover."""
    print_header("ADDING PAGE NUMBERS")
    
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
        
        # Progress indicator
        if i % 50 == 0:
            print(f"  - Added page numbers up to page {i}...")
    
    print_success(f"Page numbers added to {len(pdf_writer.pages)-1} pages")

def build_complete_book():
    """Build the complete book with all required elements."""
    print_header("BUILDING COMPLETE BOOK")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Validate components
    if not validate_cover():
        print_error("Cover validation failed, cannot continue")
        return False
        
    valid_chapters, chapter_stats = validate_chapters()
    if not valid_chapters:
        print_error("Chapter validation failed, cannot continue")
        return False
    
    # Create PDF writer
    output = PdfWriter()
    
    # Add cover page
    print("Adding cover page...")
    with open(COVER_PDF, 'rb') as f:
        cover_pdf = PdfReader(f)
        output.add_page(cover_pdf.pages[0])
    
    # Create and add TOC
    print("Creating table of contents...")
    toc_pdf = create_toc_pdf(chapter_stats)
    for page in toc_pdf.pages:
        output.add_page(page)
    
    # Add parts and chapters
    for part_num, part in enumerate(PARTS, 1):
        part_title = part.get('title', f"Part {part_num}")
        print(f"Adding divider for {part_title}")
        
        # Add part divider
        divider_pdf = create_divider_page(part_title, part_num)
        output.add_page(divider_pdf.pages[0])
        
        # Add chapters in this part
        for chapter in part.get('chapters', []):
            chapter_pdf = os.path.join(PDF_DIR, f"{chapter}.pdf")
            if os.path.exists(chapter_pdf):
                print(f"  Adding chapter: {chapter}")
                with open(chapter_pdf, 'rb') as f:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        output.add_page(page)
            else:
                print_error(f"  {chapter_pdf} not found - THIS SHOULD NOT HAPPEN!")
                return False
    
    # Add page numbers
    add_page_numbers(output)
    
    # Write the final PDF
    print(f"Writing final PDF to {FINAL_PDF}")
    with open(FINAL_PDF, "wb") as output_file:
        output.write(output_file)
    
    print_success(f"Final book created with {len(output.pages)} pages")
    print(f"PDF saved to: {FINAL_PDF}")
    return True

if __name__ == "__main__":
    print_header("NEUROAI HANDBOOK PDF BUILDER")
    print("This script builds a complete PDF with TOC, dividers, and page numbers")
    print(f"Output will be saved to: {FINAL_PDF}")
    print("\nStarting build process...")
    
    success = build_complete_book()
    
    if success:
        print_header("BUILD SUCCESSFUL")
        print(f"The complete handbook has been built and saved to:")
        print(f"  {FINAL_PDF}")
        print("\nEnjoy your handbook!")
        sys.exit(0)
    else:
        print_header("BUILD FAILED")
        print("The handbook build failed due to validation errors.")
        print("Please fix the reported issues and try again.")
        sys.exit(1)