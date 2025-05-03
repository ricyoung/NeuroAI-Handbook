#!/usr/bin/env python3
"""
02_build_supporting.py - Create all supporting pages for the handbook.

This script creates:
1. A professional cover page
2. Copyright/legal page
3. Table of contents
4. Part divider pages
5. Acknowledgments page
6. Exports frontmatter from JupyterBook (integrating export_frontmatter.sh)
7. Other front/back matter
"""

import os
import sys
import subprocess
import glob
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "pdf_components")
SUPPORTING_DIR = os.path.join(OUTPUT_DIR, "supporting")
PDF_EXPORTS_DIR = os.path.join(BASE_DIR, "pdf_exports")
PDF_SUPPORTING_DIR = os.path.join(PDF_EXPORTS_DIR, "supporting")
PDF_FRONTMATTER_DIR = os.path.join(PDF_EXPORTS_DIR, "frontmatter")
ASSETS_DIR = os.path.join(BASE_DIR, "_assets")
BOOK_DIR = os.path.join(BASE_DIR, "book")

# Book metadata
BOOK_TITLE = "NeuroAI Handbook"
BOOK_SUBTITLE = "A Comprehensive Guide to the Intersection of Neuroscience and Artificial Intelligence"
AUTHORS = ["Editors and Contributors"]
YEAR = "2023"
VERSION = "1.0"

# Part titles for dividers
PARTS = [
    {"number": "I", "title": "Brains & Inspiration"},
    {"number": "II", "title": "Brains Meet Math & Data"},
    {"number": "III", "title": "Learning Machines"},
    {"number": "IV", "title": "Frontier Models"},
    {"number": "V", "title": "Ethics & Futures"},
    {"number": "VI", "title": "Advanced Applications"},
    {"number": "VII", "title": "Appendices"}
]

def print_status(message, is_header=False):
    """Print a formatted status message."""
    if is_header:
        sep = "=" * 80
        print(f"\n{sep}")
        print(f" {message} ".center(78))
        print(f"{sep}\n")
    else:
        print(f"[INFO] {message}")

def create_cover_page():
    """Create a professional cover page."""
    output_path = os.path.join(SUPPORTING_DIR, "00_cover.pdf")
    print_status(f"Creating cover page: {output_path}")
    
    # Check if we have a custom cover in the assets directory
    custom_cover = os.path.join(ASSETS_DIR, "cover.pdf")
    if os.path.exists(custom_cover):
        print_status(f"Using custom cover from {custom_cover}")
        cmd = f"cp {custom_cover} {output_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print_status("Custom cover page copied successfully")
            
            # Also copy to pdf_exports for reference
            exports_cover = os.path.join(PDF_EXPORTS_DIR, "cover.pdf")
            subprocess.run(f"cp {custom_cover} {exports_cover}", shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            print_status("WARNING: Could not copy custom cover, falling back to generated cover")
    else:
        print_status("No custom cover found in _assets directory, generating default cover")
    
    # Create a canvas for the default cover
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Set background color (light blue gradient)
    c.setFillColorRGB(0.9, 0.95, 1.0)
    c.rect(0, 0, width, height, fill=True)
    
    # Add a decorative header bar
    c.setFillColorRGB(0.2, 0.4, 0.8)  # Dark blue
    c.rect(0, height-2*inch, width, 2*inch, fill=True)
    
    # Add a decorative footer bar
    c.setFillColorRGB(0.2, 0.4, 0.8)  # Dark blue
    c.rect(0, 0, width, 1.5*inch, fill=True)
    
    # Add title
    c.setFont("Helvetica-Bold", 32)
    c.setFillColorRGB(0.1, 0.1, 0.3)  # Dark blue/black
    c.drawCentredString(width/2, height-4*inch, BOOK_TITLE)
    
    # Add subtitle
    c.setFont("Helvetica-Oblique", 16)
    c.setFillColorRGB(0.3, 0.3, 0.5)  # Slightly lighter
    text = c.beginText(2*inch, height-4.5*inch)
    text.setFont("Helvetica-Oblique", 16)
    text.setFillColorRGB(0.3, 0.3, 0.5)
    
    # Split subtitle into lines if needed
    words = BOOK_SUBTITLE.split()
    line1 = " ".join(words[:len(words)//2])
    line2 = " ".join(words[len(words)//2:])
    
    c.drawCentredString(width/2, height-4.7*inch, line1)
    c.drawCentredString(width/2, height-5.1*inch, line2)
    
    # Add tagline
    tagline = "Where Neurons Meet Algorithms: Bridging the Gap Between Brain Science and AI"
    c.setFont("Helvetica-Italic", 14)
    c.setFillColorRGB(0.3, 0.3, 0.5)
    
    # Split tagline into two lines if needed
    tagline_words = tagline.split()
    tagline_mid = len(tagline_words) // 2
    tagline_line1 = " ".join(tagline_words[:tagline_mid])
    tagline_line2 = " ".join(tagline_words[tagline_mid:])
    
    c.drawCentredString(width/2, height-6*inch, tagline_line1)
    c.drawCentredString(width/2, height-6.4*inch, tagline_line2)
    
    # Add authors
    c.setFont("Helvetica", 14)
    c.setFillColorRGB(0.2, 0.2, 0.4)
    y_pos = height-7.5*inch
    for author in AUTHORS:
        c.drawCentredString(width/2, y_pos, author)
        y_pos -= 0.5*inch
    
    # Add year and version
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, 2*inch, f"{YEAR} · Version {VERSION}")
    
    c.save()
    
    # Also copy to pdf_exports for reference
    exports_cover = os.path.join(PDF_EXPORTS_DIR, "cover.pdf")
    cmd = f"cp {output_path} {exports_cover}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print_status("WARNING: Could not copy cover to exports directory")
    
    print_status("Cover page created successfully")
    return True

def create_copyright_page():
    """Create a copyright/legal page."""
    output_path = os.path.join(SUPPORTING_DIR, "01_copyright.pdf")
    print_status(f"Creating copyright page: {output_path}")
    
    # Create document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='BookTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=20
    )
    
    normal_style = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        name='Heading',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold',
        spaceAfter=6,
        spaceBefore=12
    )
    
    # Content elements
    elements = []
    
    # Add some vertical space at the top
    elements.append(Spacer(1, 2*inch))
    
    # Title
    elements.append(Paragraph(BOOK_TITLE, title_style))
    elements.append(Paragraph(BOOK_SUBTITLE, styles['Italic']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Version and date
    elements.append(Paragraph(f"Version {VERSION}, {YEAR}", normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Copyright notice
    elements.append(Paragraph("© Copyright", heading_style))
    elements.append(Paragraph(
        f"Copyright © {YEAR} by the authors and contributors of the NeuroAI Handbook. "
        "All rights reserved.", normal_style))
    
    # License information
    elements.append(Paragraph("License", heading_style))
    elements.append(Paragraph(
        "This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 "
        "International License. Code examples are licensed under the MIT License.", 
        normal_style))
    
    # Disclaimer
    elements.append(Paragraph("Disclaimer", heading_style))
    elements.append(Paragraph(
        "The information provided in this handbook is for educational and informational "
        "purposes only. The authors and contributors make no representations as to accuracy, "
        "completeness, currentness, suitability, or validity of any information in this handbook "
        "and will not be liable for any errors, omissions, or delays in this information or any "
        "losses, injuries, or damages arising from its use.",
        normal_style))
    
    # Build the document
    doc.build(elements)
    print_status("Copyright page created successfully")
    return True

def create_toc():
    """Create a table of contents."""
    output_path = os.path.join(SUPPORTING_DIR, "02_toc.pdf")
    print_status(f"Creating table of contents: {output_path}")
    
    # Create document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='TOCTitle',
        parent=styles['Title'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    part_style = ParagraphStyle(
        name='PartHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10
    )
    
    chapter_style = ParagraphStyle(
        name='ChapterItem',
        parent=styles['Normal'],
        fontSize=12,
        leftIndent=20,
        spaceAfter=6
    )
    
    # TOC content
    elements = []
    
    # Add title
    elements.append(Paragraph("Table of Contents", title_style))
    
    # Add parts and representative chapters
    toc_data = [
        {"part": "I", "title": "Brains & Inspiration", "chapters": [
            "Chapter 1: Introduction to Neuroscience ↔ AI",
            "Chapter 2: Neuroscience Foundations",
            "Chapter 3: Spatial Navigation",
            "Chapter 4: Perception Pipeline"
        ]},
        {"part": "II", "title": "Brains Meet Math & Data", "chapters": [
            "Chapter 5: Brain Networks",
            "Chapter 6: Neurostimulation",
            "Chapter 7: Information Theory",
            "Chapter 8: Data Science Pipeline"
        ]},
        {"part": "III", "title": "Learning Machines", "chapters": [
            "Chapter 9: Machine Learning Foundations",
            "Chapter 10: Deep Learning",
            "Chapter 11: Sequence Models"
        ]},
        {"part": "IV", "title": "Frontier Models", "chapters": [
            "Chapter 12: Large Language Models",
            "Chapter 13: Multimodal Models"
        ]},
        {"part": "V", "title": "Ethics & Futures", "chapters": [
            "Chapter 15: Ethical AI",
            "Chapter 16: Future Directions"
        ]},
        {"part": "VI", "title": "Advanced Applications", "chapters": [
            "Chapter 17: BCI & Human-AI Interfaces",
            "Chapter 18: Neuromorphic Computing",
            "Chapter 19: Cognitive Neuroscience & Deep Learning",
            "Chapter 20: Case Studies",
            "Chapter 21: AI for Neuroscience Discovery",
            "Chapter 22: Embodied AI & Robotics",
            "Chapter 24: Quantum Computing in NeuroAI",
            "Chapter 23: Lifelong Learning"
        ]},
        {"part": "VII", "title": "Appendices", "chapters": [
            "Mathematical & Python Refresher",
            "Dataset Catalogue",
            "Colab Setup Guide"
        ]}
    ]
    
    # Add TOC entries
    for part in toc_data:
        # Add part title
        elements.append(Paragraph(f"Part {part['part']}: {part['title']}", part_style))
        
        # Add chapters
        for chapter in part["chapters"]:
            elements.append(Paragraph(f"{chapter} ................... ###", chapter_style))
    
    # Build the document
    doc.build(elements)
    print_status("Table of contents created successfully")
    return True

def create_part_dividers():
    """Create divider pages for each part."""
    print_status("Creating part divider pages")
    
    for part in PARTS:
        output_path = os.path.join(SUPPORTING_DIR, f"part{part['number']}_divider.pdf")
        print_status(f"Creating divider for Part {part['number']}: {part['title']}")
        
        # Create a canvas
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        
        # Add a decorative background
        c.setFillColorRGB(0.95, 0.95, 1.0)  # Very light blue
        c.rect(0, 0, width, height, fill=True)
        
        # Add a decorative line
        c.setStrokeColorRGB(0.2, 0.4, 0.8)  # Dark blue
        c.setLineWidth(2)
        c.line(width/4, height/2, 3*width/4, height/2)
        
        # Add part number
        c.setFont("Helvetica-Bold", 36)
        c.setFillColorRGB(0.2, 0.4, 0.8)  # Dark blue
        c.drawCentredString(width/2, height/2 + 50, f"Part {part['number']}")
        
        # Add part title
        c.setFont("Helvetica-Bold", 24)
        c.setFillColorRGB(0.3, 0.3, 0.5)  # Slightly lighter
        c.drawCentredString(width/2, height/2 - 50, part['title'])
        
        c.save()
    
    print_status("Part dividers created successfully")
    return True

def create_acknowledgments():
    """Create an acknowledgments page."""
    output_path = os.path.join(SUPPORTING_DIR, "acknowledgments.pdf")
    print_status(f"Creating acknowledgments page: {output_path}")
    
    # Create document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='AckTitle',
        parent=styles['Title'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    text_style = ParagraphStyle(
        name='AckText',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Content elements
    elements = []
    
    # Add some vertical space at the top
    elements.append(Spacer(1, 1*inch))
    
    # Title
    elements.append(Paragraph("Acknowledgments", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Acknowledgment text
    elements.append(Paragraph(
        "The creation of the NeuroAI Handbook would not have been possible without "
        "the contributions and support of many individuals and organizations. We wish "
        "to express our deepest gratitude to all who have helped make this project a reality.",
        text_style))
    
    elements.append(Paragraph(
        "We thank the authors and contributors who dedicated their time and expertise "
        "to create this comprehensive resource. Their commitment to bridging neuroscience "
        "and artificial intelligence has been the foundation of this handbook.",
        text_style))
    
    elements.append(Paragraph(
        "We also acknowledge the research institutions, laboratories, and funding agencies "
        "whose support has enabled the advancement of the field of NeuroAI. Their investment "
        "in interdisciplinary research has been crucial to the progress documented in this handbook.",
        text_style))
    
    elements.append(Paragraph(
        "Finally, we extend our appreciation to the readers and students who inspire us "
        "to create educational resources that facilitate learning and discovery at the "
        "intersection of neuroscience and artificial intelligence.",
        text_style))
    
    # Build the document
    doc.build(elements)
    print_status("Acknowledgments page created successfully")
    return True

def export_frontmatter():
    """Export frontmatter pages from JupyterBook to PDF.
    
    This function replaces the functionality of export_frontmatter.sh by:
    1. Building the frontmatter pages using jupyter-book
    2. Copying PDFs to the appropriate locations for inclusion in the final handbook
    """
    print_status("Exporting frontmatter pages from JupyterBook")
    
    # Create necessary directories
    os.makedirs(PDF_FRONTMATTER_DIR, exist_ok=True)
    
    # Define frontmatter files to process
    frontmatter_files = {
        "copyright": {"src": "frontmatter/copyright.md", "dest_name": "01_copyright.pdf"},
        "acknowledgments": {"src": "frontmatter/acknowledgments.md", "dest_name": "acknowledgments.pdf"},
        "about": {"src": "frontmatter/about.md", "dest_name": "about.pdf"}
    }
    
    success = True
    
    # Build each frontmatter file with jupyter-book
    for name, info in frontmatter_files.items():
        source_path = os.path.join(BOOK_DIR, info["src"])
        if not os.path.exists(source_path):
            print_status(f"WARNING: Frontmatter file not found: {source_path}")
            success = False
            continue
            
        print_status(f"Building {name} with jupyter-book")
        cmd = f"jupyter-book build {source_path} --builder pdfhtml"
        try:
            subprocess.run(cmd, shell=True, check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print_status(f"ERROR: Failed to build {name}: {e}")
            print_status(f"STDERR: {e.stderr.decode('utf-8')}")
            success = False
            continue
            
        # Find the PDF file that was created (in _build/_page)
        pdf_name = f"{os.path.basename(info['src']).replace('.md', '.pdf')}"
        build_dir = os.path.join(BOOK_DIR, "_build/_page")
        pdf_paths = glob.glob(f"{build_dir}/**/{pdf_name}", recursive=True)
        
        if not pdf_paths:
            print_status(f"ERROR: Could not find built PDF for {name}")
            success = False
            continue
            
        # Copy to frontmatter export directory
        export_path = os.path.join(PDF_FRONTMATTER_DIR, pdf_name)
        try:
            subprocess.run(f"cp {pdf_paths[0]} {export_path}", shell=True, check=True)
            print_status(f"Copied {pdf_name} to {PDF_FRONTMATTER_DIR}")
        except subprocess.CalledProcessError:
            print_status(f"ERROR: Failed to copy {pdf_name} to export directory")
            success = False
            
        # Also copy to supporting directory for merging
        supporting_path = os.path.join(SUPPORTING_DIR, info["dest_name"])
        try:
            subprocess.run(f"cp {pdf_paths[0]} {supporting_path}", shell=True, check=True)
            print_status(f"Copied {pdf_name} to {supporting_path}")
        except subprocess.CalledProcessError:
            print_status(f"ERROR: Failed to copy {pdf_name} to supporting directory")
            success = False
    
    if success:
        print_status("Successfully exported all frontmatter from JupyterBook")
    else:
        print_status("WARNING: Some frontmatter exports failed. Check output for details.")
        
    return success

def copy_to_exports(filename):
    """Copy a supporting file to the pdf_exports directory."""
    source = os.path.join(SUPPORTING_DIR, filename)
    destination = os.path.join(PDF_SUPPORTING_DIR, filename)
    
    if os.path.exists(source):
        cmd = f"cp {source} {destination}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print_status(f"Copied {filename} to {PDF_SUPPORTING_DIR}")
            return True
        except subprocess.CalledProcessError:
            print_status(f"WARNING: Failed to copy {filename} to exports directory")
            return False
    else:
        print_status(f"WARNING: Source file not found: {source}")
        return False

def build_all_supporting():
    """Create all supporting pages for the handbook."""
    print_status("BUILDING SUPPORTING PAGES", True)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUPPORTING_DIR, exist_ok=True)
    os.makedirs(PDF_SUPPORTING_DIR, exist_ok=True)
    os.makedirs(PDF_FRONTMATTER_DIR, exist_ok=True)
    
    print_status(f"Supporting pages will be saved to: {SUPPORTING_DIR}")
    print_status(f"Copies will also be saved to: {PDF_SUPPORTING_DIR}")
    print_status(f"Frontmatter pages will be saved to: {PDF_FRONTMATTER_DIR}")
    
    # Create all supporting pages
    success = (
        create_cover_page() and
        create_copyright_page() and
        create_toc() and
        create_part_dividers() and
        create_acknowledgments() and
        export_frontmatter()
    )
    
    # Copy files to pdf_exports directory
    if success:
        print_status("Copying files to exports directory...")
        
        # Copy cover page (special case, also copy to root of pdf_exports)
        cover_source = os.path.join(SUPPORTING_DIR, "00_cover.pdf")
        cover_dest = os.path.join(PDF_EXPORTS_DIR, "cover.pdf")
        if os.path.exists(cover_source):
            subprocess.run(f"cp {cover_source} {cover_dest}", shell=True, check=False)
            print_status(f"Copied cover to {cover_dest}")
        
        # Copy all files to supporting directory
        for file in os.listdir(SUPPORTING_DIR):
            if file.endswith(".pdf"):
                copy_to_exports(file)
    
    if success:
        print_status("All supporting pages created successfully", True)
        print(f"Files saved to: {SUPPORTING_DIR}")
        print(f"Copies also saved to: {PDF_SUPPORTING_DIR}")
    else:
        print_status("Failed to create some supporting pages", True)
    
    return success

if __name__ == "__main__":
    print_status("NeuroAI Handbook - Supporting Pages Builder", True)
    sys.exit(0 if build_all_supporting() else 1)