#!/usr/bin/env python3
"""
03_merge_final.py - Merge all components into the final handbook PDF.

This script:
1. Takes all components (cover, TOC, chapters, etc.)
2. Combines them in the correct order
3. Creates a single comprehensive PDF with page numbers 
4. Uses Acrobat-compatible methods for maximum compatibility

Usage:
  python 03_merge_final.py [--output OUTPUT_DIR]

Options:
  --output OUTPUT_DIR  Path to output directory (default: pdf_exports/complete_handbook)
"""

import os
import sys
import glob
import argparse
import logging
import datetime
import pathlib
import io
import shutil
from pypdf import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PDF_COMPONENTS = os.path.join(BASE_DIR, "pdf_components")
CHAPTERS_DIR = os.path.join(PDF_COMPONENTS, "chapters")
SUPPORTING_DIR = os.path.join(PDF_COMPONENTS, "supporting")
PDF_EXPORTS_DIR = os.path.join(BASE_DIR, "pdf_exports")
OUTPUT_DIR = os.path.join(PDF_EXPORTS_DIR, "complete_handbook")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ASSETS_DIR = os.path.join(BASE_DIR, "_assets")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Configure logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"merge_final_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Output file names
BASIC_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_basic.pdf")
NUMBERED_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_numbered.pdf")
ACROBAT_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_acrobat.pdf")

# Part structure (for correct ordering)
STRUCTURE = [
    # Front matter
    {"type": "supporting", "file": "00_cover.pdf", "required": True},
    {"type": "supporting", "file": "01_copyright.pdf", "required": True},
    {"type": "supporting", "file": "02_toc.pdf", "required": True},
    {"type": "supporting", "file": "acknowledgments.pdf", "required": False},
    {"type": "supporting", "file": "about.pdf", "required": False},
    
    # Part I
    {"type": "supporting", "file": "partI_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "01_*", "required": True},
    {"type": "chapter_pattern", "pattern": "02_*", "required": True},
    {"type": "chapter_pattern", "pattern": "03_*", "required": True},
    {"type": "chapter_pattern", "pattern": "04_*", "required": True},
    
    # Part II
    {"type": "supporting", "file": "partII_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "05_*", "required": True},
    {"type": "chapter_pattern", "pattern": "06_*", "required": True},
    {"type": "chapter_pattern", "pattern": "07_*", "required": True},
    {"type": "chapter_pattern", "pattern": "08_*", "required": True},
    
    # Part III
    {"type": "supporting", "file": "partIII_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "09_*", "required": True},
    {"type": "chapter_pattern", "pattern": "10_*", "required": True},
    {"type": "chapter_pattern", "pattern": "11_*", "required": True},
    
    # Part IV
    {"type": "supporting", "file": "partIV_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "12_*", "required": True},
    {"type": "chapter_pattern", "pattern": "13_*", "required": True},
    
    # Part V
    {"type": "supporting", "file": "partV_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "15_*", "required": True},
    {"type": "chapter_pattern", "pattern": "16_*", "required": True},
    
    # Part VI
    {"type": "supporting", "file": "partVI_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "17_*", "required": True},
    {"type": "chapter_pattern", "pattern": "18_*", "required": True},
    {"type": "chapter_pattern", "pattern": "19_*", "required": True},
    {"type": "chapter_pattern", "pattern": "20_*", "required": True},
    {"type": "chapter_pattern", "pattern": "21_*", "required": True},
    {"type": "chapter_pattern", "pattern": "22_*", "required": True},
    {"type": "chapter_pattern", "pattern": "24_*", "required": True},
    {"type": "chapter_pattern", "pattern": "23_*", "required": True},
    
    # Appendices
    {"type": "supporting", "file": "partVII_divider.pdf", "required": False},
    {"type": "chapter_pattern", "pattern": "A1_*", "required": False},
    {"type": "chapter_pattern", "pattern": "A2_*", "required": True},
    {"type": "chapter_pattern", "pattern": "A3_*", "required": True},
    {"type": "chapter_pattern", "pattern": "A4_*", "required": True}
]

def print_status(message, is_header=False):
    """Print a formatted status message and log it."""
    if is_header:
        sep = "=" * 80
        header = f"\n{sep}\n{message.center(78)}\n{sep}\n"
        print(header)
        logging.info(f"HEADER: {message}")
    else:
        print(f"[INFO] {message}")
        logging.info(message)

def add_page_numbers(src_path, dst_path, start_page=3):
    """
    Add page numbers to a PDF, skipping the first few pages (front matter).
    Uses a safer pattern that avoids Acrobat compatibility issues.
    """
    print_status(f"Adding page numbers to PDF: {os.path.basename(src_path)}")
    
    # Create a new writer
    writer = PdfWriter()
    
    # First, copy all pages without modification
    reader = PdfReader(src_path)
    total_pages = len(reader.pages)
    
    print_status(f"Copying {total_pages} pages from source document")
    for page in reader.pages:
        writer.add_page(page)
    
    # Now add page numbers to each page after start_page
    print_status(f"Adding page numbers to {total_pages - start_page} pages, starting after page {start_page}")
    for i in range(start_page, total_pages):
        # Create a PDF with just the page number
        packet = io.BytesIO()
        # Use the page's mediabox size instead of hardcoded letter size
        page_size = writer.pages[i].mediabox
        width = float(page_size.width)
        
        c = canvas.Canvas(packet, pagesize=(width, float(page_size.height)))
        c.setFont("Helvetica", 10)
        c.drawCentredString(width/2, 30, str(i - start_page + 1))
        c.save()
        
        # Add the page number to the page
        packet.seek(0)
        overlay = PdfReader(packet).pages[0]
        writer.pages[i].merge_page(overlay)
        writer.pages[i].compress_content_streams()  # Compress to keep file size small
        
        # Progress indicator for large documents
        if (i - start_page + 1) % 50 == 0:
            print_status(f"Processed {i - start_page + 1} pages...")
    
    # Make sure the output directory exists
    pathlib.Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write the output file
    print_status(f"Writing file with page numbers to: {dst_path}")
    with open(dst_path, "wb") as f:
        writer.write(f)
    
    return True

def find_component_files():
    """Find all component files needed for the handbook."""
    print_status("Identifying component files")
    
    files_to_merge = []
    errors = False
    
    # Verify the special ch24/ch23 ordering to ensure it's correct
    ch24_position = -1
    ch23_position = -1
    
    for idx, item in enumerate(STRUCTURE):
        if item.get("type") == "chapter_pattern" and item.get("pattern") == "24_*":
            ch24_position = idx
        if item.get("type") == "chapter_pattern" and item.get("pattern") == "23_*":
            ch23_position = idx
    
    if ch24_position > 0 and ch23_position > 0:
        if ch24_position > ch23_position:
            print_status("WARNING: Chapter order issue - ch23 comes before ch24 in STRUCTURE")
            print_status("         Please check the STRUCTURE list to ensure ch24 comes before ch23")
    
    # Process files in the defined order
    for item in STRUCTURE:
        item_type = item["type"]
        required = item["required"]
        
        if item_type == "supporting":
            # Find a specific supporting file
            file_path = os.path.join(SUPPORTING_DIR, item["file"])
            if os.path.exists(file_path):
                files_to_merge.append(file_path)
                print_status(f"Found supporting file: {item['file']}")
            elif required:
                print_status(f"ERROR: Required supporting file not found: {item['file']}")
                errors = True
            else:
                print_status(f"Optional supporting file not found: {item['file']}")
        
        elif item_type == "chapter_pattern":
            # Find chapters matching a pattern
            pattern = os.path.join(CHAPTERS_DIR, item["pattern"])
            matching_files = sorted(glob.glob(pattern))
            
            if matching_files:
                # Add the first match (should be only one per pattern)
                files_to_merge.append(matching_files[0])
                print_status(f"Found chapter file: {os.path.basename(matching_files[0])}")
            elif required:
                print_status(f"ERROR: Required chapter not found: {item['pattern']}")
                errors = True
            else:
                print_status(f"Optional chapter not found: {item['pattern']}")
    
    if errors:
        print_status("WARNING: Some required components were not found")
    
    return files_to_merge, not errors

def create_basic_pdf(files):
    """Create a basic PDF combining all components using PdfWriter."""
    print_status("Creating basic PDF (no page numbers)")
    
    # Use PdfWriter for better Acrobat compatibility
    writer = PdfWriter()
    
    # Add each file
    for file_path in files:
        print_status(f"Adding file: {os.path.basename(file_path)}")
        reader = PdfReader(file_path)
        for page in reader.pages:
            writer.add_page(page)
    
    # Write the output file
    os.makedirs(os.path.dirname(BASIC_PDF), exist_ok=True)
    with open(BASIC_PDF, 'wb') as f:
        writer.write(f)
    
    print_status(f"Basic PDF created with {len(writer.pages)} pages: {BASIC_PDF}")
    return True

def create_numbered_pdf(files):
    """Create a PDF with page numbers using the safer pattern."""
    print_status("Creating PDF with page numbers")
    
    # First merge all files into a temporary PDF without page numbers
    temp_pdf = os.path.join(OUTPUT_DIR, "temp_merged.pdf")
    
    # Use PdfWriter for better Acrobat compatibility
    writer = PdfWriter()
    
    # Add each file
    for file_path in files:
        print_status(f"Adding file: {os.path.basename(file_path)}")
        reader = PdfReader(file_path)
        for page in reader.pages:
            writer.add_page(page)
    
    # Write the merged file first
    os.makedirs(os.path.dirname(temp_pdf), exist_ok=True)
    with open(temp_pdf, 'wb') as f:
        writer.write(f)
    
    # Now add page numbers to the merged file
    print_status("Adding page numbers to merged PDF")
    success = add_page_numbers(temp_pdf, NUMBERED_PDF, start_page=3)
    
    # Clean up the temporary file
    if success and os.path.exists(temp_pdf):
        os.remove(temp_pdf)
    
    print_status(f"Numbered PDF created: {NUMBERED_PDF}")
    return success

def create_acrobat_compatible(files):
    """Create an Acrobat-compatible PDF (no page modifications) using PdfWriter."""
    print_status("Creating Acrobat-compatible PDF")
    
    # For Acrobat compatibility, use PdfWriter and avoid any page modifications
    writer = PdfWriter()
    
    # Add each file
    for file_path in files:
        print_status(f"Adding file: {os.path.basename(file_path)}")
        reader = PdfReader(file_path)
        for page in reader.pages:
            writer.add_page(page)
    
    # Write the output file
    os.makedirs(os.path.dirname(ACROBAT_PDF), exist_ok=True)
    with open(ACROBAT_PDF, 'wb') as f:
        writer.write(f)
    
    print_status(f"Acrobat-compatible PDF created with {len(writer.pages)} pages: {ACROBAT_PDF}")
    return True

def merge_final_handbook(basic_only=False, acrobat_only=False):
    """Create the final handbook PDF."""
    print_status("CREATING NEUROAI HANDBOOK", True)
    
    # Find all component files
    files, all_found = find_component_files()
    
    if not files:
        print_status("ERROR: No component files found. Run scripts 01 and 02 first.")
        return False
    
    if not all_found:
        print_status("WARNING: Some components are missing, but continuing anyway.")
    
    # Create a single, comprehensive PDF with page numbers
    final_pdf = os.path.join(OUTPUT_DIR, "neuroai_handbook.pdf")
    
    # Ignore legacy options
    if basic_only or acrobat_only:
        print_status("Note: Creating only the comprehensive version with page numbers (legacy options ignored)")
    
    # Create the merged file without page numbers first
    print_status("Creating comprehensive PDF...")
    temp_pdf = os.path.join(OUTPUT_DIR, "temp_merged.pdf")
    
    # Use PdfWriter for better Acrobat compatibility
    writer = PdfWriter()
    
    # Add each file
    for file_path in files:
        print_status(f"Adding file: {os.path.basename(file_path)}")
        reader = PdfReader(file_path)
        for page in reader.pages:
            writer.add_page(page)
    
    # Write the merged file first
    os.makedirs(os.path.dirname(temp_pdf), exist_ok=True)
    with open(temp_pdf, 'wb') as f:
        writer.write(f)
    
    # Now add page numbers to the merged file
    print_status("Adding page numbers to PDF...")
    success = add_page_numbers(temp_pdf, final_pdf, start_page=3)
    
    # Clean up the temporary file
    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)
    
    # Create a qpdf check if available
    try:
        import subprocess
        result = subprocess.run(["which", "qpdf"], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("Running QPDF check to validate PDF...")
            check_result = subprocess.run(["qpdf", "--check", final_pdf], 
                                        capture_output=True, text=True)
            if check_result.returncode == 0:
                print_status("QPDF check passed: The PDF should work well in all viewers")
            else:
                print_status(f"WARNING: QPDF check reported issues: {check_result.stderr}")
    except Exception as e:
        pass  # Silently continue if qpdf is not available
    
    print_status("BUILD COMPLETE", True)
    print(f"Final handbook PDF created: {final_pdf}")
    
    return success

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge components into final handbook PDF")
    parser.add_argument("--output", help="Path to output directory (default: pdf_exports/complete_handbook)")
    return parser.parse_args()

if __name__ == "__main__":
    print_status("NeuroAI Handbook - Final Merge", True)
    
    args = parse_arguments()
    
    # Update output directory if specified
    if args.output:
        OUTPUT_DIR = os.path.abspath(args.output)
        # Update the final PDF path - we now only create one version
        final_pdf = os.path.join(OUTPUT_DIR, "neuroai_handbook.pdf")
        print_status(f"Using custom output directory: {OUTPUT_DIR}")
    
    print_status(f"Log file created at: {log_file}")
    success = merge_final_handbook()
    
    if success:
        print_status("PDF creation completed successfully")
        logging.info("PDF creation completed successfully")
    else:
        print_status("ERROR: PDF creation failed")
        logging.error("PDF creation failed")
    
    sys.exit(0 if success else 1)