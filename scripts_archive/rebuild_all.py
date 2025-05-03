#!/usr/bin/env python3
"""
Comprehensive script to rebuild all chapters and create a fresh PDF.
The process includes:
1. Rebuilding all individual chapter PDFs
2. Creating a fresh PDF with all content
"""

import os
import subprocess
import sys
import time
import glob
from PyPDF2 import PdfWriter, PdfReader
import yaml

# Configuration
BASE_DIR = "/Volumes/data_bolt/_github/NeuroAI-Handbook"
BOOK_DIR = os.path.join(BASE_DIR, "book")
PDF_DIR = os.path.join(BASE_DIR, "pdf_exports")
OUTPUT_DIR = os.path.join(PDF_DIR, "with_covers")
FRESH_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_fresh.pdf")

# Chapter paths from TOC
TOC_FILE = os.path.join(BOOK_DIR, "_toc.yml")

# Parts structure for organization
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

def run_command(command, description=None):
    """Run a shell command and print output."""
    if description:
        print(f"[RUNNING] {description}")
    else:
        print(f"[COMMAND] {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True)
        print(f"[SUCCESS] {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        print(f"[STDERR] {e.stderr.strip()}")
        return False

def get_chapter_paths():
    """Get chapter paths from _toc.yml."""
    all_chapters = []
    for part in PARTS:
        all_chapters.extend(part["chapters"])
    return all_chapters

def rebuild_all_chapters():
    """Rebuild all chapters using jupyter-book."""
    print_header("REBUILDING ALL CHAPTERS")
    
    # Ensure output directories exist
    os.makedirs(PDF_DIR, exist_ok=True)
    
    chapter_paths = get_chapter_paths()
    success_count = 0
    failure_count = 0
    
    for chapter in chapter_paths:
        # Determine the actual file path
        if chapter.startswith("ch"):
            # Regular chapter
            part_num = int(chapter[2]) if chapter[2].isdigit() else 1
            file_path = f"part{part_num}/{chapter}.md"
        else:
            # Appendix
            file_path = f"appendices/{chapter}.md"
        
        full_path = os.path.join(BOOK_DIR, file_path)
        
        if not os.path.exists(full_path):
            print(f"[WARNING] Chapter file not found: {full_path}")
            continue
        
        print(f"[INFO] Building chapter: {chapter}")
        
        # Run jupyter-book build command
        cmd = f"cd {BASE_DIR} && jupyter-book build {full_path} --builder pdfhtml"
        if run_command(cmd, f"Building {chapter}"):
            # Copy the PDF to the pdf_exports directory
            pdf_name = f"{chapter}.pdf"
            build_dir = os.path.join(BOOK_DIR, "_build/_page")
            
            # Find the PDF in the build directory
            pdf_paths = glob.glob(f"{build_dir}/**/{pdf_name}", recursive=True)
            if pdf_paths:
                pdf_path = pdf_paths[0]
                # Copy to pdf_exports
                if run_command(f"cp {pdf_path} {PDF_DIR}/", f"Copying {pdf_name}"):
                    print(f"[SUCCESS] Successfully rebuilt {chapter}")
                    success_count += 1
                else:
                    print(f"[ERROR] Failed to copy {pdf_name}")
                    failure_count += 1
            else:
                print(f"[ERROR] Could not find built PDF for {chapter}")
                failure_count += 1
        else:
            print(f"[ERROR] Failed to build {chapter}")
            failure_count += 1
        
        # Add a separator for readability
        print("\n" + "-" * 80 + "\n")
    
    print_header("REBUILD SUMMARY")
    print(f"Successfully rebuilt: {success_count} chapters")
    print(f"Failed to rebuild: {failure_count} chapters")
    
    return success_count > 0

def create_fresh_pdf():
    """Create a fresh PDF from the rebuilt chapters."""
    print_header("CREATING FRESH PDF")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a simple PDF writer
    output = PdfWriter()
    
    # Add cover page if it exists
    cover_path = os.path.join(PDF_DIR, "cover.pdf")
    if os.path.exists(cover_path):
        print(f"[INFO] Adding cover from {cover_path}")
        with open(cover_path, 'rb') as f:
            pdf = PdfReader(f)
            output.add_page(pdf.pages[0])
    
    # Add each part and its chapters
    for part in PARTS:
        print(f"[INFO] Adding chapters from {part['title']}")
        
        for chapter in part["chapters"]:
            chapter_path = os.path.join(PDF_DIR, f"{chapter}.pdf")
            
            if os.path.exists(chapter_path):
                print(f"[INFO] Adding chapter: {chapter}")
                with open(chapter_path, 'rb') as f:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        output.add_page(page)
            else:
                print(f"[WARNING] Chapter PDF not found: {chapter_path}")
    
    # Write the final PDF
    print(f"[INFO] Writing final PDF with {len(output.pages)} pages")
    with open(FRESH_PDF, 'wb') as f:
        output.write(f)
    
    print(f"[SUCCESS] Created fresh PDF at {FRESH_PDF}")
    return True

if __name__ == "__main__":
    print_header("NEUROAI HANDBOOK REBUILD PROCESS")
    
    # Rebuild all chapters
    if rebuild_all_chapters():
        # Create fresh PDF
        create_fresh_pdf()
        print_header("REBUILD COMPLETE")
        print(f"Fresh PDF created at: {FRESH_PDF}")
    else:
        print_header("REBUILD FAILED")
        print("Failed to rebuild chapters. Please check the errors above.")
        sys.exit(1)