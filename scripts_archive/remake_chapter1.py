#!/usr/bin/env python3
"""
Script to rebuild just Chapter 1 and create a new PDF with all chapters.
"""

import os
import subprocess
import sys
from PyPDF2 import PdfWriter, PdfReader

# Configuration
BASE_DIR = "/Volumes/data_bolt/_github/NeuroAI-Handbook"
BOOK_DIR = os.path.join(BASE_DIR, "book")
PDF_DIR = os.path.join(BASE_DIR, "pdf_exports")
OUTPUT_DIR = os.path.join(PDF_DIR, "with_covers")
FINAL_PDF = os.path.join(OUTPUT_DIR, "neuroai_handbook_fresh.pdf")

# Chapters to include in order
CHAPTERS = [
    "ch01_intro",
    "ch02_neuro_foundations",
    "ch03_spatial_navigation",
    "ch04_perception_pipeline",
    "ch05_brain_networks",
    "ch06_neurostimulation",
    "ch07_information_theory",
    "ch08_data_science_pipeline",
    "ch09_ml_foundations",
    "ch10_deep_learning",
    "ch11_sequence_models",
    "ch12_large_language_models",
    "ch13_multimodal_models",
    "ch15_ethical_ai",
    "ch16_future_directions",
    "ch17_bci_human_ai_interfaces",
    "ch18_neuromorphic_computing",
    "ch19_cognitive_neuro_dl",
    "ch20_case_studies",
    "ch21_ai_for_neuro_discovery",
    "ch22_embodied_ai_robotics",
    "ch24_quantum_computing_neuroai", 
    "ch23_lifelong_learning",
    "math_python_refresher",
    "dataset_catalogue",
    "colab_setup"
]

def print_status(message):
    """Print a status message."""
    print(f"[INFO] {message}")

def rebuild_chapter_one():
    """Rebuild chapter one specifically."""
    print_status("Rebuilding Chapter 1...")
    
    chapter_file = os.path.join(BOOK_DIR, "part1/ch01_intro.md")
    
    # Run the build command
    cmd = f"cd {BASE_DIR} && jupyter-book build {chapter_file} --builder pdfhtml"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True)
        print_status("Build command completed successfully")
        
        # Find the output PDF
        output_pdf = os.path.join(BOOK_DIR, "_build/_page/part1-ch01_intro/pdf/ch01_intro.pdf")
        
        if os.path.exists(output_pdf):
            # Copy to pdf_exports
            target_path = os.path.join(PDF_DIR, "ch01_intro.pdf")
            cmd_copy = f"cp {output_pdf} {target_path}"
            subprocess.run(cmd_copy, shell=True, check=True)
            print_status(f"Copied rebuilt Chapter 1 to {target_path}")
            return True
        else:
            print_status(f"ERROR: Built PDF not found at {output_pdf}")
            return False
    
    except subprocess.CalledProcessError as e:
        print_status(f"ERROR: Build command failed with exit code {e.returncode}")
        print(f"Error details: {e.stderr}")
        return False

def create_new_pdf():
    """Create a new combined PDF with all chapters."""
    print_status("Creating new combined PDF...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create PDF writer
    output = PdfWriter()
    
    # Add cover if it exists
    cover_path = os.path.join(PDF_DIR, "cover.pdf")
    if os.path.exists(cover_path):
        print_status(f"Adding cover from {cover_path}")
        with open(cover_path, 'rb') as f:
            pdf = PdfReader(f)
            output.add_page(pdf.pages[0])
    
    # Add each chapter
    for chapter in CHAPTERS:
        chapter_path = os.path.join(PDF_DIR, f"{chapter}.pdf")
        
        if os.path.exists(chapter_path):
            print_status(f"Adding chapter: {chapter}")
            with open(chapter_path, 'rb') as f:
                pdf = PdfReader(f)
                for page in pdf.pages:
                    output.add_page(page)
        else:
            print_status(f"Warning: Chapter PDF not found: {chapter_path}")
    
    # Write the final PDF
    print_status(f"Writing final PDF with {len(output.pages)} pages")
    with open(FINAL_PDF, 'wb') as f:
        output.write(f)
    
    print_status(f"Created new PDF at {FINAL_PDF}")
    return True

if __name__ == "__main__":
    print_status("REBUILDING CHAPTER 1 AND CREATING NEW PDF")
    
    # Rebuild Chapter 1
    success = rebuild_chapter_one()
    
    if success:
        # Create new PDF
        create_new_pdf()
        print_status("PROCESS COMPLETE")
    else:
        print_status("PROCESS FAILED - Could not rebuild Chapter 1")
        sys.exit(1)