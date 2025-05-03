#!/usr/bin/env python3
"""
01_build_chapters.py - Build all content chapters and place them in the output folder.

This script:
1. Builds all individual chapter PDFs using jupyter-book
2. Creates a simple cover page
3. Places all PDFs in a designated output directory
"""

import os
import subprocess
import sys
import glob
import time

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BOOK_DIR = os.path.join(BASE_DIR, "book")
OUTPUT_DIR = os.path.join(BASE_DIR, "pdf_components")
CHAPTERS_DIR = os.path.join(OUTPUT_DIR, "chapters")
PDF_EXPORTS_DIR = os.path.join(BASE_DIR, "pdf_exports")
PDF_CHAPTERS_DIR = os.path.join(PDF_EXPORTS_DIR, "chapters")
ASSETS_DIR = os.path.join(BASE_DIR, "_assets")

# Chapter structure
CHAPTERS = [
    # Part 1: Brains & Inspiration
    {"path": "part1/ch01_intro.md", "order": "01"},
    {"path": "part1/ch02_neuro_foundations.md", "order": "02"},
    {"path": "part1/ch03_spatial_navigation.md", "order": "03"},
    {"path": "part1/ch04_perception_pipeline.md", "order": "04"},
    
    # Part 2: Brains Meet Math & Data
    {"path": "part2/ch05_brain_networks.md", "order": "05"},
    {"path": "part2/ch06_neurostimulation.md", "order": "06"},
    {"path": "part2/ch07_information_theory.md", "order": "07"},
    {"path": "part2/ch08_data_science_pipeline.md", "order": "08"},
    
    # Part 3: Learning Machines
    {"path": "part3/ch09_ml_foundations.md", "order": "09"},
    {"path": "part3/ch10_deep_learning.md", "order": "10"},
    {"path": "part3/ch11_sequence_models.md", "order": "11"},
    
    # Part 4: Frontier Models
    {"path": "part4/ch12_large_language_models.md", "order": "12"},
    {"path": "part4/ch13_multimodal_models.md", "order": "13"},
    
    # Part 5: Ethics & Futures
    {"path": "part5/ch15_ethical_ai.md", "order": "15"},
    {"path": "part5/ch16_future_directions.md", "order": "16"},
    
    # Part 6: Advanced Applications
    {"path": "part6/ch17_bci_human_ai_interfaces.md", "order": "17"},
    {"path": "part6/ch18_neuromorphic_computing.md", "order": "18"},
    {"path": "part6/ch19_cognitive_neuro_dl.md", "order": "19"},
    {"path": "part6/ch20_case_studies.md", "order": "20"},
    {"path": "part6/ch21_ai_for_neuro_discovery.md", "order": "21"},
    {"path": "part6/ch22_embodied_ai_robotics.md", "order": "22"},
    {"path": "part6/ch23_lifelong_learning.md", "order": "23"},
    {"path": "part6/ch24_quantum_computing_neuroai.md", "order": "24"},
    
    # Appendices
    {"path": "appendices/glossary.md", "order": "A1"},
    {"path": "appendices/math_python_refresher.md", "order": "A2"},
    {"path": "appendices/dataset_catalogue.md", "order": "A3"},
    {"path": "appendices/colab_setup.md", "order": "A4"}
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

def run_command(command):
    """Run a shell command and return its success status."""
    print(f"[CMD] {command}")
    try:
        subprocess.run(command, shell=True, check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        print(f"[STDERR] {e.stderr.decode('utf-8')}")
        return False

def build_chapter(chapter_info):
    """Build a single chapter PDF."""
    order = chapter_info["order"]
    path = chapter_info["path"]
    
    # Get the chapter name from the path (without .md extension)
    chapter_name = os.path.basename(path).replace(".md", "")
    
    # Full path to the chapter file
    chapter_path = os.path.join(BOOK_DIR, path)
    
    if not os.path.exists(chapter_path):
        print(f"[ERROR] Chapter file not found: {chapter_path}")
        return False
    
    # Build the chapter
    print_status(f"Building chapter {order}: {chapter_name}")
    cmd = f"cd {BASE_DIR} && jupyter-book build {chapter_path} --builder pdfhtml"
    if not run_command(cmd):
        return False
    
    # Find and copy the built PDF
    pdf_name = f"{chapter_name}.pdf"
    build_dir = os.path.join(BOOK_DIR, "_build/_page")
    
    # Search for the built PDF (recursively)
    pdf_paths = glob.glob(f"{build_dir}/**/{pdf_name}", recursive=True)
    if not pdf_paths:
        print(f"[ERROR] Could not find built PDF for {chapter_name}")
        return False
    
    # Rename with order prefix and copy to output directories
    output_pdf = os.path.join(CHAPTERS_DIR, f"{order}_{chapter_name}.pdf")
    exports_pdf = os.path.join(PDF_CHAPTERS_DIR, f"{chapter_name}.pdf")
    
    # Copy to output directory (for our build process)
    cmd = f"cp {pdf_paths[0]} {output_pdf}"
    if not run_command(cmd):
        return False
    
    # Also copy to pdf_exports/chapters for reference
    cmd = f"cp {pdf_paths[0]} {exports_pdf}"
    if not run_command(cmd):
        print(f"[WARNING] Could not copy to {exports_pdf}, but continuing")
    
    print_status(f"Successfully built chapter {order}: {chapter_name}")
    return True

def create_simple_cover():
    """Create a simple cover page PDF."""
    print_status("Creating simple cover page")
    
    # Check if we can use a custom cover from assets directory
    custom_cover = os.path.join(ASSETS_DIR, "cover.pdf")
    if os.path.exists(custom_cover):
        print_status(f"Using custom cover from assets: {custom_cover}")
        output_cover = os.path.join(OUTPUT_DIR, "00_cover.pdf")
        cmd = f"cp {custom_cover} {output_cover}"
        
        # Also copy to pdf_exports for reference
        exports_cover = os.path.join(PDF_EXPORTS_DIR, "cover.pdf")
        if run_command(cmd):
            cmd = f"cp {custom_cover} {exports_cover}"
            run_command(cmd)
            return True
    
    # Check if we can use the existing cover
    existing_cover = os.path.join(BASE_DIR, "pdf_exports/cover.pdf")
    if os.path.exists(existing_cover):
        output_cover = os.path.join(OUTPUT_DIR, "00_cover.pdf")
        cmd = f"cp {existing_cover} {output_cover}"
        return run_command(cmd)
    
    # If no existing cover, use the first page of the first chapter as placeholder
    print_status("No cover found, using chapter 1 first page as placeholder")
    first_chapter = os.path.join(CHAPTERS_DIR, "01_ch01_intro.pdf")
    if os.path.exists(first_chapter):
        output_cover = os.path.join(OUTPUT_DIR, "00_cover.pdf")
        # Extract just the first page
        cmd = f"cd {BASE_DIR} && python -c \"from PyPDF2 import PdfWriter, PdfReader; writer = PdfWriter(); writer.add_page(PdfReader('{first_chapter}').pages[0]); writer.write('{output_cover}')\""
        return run_command(cmd)
    
    print_status("WARNING: Could not create a cover page")
    return False

def build_all_chapters():
    """Build all chapters and organize them in the output directory."""
    print_status("BUILDING ALL CHAPTERS", True)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHAPTERS_DIR, exist_ok=True)
    os.makedirs(PDF_CHAPTERS_DIR, exist_ok=True)
    
    print_status(f"Chapters will be saved to: {CHAPTERS_DIR}")
    print_status(f"Chapter copies will also be saved to: {PDF_CHAPTERS_DIR}")
    
    # Build each chapter
    success_count = 0
    failure_count = 0
    
    start_time = time.time()
    for i, chapter in enumerate(CHAPTERS):
        print_status(f"Processing chapter {i+1} of {len(CHAPTERS)}: {chapter['path']}")
        if build_chapter(chapter):
            success_count += 1
        else:
            failure_count += 1
    
    # Create a simple cover page
    create_simple_cover()
    
    # Print summary
    elapsed_time = time.time() - start_time
    print_status("BUILD SUMMARY", True)
    print(f"Successfully built: {success_count} chapters")
    print(f"Failed to build: {failure_count} chapters")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"All built chapters are in: {CHAPTERS_DIR}")
    print(f"Clean copies are also in: {PDF_CHAPTERS_DIR}")
    
    return success_count > 0

if __name__ == "__main__":
    print_status("NeuroAI Handbook - Chapter Builder", True)
    sys.exit(0 if build_all_chapters() else 1)