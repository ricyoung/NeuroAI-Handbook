#!/usr/bin/env python3
"""
Minimal script to build the NeuroAI Handbook PDF using basic PyPDF2 operations.
Uses minimal processing to avoid compatibility issues.
"""

import os
import io
from PyPDF2 import PdfWriter, PdfReader

# Configuration
PDF_DIR = "pdf_exports"
COVER_PDF = os.path.join(PDF_DIR, "cover.pdf")
OUTPUT_PDF = os.path.join(PDF_DIR, "neuroai_minimal.pdf")

# Chapter order (keep it simple)
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

def build_minimal_pdf():
    """Build a minimal PDF with basic PyPDF2 operations."""
    print(f"Building minimal PDF to {OUTPUT_PDF}")
    
    # Create PDF writer
    output = PdfWriter()
    
    # Add cover page if it exists
    if os.path.exists(COVER_PDF):
        print(f"Adding cover from {COVER_PDF}")
        with open(COVER_PDF, 'rb') as f:
            cover_pdf = PdfReader(f)
            for page in cover_pdf.pages:
                output.add_page(page)
    
    # Add each chapter in order
    for chapter in CHAPTERS:
        chapter_pdf = os.path.join(PDF_DIR, f"{chapter}.pdf")
        if os.path.exists(chapter_pdf):
            print(f"Adding chapter: {chapter}")
            with open(chapter_pdf, 'rb') as f:
                pdf = PdfReader(f)
                for page in pdf.pages:
                    output.add_page(page)
        else:
            print(f"Warning: Chapter {chapter} not found at {chapter_pdf}")
    
    # Write the final PDF
    print(f"Writing final PDF with {len(output.pages)} pages")
    with open(OUTPUT_PDF, 'wb') as output_file:
        output.write(output_file)
    
    print(f"Created PDF at {OUTPUT_PDF}")

if __name__ == "__main__":
    build_minimal_pdf()