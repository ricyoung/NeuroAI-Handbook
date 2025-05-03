#!/usr/bin/env python3
"""
Most simple approach to combine PDFs into a complete book
"""

import os
from PyPDF2 import PdfWriter, PdfReader

# Input files
COVER_PDF = "/Volumes/data_bolt/_github/NeuroAI-Handbook/pdf_exports/cover.pdf"
CHAPTERS_DIR = "/Volumes/data_bolt/_github/NeuroAI-Handbook/pdf_exports"
OUTPUT_PDF = "/Volumes/data_bolt/_github/NeuroAI-Handbook/pdf_exports/handbook_fixed.pdf"

# Order of chapters (without .pdf extension)
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

def combine_pdfs():
    """Combine PDFs in the specified order"""
    output = PdfWriter()
    
    # Add cover if it exists
    if os.path.exists(COVER_PDF):
        print(f"Adding cover: {COVER_PDF}")
        with open(COVER_PDF, 'rb') as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                output.add_page(page)
    
    # Add chapters in the specified order
    for chapter in CHAPTERS:
        chapter_pdf = os.path.join(CHAPTERS_DIR, f"{chapter}.pdf")
        if os.path.exists(chapter_pdf):
            print(f"Adding chapter: {chapter}")
            with open(chapter_pdf, 'rb') as f:
                pdf = PdfReader(f)
                for page in pdf.pages:
                    output.add_page(page)
        else:
            print(f"Warning: Chapter PDF not found: {chapter_pdf}")
    
    # Write the combined PDF
    print(f"Writing final PDF: {OUTPUT_PDF}")
    with open(OUTPUT_PDF, 'wb') as f:
        output.write(f)
    
    print(f"Created complete book with {len(output.pages)} pages")

if __name__ == "__main__":
    combine_pdfs()