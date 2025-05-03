#!/bin/bash
# Script to export all chapters of the NeuroAI-Handbook to PDF

# Create output directory
mkdir -p pdf_exports

# Part 1: Brains & Inspiration
echo "Exporting Part 1 chapters..."
jupyter-book build book/part1/ch01_intro.md --builder pdfhtml
jupyter-book build book/part1/ch02_neuro_foundations.md --builder pdfhtml
jupyter-book build book/part1/ch03_spatial_navigation.md --builder pdfhtml
jupyter-book build book/part1/ch04_perception_pipeline.md --builder pdfhtml

# Part 2: Brains Meet Math & Data
echo "Exporting Part 2 chapters..."
jupyter-book build book/part2/ch05_brain_networks.md --builder pdfhtml
jupyter-book build book/part2/ch06_neurostimulation.md --builder pdfhtml
jupyter-book build book/part2/ch07_information_theory.md --builder pdfhtml
jupyter-book build book/part2/ch08_data_science_pipeline.md --builder pdfhtml

# Part 3: Learning Machines
echo "Exporting Part 3 chapters..."
jupyter-book build book/part3/ch09_ml_foundations.md --builder pdfhtml
jupyter-book build book/part3/ch10_deep_learning.md --builder pdfhtml
jupyter-book build book/part3/ch11_sequence_models.md --builder pdfhtml

# Part 4: Frontier Models
echo "Exporting Part 4 chapters..."
jupyter-book build book/part4/ch12_large_language_models.md --builder pdfhtml
jupyter-book build book/part4/ch13_multimodal_models.md --builder pdfhtml

# Part 5: Ethics & Futures
echo "Exporting Part 5 chapters..."
jupyter-book build book/part5/ch15_ethical_ai.md --builder pdfhtml
jupyter-book build book/part5/ch16_future_directions.md --builder pdfhtml

# Part 6: Advanced Applications
echo "Exporting Part 6 chapters..."
jupyter-book build book/part6/ch17_bci_human_ai_interfaces.md --builder pdfhtml
jupyter-book build book/part6/ch18_neuromorphic_computing.md --builder pdfhtml
jupyter-book build book/part6/ch19_cognitive_neuro_dl.md --builder pdfhtml
jupyter-book build book/part6/ch20_case_studies.md --builder pdfhtml
jupyter-book build book/part6/ch21_ai_for_neuro_discovery.md --builder pdfhtml
jupyter-book build book/part6/ch22_embodied_ai_robotics.md --builder pdfhtml
jupyter-book build book/part6/ch23_lifelong_learning.md --builder pdfhtml
jupyter-book build book/part6/ch24_quantum_computing_neuroai.md --builder pdfhtml

# Appendices
echo "Exporting Appendices..."
jupyter-book build book/appendices/math_python_refresher.md --builder pdfhtml
jupyter-book build book/appendices/dataset_catalogue.md --builder pdfhtml
jupyter-book build book/appendices/colab_setup.md --builder pdfhtml

# Move all PDFs to the pdf_exports directory
echo "Moving PDFs to pdf_exports directory..."
find book/_build/_page -name "*.pdf" -exec cp {} pdf_exports/ \;

# Build the full book with cover
echo "Building full book PDF with cover..."
jupyter-book build book --builder pdflatex

# Copy the full book PDF to the pdf_exports directory
echo "Moving full book PDF to pdf_exports directory..."
cp book/_build/latex/book.pdf pdf_exports/neuroai_handbook_full.pdf

echo "PDF export complete. All PDFs are in the pdf_exports directory."