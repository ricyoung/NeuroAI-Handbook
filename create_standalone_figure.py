#!/usr/bin/env python3
"""
Create a standalone PDF with just the neuron vs perceptron figure.
"""

import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import cairosvg

def create_figure_pdf():
    # Ensure output directory exists
    os.makedirs("pdf_exports", exist_ok=True)
    
    # Copy neuron_vs_perceptron.svg to ch01 directory if it doesn't exist
    src_svg = "/Volumes/data_bolt/_github/NeuroAI-Handbook/book/figures/neuron_vs_perceptron.svg"
    dst_dir = "/Volumes/data_bolt/_github/NeuroAI-Handbook/book/figures/ch01"
    dst_svg = os.path.join(dst_dir, "neuron_vs_perceptron_large.svg")
    
    if not os.path.exists(dst_svg):
        print(f"Copying {src_svg} to {dst_svg}")
        shutil.copy(src_svg, dst_svg)
    
    # Convert SVG to PNG for PDF inclusion
    png_path = "/Volumes/data_bolt/_github/NeuroAI-Handbook/pdf_exports/neuron_vs_perceptron.png"
    try:
        cairosvg.svg2png(url=src_svg, write_to=png_path, scale=2.0)
        print(f"Converted SVG to PNG: {png_path}")
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        # Create a simple text file instead
        with open(png_path.replace(".png", ".txt"), "w") as f:
            f.write("Neuron vs perceptron figure\n")
        png_path = png_path.replace(".png", ".txt")
    
    # Create PDF
    pdf_path = "/Volumes/data_bolt/_github/NeuroAI-Handbook/pdf_exports/neuron_vs_perceptron.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create document with the figure
    elements = []
    
    # Add title
    title_style = styles['Title']
    elements.append(Paragraph("Neuron vs. Perceptron Comparison", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Add description
    elements.append(Paragraph("Figure 1.3: Comparison between a biological neuron and an artificial perceptron, showing structural and functional parallels.", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Try to add the image if it's a PNG
    if png_path.endswith(".png"):
        try:
            img = Image(png_path, width=6*inch, height=3*inch)
            elements.append(img)
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
            elements.append(Paragraph(f"Error displaying image: {e}", styles['Normal']))
    else:
        # Add a placeholder text
        elements.append(Paragraph("Figure could not be rendered as an image.", styles['Normal']))
    
    # Build the PDF
    try:
        doc.build(elements)
        print(f"Created PDF: {pdf_path}")
    except Exception as e:
        print(f"Error creating PDF: {e}")

if __name__ == "__main__":
    create_figure_pdf()