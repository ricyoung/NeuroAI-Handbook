#!/usr/bin/env python
import os
import yaml
from PyPDF2 import PdfWriter, PdfReader
import glob

def combine_chapters_in_toc_order():
    """Combine PDFs in the exact order specified in _toc.yml."""
    # Paths
    toc_yml_path = "book/_toc.yml"
    cover_pdf = "pdf_exports/cover.pdf"
    output_pdf = "pdf_exports/with_covers/neuroai_handbook_complete_ordered.pdf"
    
    # Load TOC structure
    with open(toc_yml_path, 'r') as file:
        toc_data = yaml.safe_load(file)
    
    # Create ordered list of file paths
    ordered_files = []
    for part in toc_data.get('parts', []):
        for chapter in part.get('chapters', []):
            # Skip the cover
            if chapter.get('file') == 'cover':
                continue
                
            # Get file base name
            file_path = chapter.get('file', '')
            if file_path:
                # Get base name from path
                base_name = os.path.basename(file_path)
                pdf_path = f"pdf_exports/{base_name}.pdf"
                
                # Only add if the PDF exists
                if os.path.exists(pdf_path):
                    ordered_files.append(pdf_path)
                else:
                    print(f"Warning: PDF not found: {pdf_path}")
            
            # Add sections if any
            if 'sections' in chapter:
                for section in chapter.get('sections', []):
                    section_path = section.get('file', '')
                    if section_path:
                        # Get base name
                        base_name = os.path.basename(section_path)
                        pdf_path = f"pdf_exports/{base_name}.pdf"
                        
                        # Only add if the PDF exists
                        if os.path.exists(pdf_path):
                            ordered_files.append(pdf_path)
                        else:
                            print(f"Warning: PDF not found: {pdf_path}")
    
    # Print order
    print("PDFs in order:")
    for i, path in enumerate(ordered_files):
        print(f"  {i+1}. {path}")
    
    # Create merged PDF
    merger = PdfWriter()
    
    # First add the cover
    if os.path.exists(cover_pdf):
        merger.append(cover_pdf)
        print(f"Added cover: {cover_pdf}")
    else:
        print(f"Warning: Cover PDF not found: {cover_pdf}")
    
    # Add all chapter PDFs in order
    for pdf_path in ordered_files:
        try:
            merger.append(pdf_path)
            print(f"Added: {pdf_path}")
        except Exception as e:
            print(f"Error adding {pdf_path}: {e}")
    
    # Write to output
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, "wb") as f:
        merger.write(f)
    
    print(f"\nCreated combined PDF at: {output_pdf}")
    print(f"Total PDFs combined: {len(ordered_files) + 1}")

if __name__ == "__main__":
    combine_chapters_in_toc_order()