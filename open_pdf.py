#!/usr/bin/env python3
"""
open_pdf.py - Quickly open the NeuroAI handbook PDF

This script opens the generated PDF handbook in the default PDF viewer.
"""

import os
import sys
import platform
import subprocess
import argparse

def print_status(message):
    """Print a formatted status message."""
    print(f"[INFO] {message}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Open the NeuroAI handbook PDF")
    parser.add_argument("--output", help="Path to output directory (default: pdf_exports/complete_handbook)")
    return parser.parse_args()

def open_pdf(pdf_path):
    """Open the PDF using the system's default PDF viewer."""
    print_status(f"Opening PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print_status(f"ERROR: PDF not found at {pdf_path}")
        print_status("Run 'python build_neuroai_handbook.py' to build the PDF first")
        return False
    
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", pdf_path])
        elif system == "Windows":
            os.startfile(pdf_path)
        else:  # Linux or other
            subprocess.run(["xdg-open", pdf_path])
        print_status(f"PDF opened successfully")
        return True
    except Exception as e:
        print_status(f"ERROR: Failed to open PDF: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Determine the PDF path
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = args.output if args.output else os.path.join(base_dir, "pdf_exports/complete_handbook")
    pdf_path = os.path.join(output_dir, "neuroai_handbook.pdf")
    
    return open_pdf(pdf_path)

if __name__ == "__main__":
    print_status("NeuroAI Handbook - PDF Viewer")
    success = main()
    sys.exit(0 if success else 1)