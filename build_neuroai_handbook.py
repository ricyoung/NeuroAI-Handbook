#!/usr/bin/env python3
"""
build_neuroai_handbook.py - Complete build script for the NeuroAI Handbook

This script:
1. Builds all individual chapter PDFs using jupyter-book
2. Creates supporting pages (cover, TOC, dividers, etc.)
3. Combines everything into a single, well-formatted PDF handbook

Usage:
  python build_neuroai_handbook.py [--output OUTPUT_DIR]
  
Options:
  --output OUTPUT_DIR  Path to output directory for final PDF
  --help               Show this help message
"""

import os
import sys
import time
import argparse
import subprocess

def print_status(message, is_header=False):
    """Print a formatted status message."""
    if is_header:
        sep = "=" * 80
        print(f"\n{sep}")
        print(f" {message} ".center(78))
        print(f"{sep}\n")
    else:
        print(f"[INFO] {message}")

def run_command(command, description):
    """Run a shell command and print status."""
    print_status(description, True)
    print(f"Running command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print_status(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"ERROR: {description} failed with error code {e.returncode}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build the complete NeuroAI Handbook")
    parser.add_argument("--output", help="Path to output directory for final PDF")
    return parser.parse_args()

def main():
    """Run the complete build process."""
    args = parse_arguments()
    
    script_dir = os.path.abspath(os.path.dirname(__file__))
    start_time = time.time()
    
    # Build command arguments
    output_arg = f"--output {args.output}" if args.output else ""
    
    # Step 1: Build all chapter PDFs
    if not run_command(f"cd {script_dir} && python3 01_build_chapters.py",
                      "Step 1: Building chapter PDFs"):
        return False
    
    # Step 2: Create supporting pages (now includes frontmatter export)
    if not run_command(f"cd {script_dir} && python3 02_build_supporting.py",
                      "Step 2: Creating supporting pages and exporting frontmatter"):
        return False
    
    # Step 3: Merge the final handbook
    cmd = f"cd {script_dir} && python3 03_merge_final.py {output_arg}"
    if not run_command(cmd, "Step 3: Creating final handbook PDF"):
        return False
    
    # Print summary
    elapsed_time = time.time() - start_time
    print_status("BUILD SUMMARY", True)
    print(f"Total build time: {elapsed_time:.1f} seconds")
    print(f"Output directory: {args.output if args.output else 'pdf_exports/complete_handbook'}")
    print(f"Final PDF: neuroai_handbook.pdf")
    
    return True

if __name__ == "__main__":
    print_status("NeuroAI Handbook - Complete Build Process", True)
    success = main()
    sys.exit(0 if success else 1)