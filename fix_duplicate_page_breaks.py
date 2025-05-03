#!/usr/bin/env python3
"""
Fix duplicate page break divs in all chapter files.
This script will find and remove duplicate page break divs that cause extra blank pages.
"""

import os
import re
from pathlib import Path
import sys

# Directory containing book content
BOOK_DIR = Path(__file__).parent / "book"

def fix_duplicate_page_breaks(file_path):
    """Remove duplicate page break divs in a file."""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Original content for comparison
    original_content = content
    
    # Define the page break div pattern
    page_break_pattern = r'<div style="page-break-before:always;"></div>\s*'
    
    # Find instances of multiple consecutive page breaks
    multiple_breaks_pattern = f'({page_break_pattern}){{2,}}'
    
    # Replace multiple consecutive page breaks with a single one
    content = re.sub(multiple_breaks_pattern, r'<div style="page-break-before:always;"></div>\n\n', content)
    
    # Check if content changed before writing
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Fixed duplicate page breaks in {file_path}")
        return True
    else:
        return False

def process_all_chapters():
    """Process all chapter files in the book directory."""
    fixed_files = 0
    total_files = 0
    
    # Process all .md files in the book directory and subdirectories
    for root, _, files in os.walk(BOOK_DIR):
        for file in files:
            if file.endswith(".md") and "ch" in file and "_build" not in root:
                total_files += 1
                file_path = os.path.join(root, file)
                if fix_duplicate_page_breaks(file_path):
                    fixed_files += 1
    
    print(f"Processed {total_files} files, fixed {fixed_files} files with duplicate page breaks.")

if __name__ == "__main__":
    print("Fixing duplicate page breaks in chapter files...")
    process_all_chapters()
    print("Done!")