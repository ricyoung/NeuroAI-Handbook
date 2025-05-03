#!/usr/bin/env python3
"""
Add page breaks before Chapter Summary sections and Further Reading sections in all markdown files.
"""

import os
import re
from pathlib import Path
import sys

# Directory containing book content
BOOK_DIR = Path(__file__).parent / "book"
PAGE_BREAK_DIV = '<div style="page-break-before:always;"></div>\n\n'

def add_page_breaks_to_file(file_path):
    """Add page breaks before Chapter Summary and Further Reading sections in a file."""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Add page break before Chapter Summary
    original_content = content
    summary_pattern = r'(As.*?\n\n)```\{admonition\} Chapter Summary'
    content = re.sub(summary_pattern, r'\1' + PAGE_BREAK_DIV + '```{admonition} Chapter Summary', 
                    content, flags=re.DOTALL)
    
    # Also look for other patterns for Chapter Summary that might occur
    alt_summary_pattern = r'(\n\n)```\{admonition\} Chapter Summary'
    content = re.sub(alt_summary_pattern, r'\1' + PAGE_BREAK_DIV + '```{admonition} Chapter Summary', 
                    content, flags=re.DOTALL)
    
    # Add page break before Further Reading
    further_reading_pattern = r'(```\n\n)(## .*?Further Reading.*?)'
    content = re.sub(further_reading_pattern, r'\1' + PAGE_BREAK_DIV + r'\2', 
                   content, flags=re.DOTALL)
    
    # Also match if there's no admonition right before Further Reading
    alt_reading_pattern = r'(\n\n)(## .*?Further Reading.*?)'
    content = re.sub(alt_reading_pattern, r'\1' + PAGE_BREAK_DIV + r'\2', 
                   content, flags=re.DOTALL)
    
    # Check if content changed before writing
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Added page breaks to {file_path}")
        return True
    else:
        print(f"No changes needed for {file_path}")
        return False

def process_all_chapters():
    """Process all chapter files in the book directory."""
    changed_files = 0
    total_files = 0
    
    # Process all .md files in the book directory and subdirectories
    for root, _, files in os.walk(BOOK_DIR):
        for file in files:
            if file.endswith(".md") and "ch" in file and "_build" not in root:
                total_files += 1
                file_path = os.path.join(root, file)
                if add_page_breaks_to_file(file_path):
                    changed_files += 1
    
    print(f"Processed {total_files} files, updated {changed_files} files.")

if __name__ == "__main__":
    process_all_chapters()
    print("Page breaks added successfully!")