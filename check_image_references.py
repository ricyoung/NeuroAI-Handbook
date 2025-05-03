#!/usr/bin/env python3
"""
Check for image references in markdown files and validate they point to existing files.
This script will find and report any image references that might be broken.
"""

import os
import re
from pathlib import Path
import sys

# Directory containing book content
BOOK_DIR = Path(__file__).parent / "book"

def check_markdown_file(file_path):
    """Check a markdown file for potentially broken image references."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    file_dir = os.path.dirname(file_path)
    issues = []
    
    # Look for markdown image syntax ![alt](path)
    md_image_pattern = r'!\[(.*?)\]\((.*?)\)'
    for match in re.finditer(md_image_pattern, content):
        alt_text, image_path = match.groups()
        # Skip external URLs
        if image_path.startswith(('http://', 'https://')):
            continue
        
        # Resolve the image path relative to the markdown file
        if image_path.startswith('/'):
            # Absolute path within project
            resolved_path = os.path.join(BOOK_DIR, image_path.lstrip('/'))
        else:
            # Relative path
            resolved_path = os.path.normpath(os.path.join(file_dir, image_path))
        
        if not os.path.exists(resolved_path):
            issues.append(f"Image not found: {image_path}")
    
    # Look for HTML image tags <img src="path">
    html_image_pattern = r'<img\s+[^>]*?src=["\']([^"\']+)["\'][^>]*?>'
    for match in re.finditer(html_image_pattern, content):
        image_path = match.group(1)
        # Skip external URLs
        if image_path.startswith(('http://', 'https://')):
            continue
        
        # Resolve the image path relative to the markdown file
        if image_path.startswith('/'):
            # Absolute path within project
            resolved_path = os.path.join(BOOK_DIR, image_path.lstrip('/'))
        else:
            # Relative path
            resolved_path = os.path.normpath(os.path.join(file_dir, image_path))
        
        if not os.path.exists(resolved_path):
            issues.append(f"Image not found: {image_path}")
    
    return issues

def process_all_chapters():
    """Process all chapter files in the book directory."""
    issue_count = 0
    
    # Process all .md files in the book directory, skipping _build directory
    for root, _, files in os.walk(BOOK_DIR):
        if "_build" in root:
            continue
            
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                issues = check_markdown_file(file_path)
                
                if issues:
                    print(f"\nIssues in {os.path.relpath(file_path, BOOK_DIR)}:")
                    for issue in issues:
                        print(f"  - {issue}")
                        issue_count += 1
    
    return issue_count

if __name__ == "__main__":
    print("Checking image references in markdown files...")
    issue_count = process_all_chapters()
    
    if issue_count > 0:
        print(f"\nFound {issue_count} potential issues with image references.")
    else:
        print("\nNo issues found with image references.")