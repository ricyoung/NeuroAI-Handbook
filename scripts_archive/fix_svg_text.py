#!/usr/bin/env python
import os
import re
from xml.etree import ElementTree as ET

def fix_svg_text(svg_file, output_file=None, scale_factor=1.5, font_size_min=12):
    """
    Improve text readability in SVG files by:
    1. Increasing font sizes
    2. Making text bold where appropriate
    3. Ensuring sufficient contrast
    
    Args:
        svg_file: Path to input SVG file
        output_file: Path to output SVG file (defaults to overwriting input)
        scale_factor: Factor by which to scale existing font sizes
        font_size_min: Minimum font size to enforce
    """
    if output_file is None:
        output_file = svg_file
    
    # Register SVG namespace
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    
    try:
        # Parse the SVG file
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Find all text elements
        text_elements = root.findall(".//{http://www.w3.org/2000/svg}text")
        print(f"Found {len(text_elements)} text elements in {svg_file}")
        
        # Process each text element
        for text_elem in text_elements:
            # Get the style attribute
            style = text_elem.get("style", "")
            
            # Extract font-size if present
            font_size_match = re.search(r'font-size:([0-9.]+)px', style)
            if font_size_match:
                current_size = float(font_size_match.group(1))
                new_size = max(current_size * scale_factor, font_size_min)
                
                # Replace font size in style
                new_style = re.sub(
                    r'font-size:[0-9.]+px', 
                    f'font-size:{new_size:.1f}px', 
                    style
                )
                
                # Add font-weight:bold if not present
                if 'font-weight:' not in new_style:
                    new_style += '; font-weight:bold'
                
                text_elem.set("style", new_style)
            
            # Process tspan elements within text elements
            for tspan in text_elem.findall(".//{http://www.w3.org/2000/svg}tspan"):
                tspan_style = tspan.get("style", "")
                
                # Extract font-size if present
                tspan_font_match = re.search(r'font-size:([0-9.]+)px', tspan_style)
                if tspan_font_match:
                    current_size = float(tspan_font_match.group(1))
                    new_size = max(current_size * scale_factor, font_size_min)
                    
                    # Replace font size in style
                    new_style = re.sub(
                        r'font-size:[0-9.]+px', 
                        f'font-size:{new_size:.1f}px', 
                        tspan_style
                    )
                    
                    # Add font-weight:bold if not present
                    if 'font-weight:' not in new_style:
                        new_style += '; font-weight:bold'
                    
                    tspan.set("style", new_style)
        
        # Find all SVG elements with width and height attributes
        if root.get("width") and root.get("height"):
            # Increase SVG dimensions if needed
            try:
                width = float(root.get("width").replace("pt", "").replace("px", ""))
                height = float(root.get("height").replace("pt", "").replace("px", ""))
                
                # Scale dimensions (optional)
                # root.set("width", f"{width * 1.2}")
                # root.set("height", f"{height * 1.2}")
                
                # Ensure viewBox is set correctly
                if not root.get("viewBox"):
                    root.set("viewBox", f"0 0 {width} {height}")
            except:
                # Skip if dimensions can't be parsed
                pass
        
        # Save the modified SVG
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"Modified SVG saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error processing SVG file: {e}")
        return False

if __name__ == "__main__":
    # Process timeline SVG
    timeline_svg = "/Volumes/data_bolt/_github/NeuroAI-Handbook/book/figures/ch01/neuro_ai_timeline.svg"
    fix_svg_text(timeline_svg, scale_factor=1.8, font_size_min=14)
    
    # Process history SVG
    history_svg = "/Volumes/data_bolt/_github/NeuroAI-Handbook/book/figures/ch01/neuro_ai_history.svg"
    fix_svg_text(history_svg, scale_factor=1.8, font_size_min=14)