import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PIL import Image
import io
import sys

def create_cover_pdf(output_path):
    """Create a cover page PDF using ReportLab."""
    # Create a directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Background color (light gray)
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(0, 0, width, height, fill=True)
    
    # Title at the top
    c.setFillColorRGB(0.1, 0.1, 0.6)  # Dark blue for title
    c.setFont("Helvetica-Bold", 40)
    c.drawCentredString(width/2, height-2.5*inch, "The Neuroscience of AI")
    
    # Subtitle
    c.setFillColorRGB(0.3, 0.3, 0.3)  # Dark gray for subtitle
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-3.2*inch, "Bridging Brains and Machines")
    
    # Try to add the logo in the center
    try:
        logo_path = "book/nai.png"
        # Fallback to transparent if regular not found
        if not os.path.exists(logo_path):
            logo_path = "book/nai_transparent.png"
        
        if os.path.exists(logo_path):
            img = Image.open(logo_path)
            img_width, img_height = img.size
            
            # Scale the image to fit the page width (70%)
            scale = 0.7 * width / img_width
            img_width = img_width * scale
            img_height = img_height * scale
            
            # Center the image
            x = (width - img_width) / 2
            y = height/2 - img_height/2  # Center vertically
            
            c.drawImage(logo_path, x, y, width=img_width, height=img_height)
    except Exception as e:
        print(f"Error adding logo: {e}")
    
    # Author at the bottom
    c.setFillColorRGB(0.1, 0.1, 0.6)  # Dark blue for author
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, 2*inch, "Richard Young")
    
    # Add year
    c.setFillColorRGB(0.5, 0.5, 0.5)  # Gray for year
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, 1.5*inch, "2025")
    
    c.save()
    print(f"Cover page created at {output_path}")

if __name__ == "__main__":
    output_path = "pdf_exports/cover.pdf"
    create_cover_pdf(output_path)