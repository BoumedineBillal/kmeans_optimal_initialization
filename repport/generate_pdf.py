import os
import sys
import subprocess

# Paths
REPORT_DIR = os.path.abspath(os.path.dirname(__file__))
HTML_PATH = os.path.join(REPORT_DIR, "rapport.html")
PDF_PATH = os.path.join(REPORT_DIR, "rapport_kmeans_initialisation_optimale.pdf")

def check_weasyprint_installed():
    """Check if WeasyPrint is installed"""
    try:
        import weasyprint
        return True
    except ImportError:
        print("WeasyPrint is not installed. Please install it manually:")
        print("pip install weasyprint")
        return False

def generate_pdf():
    """Generate PDF from HTML using WeasyPrint"""
    if not os.path.exists(HTML_PATH):
        print(f"HTML file not found: {HTML_PATH}")
        return False
    
    try:
        from weasyprint import HTML
        print(f"Generating PDF from {HTML_PATH}")
        
        # Set the base URL to the report directory for local file access
        html = HTML(HTML_PATH, base_url=REPORT_DIR)
        html.write_pdf(PDF_PATH)
        
        # Also create a copy with the current date in the filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dated_pdf_path = os.path.join(REPORT_DIR, f"rapport_kmeans_initialisation_optimale_{timestamp}.pdf")
        html.write_pdf(dated_pdf_path)
        
        print(f"PDF generated successfully: {PDF_PATH}")
        print(f"Dated copy created: {dated_pdf_path}")
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    if check_weasyprint_installed():
        generate_pdf()
    else:
        print("Aborting PDF generation.")
