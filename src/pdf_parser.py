"""
PDF Parser Module

This module handles the extraction of text from PDF resumes.
It uses pdfplumber and PyPDF2 as fallback options to ensure reliable text extraction.
"""

import os
import logging
import pdfplumber
import PyPDF2
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFParser:
    """
    A class to extract text from PDF files, specifically designed for resumes.
    """
    
    def __init__(self):
        """Initialize the PDF parser."""
        logger.info("Initializing PDF Parser")
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text from a PDF using pdfplumber.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    text += "\n\n"  # Add spacing between pages
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return ""
    
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text from a PDF using PyPDF2 (fallback method).
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() or ""
                    text += "\n\n"  # Add spacing between pages
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF using multiple methods for reliability.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return ""
        
        # Try pdfplumber first (better formatting)
        text = self.extract_text_with_pdfplumber(pdf_path)
        
        # If pdfplumber fails or returns empty text, try PyPDF2
        if not text:
            logger.info("pdfplumber extraction failed, trying PyPDF2")
            text = self.extract_text_with_pypdf2(pdf_path)
        
        if not text:
            logger.error("All PDF extraction methods failed")
            return ""
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing metadata
        """
        metadata = {}
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                # Add page count
                metadata['/PageCount'] = len(reader.pages)
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return metadata


# Example usage
if __name__ == "__main__":
    parser = PDFParser()
    sample_pdf = "../data/sample_resume.pdf"
    
    if os.path.exists(sample_pdf):
        text = parser.extract_text(sample_pdf)
        print(f"Extracted {len(text)} characters")
        print("First 500 characters:")
        print(text[:500])
    else:
        print(f"Sample PDF not found at {sample_pdf}") 