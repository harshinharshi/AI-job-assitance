"""
Data Loading and Document Processing Module

This module provides utilities for loading CV/resume files and generating
Word documents. It handles PDF extraction and DOCX generation for the
AI Job Assistance application.

Dependencies:
    - langchain_community.document_loaders: For PDF processing
    - python-docx: For Word document generation
"""

from langchain_community.document_loaders import PyPDFLoader
from docx import Document


def load_cv(file_path: str) -> str:
    """
    Load and extract text content from a PDF CV/resume file.
    
    This function uses LangChain's PyPDFLoader to read a PDF file and
    concatenate all pages into a single text string for processing.
    
    Args:
        file_path (str): Path to the PDF file to be loaded
        
    Returns:
        str: Concatenated text content from all pages of the PDF
        
    Example:
        >>> cv_text = load_cv("tmp/cv.pdf")
        >>> print(len(cv_text))
        1234
    """
    try:
        # Initialize PDF loader with the specified file path
        loader = PyPDFLoader(file_path)
        
        # Load all pages from the PDF
        pages = loader.load()
        
        # Concatenate content from all pages
        page_content = ''
        for i in range(len(pages)):
            page_content += pages[i].page_content
            
        return page_content
        
    except Exception as e:
        print(f"Error loading CV from {file_path}: {str(e)}")
        return ""


def write_to_docx(text: str, filename: str = 'tmp/cover_letter.docx') -> str:
    """
    Write text content to a Word document (.docx) file.
    
    This function takes plain text and creates a formatted Word document,
    splitting the text by newlines and creating separate paragraphs.
    
    Args:
        text (str): The text content to write to the document
        filename (str, optional): Output file path. Defaults to 'tmp/cover_letter.docx'
        
    Returns:
        str: The filename/path of the created document
        
    Example:
        >>> cover_letter = "Dear Hiring Manager,\\n\\nI am writing to apply..."
        >>> file_path = write_to_docx(cover_letter)
        >>> print(file_path)
        tmp/cover_letter.docx
    """
    try:
        print("Creating Word document...")
        print(f"Text content length: {len(text)} characters")
        
        # Create a new Word document
        doc = Document()
        
        # Split text into paragraphs by newlines
        paragraphs = text.split('\n')
        
        # Add each paragraph to the document
        for paragraph_text in paragraphs:
            # Skip empty paragraphs to avoid unnecessary whitespace
            if paragraph_text.strip():
                doc.add_paragraph(paragraph_text)
        
        # Save the document to the specified file path
        doc.save(filename)
        
        print(f"Document successfully saved as {filename}")
        return filename
        
    except Exception as e:
        print(f"Error creating Word document: {str(e)}")
        return ""


# Additional utility functions for future extensions

def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if a file exists and has a PDF extension.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        bool: True if file exists and is a PDF, False otherwise
    """
    import os
    
    if not os.path.exists(file_path):
        return False
    
    if not file_path.lower().endswith('.pdf'):
        return False
    
    return True


def get_document_metadata(file_path: str) -> dict:
    """
    Extract metadata from a PDF document.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary containing document metadata
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        if pages:
            return {
                'total_pages': len(pages),
                'first_page_chars': len(pages[0].page_content),
                'total_characters': sum(len(page.page_content) for page in pages)
            }
        
        return {'total_pages': 0, 'first_page_chars': 0, 'total_characters': 0}
        
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {'error': str(e)}