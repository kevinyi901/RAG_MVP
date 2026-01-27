"""Document loading with support for PDF, DOCX, TXT, and OCR."""

import os
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# PDF processing
from pypdf import PdfReader

# DOCX processing
from docx import Document as DocxDocument

# OCR support
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image


@dataclass
class LoadedDocument:
    """Represents a loaded document with extracted text."""
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int]
    content: str
    pages: List[Dict[str, Any]]  # List of {page_number, content, section_title}


class DocumentLoader:
    """Load and extract text from various document formats."""

    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.doc'}

    def __init__(self, ocr_enabled: bool = True, ocr_language: str = 'eng'):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language

    def load(self, file_path: str) -> LoadedDocument:
        """Load a document from file path."""
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        file_size = path.stat().st_size

        with open(file_path, 'rb') as f:
            content = f.read()

        return self.load_bytes(content, path.name, extension, file_size)

    def load_bytes(
        self,
        content: bytes,
        filename: str,
        file_type: str = None,
        file_size: int = None
    ) -> LoadedDocument:
        """Load a document from bytes."""
        if file_type is None:
            file_type = Path(filename).suffix.lower()

        if file_size is None:
            file_size = len(content)

        if file_type == '.pdf':
            return self._load_pdf(content, filename, file_size)
        elif file_type in ('.txt', '.md'):
            return self._load_text(content, filename, file_type, file_size)
        elif file_type in ('.docx', '.doc'):
            return self._load_docx(content, filename, file_size)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _load_pdf(self, content: bytes, filename: str, file_size: int) -> LoadedDocument:
        """Load a PDF document with optional OCR."""
        reader = PdfReader(io.BytesIO(content))
        pages = []
        full_text = []

        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""

            # If no text extracted and OCR is enabled, try OCR
            if not page_text.strip() and self.ocr_enabled:
                page_text = self._ocr_pdf_page(content, page_num - 1)

            # Try to extract section title from first line
            section_title = self._extract_section_title(page_text)

            pages.append({
                "page_number": page_num,
                "content": page_text,
                "section_title": section_title
            })
            full_text.append(page_text)

        return LoadedDocument(
            filename=filename,
            file_type="pdf",
            file_size=file_size,
            page_count=len(reader.pages),
            content="\n\n".join(full_text),
            pages=pages
        )

    def _ocr_pdf_page(self, pdf_content: bytes, page_index: int) -> str:
        """OCR a single PDF page."""
        try:
            images = convert_from_bytes(
                pdf_content,
                first_page=page_index + 1,
                last_page=page_index + 1,
                dpi=200
            )
            if images:
                return pytesseract.image_to_string(
                    images[0],
                    lang=self.ocr_language
                )
        except Exception as e:
            print(f"OCR failed for page {page_index + 1}: {e}")
        return ""

    def _load_text(
        self,
        content: bytes,
        filename: str,
        file_type: str,
        file_size: int
    ) -> LoadedDocument:
        """Load a plain text or markdown file."""
        text = content.decode('utf-8', errors='replace')

        return LoadedDocument(
            filename=filename,
            file_type=file_type.lstrip('.'),
            file_size=file_size,
            page_count=None,
            content=text,
            pages=[{
                "page_number": None,
                "content": text,
                "section_title": None
            }]
        )

    def _load_docx(self, content: bytes, filename: str, file_size: int) -> LoadedDocument:
        """Load a DOCX document."""
        doc = DocxDocument(io.BytesIO(content))
        paragraphs = []
        current_section = None
        pages = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Check if this is a heading
            if para.style and para.style.name.startswith('Heading'):
                current_section = text

            paragraphs.append(text)

        full_text = "\n\n".join(paragraphs)

        # DOCX doesn't have real page numbers, treat as single page
        pages.append({
            "page_number": None,
            "content": full_text,
            "section_title": current_section
        })

        return LoadedDocument(
            filename=filename,
            file_type="docx",
            file_size=file_size,
            page_count=None,
            content=full_text,
            pages=pages
        )

    def _extract_section_title(self, text: str, max_length: int = 100) -> Optional[str]:
        """Extract a potential section title from text."""
        if not text:
            return None

        lines = text.strip().split('\n')
        if not lines:
            return None

        first_line = lines[0].strip()

        # Heuristics for section titles:
        # - Short lines (under max_length chars)
        # - All caps or title case
        # - Doesn't end with common sentence endings
        if len(first_line) > max_length:
            return None

        if first_line.endswith(('.', ',', ';', ':')):
            return None

        if first_line.isupper() or first_line.istitle():
            return first_line

        return None

    def ocr_image(self, image_path: str) -> str:
        """OCR a standalone image file."""
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang=self.ocr_language)
