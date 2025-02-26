from docling.document_converter import DocumentConverter

def pdf_to_markdown(file_path):
    """
    Extracts text from a PDF file and returns it as Markdown using Docling.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        md_text = result.document.export_to_markdown()
        return md_text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""
