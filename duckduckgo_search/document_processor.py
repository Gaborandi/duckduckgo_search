# document_processor.py

import os
import PyPDF2
import docx
import csv

def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext == '.doc':
        return extract_text_from_doc(path)
    elif ext == '.csv':
        return extract_text_from_csv(path)
    else:
        return read_any_text_file(path)

def extract_text_from_pdf(path):
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return '\n'.join(paragraphs)

def extract_text_from_doc(path):
    return ''

def extract_text_from_csv(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            rows.append(','.join(row))
    return '\n'.join(rows)

def read_any_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
