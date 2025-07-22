#!/usr/bin/env python3
"""
Script to clean up the scraped text files in docs_vbt_clean.
"""

import re
from pathlib import Path
from tqdm import tqdm

DOCS_DIR = Path("docs_vbt_clean")

def clean_text_file(filepath):
    """
    Cleans a single text file by removing unwanted characters and patterns.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace \r\n and \r with \n for consistent newline handling
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # Remove '¶' and any surrounding whitespace
    content = re.sub(r'\s*¶\s*', '', content)

    # Remove redundant labels like 'module', 'class', 'method', 'property' when they appear as standalone lines
    # or are followed by a newline and then more text, but not part of a code example.
    # This pattern looks for the word at the start of a line (or preceded by whitespace) followed by a newline
    # and then another newline or more text, making it a standalone descriptor.
    content = re.sub(r'(^|\n)\s*(module|class|method|property)\s*\n', '\n', content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove patterns like '.<locals>.'
    content = re.sub(r'\.\<locals\>\.', '', content)

    # Remove Python console prompts (lines starting with '>>>')
    lines = content.splitlines()
    cleaned_lines_initial = []
    for line in lines:
        if not line.strip().startswith('>>>'):
            cleaned_lines_initial.append(line)
    content = "\n".join(cleaned_lines_initial)

    # Inlining: Split into paragraphs by multiple newlines, then join lines within paragraphs
    # First, normalize multiple newlines to just two newlines for consistent paragraph separation
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Split the content into blocks based on two or more newlines (paragraph breaks)
    blocks = re.split(r'\n\n+', content)
    inlined_blocks = []
    for block in blocks:
        # Replace single newlines within a block with a space
        inlined_block = re.sub(r'\s*\n\s*', ' ', block).strip()
        # Replace multiple spaces with a single space
        inlined_block = re.sub(r'\s+', ' ', inlined_block)
        if inlined_block:
            inlined_blocks.append(inlined_block)
    
    # Join the inlined blocks with two newlines to preserve paragraph structure
    content = "\n\n".join(inlined_blocks)

    # Final trim of leading/trailing whitespace for the entire content
    content = content.strip()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    if not DOCS_DIR.exists():
        print(f"Error: Documentation directory not found at {DOCS_DIR}")
        return

    text_files = list(DOCS_DIR.glob("*.txt"))
    if not text_files:
        print(f"No .txt files found in {DOCS_DIR}")
        return

    print(f"Cleaning {len(text_files)} text files in {DOCS_DIR}...")
    for filepath in tqdm(text_files, desc="Cleaning files"):
        clean_text_file(filepath)
    print("Cleaning complete!")

if __name__ == "__main__":
    main() 