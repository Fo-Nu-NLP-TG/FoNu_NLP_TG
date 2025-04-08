#!/usr/bin/env python3
"""
Script to convert Markdown blog posts to HTML for GitHub Pages and prepare for Medium.
Requires the markdown package: pip install markdown
"""

import os
import sys
import markdown
import re
import shutil
from datetime import datetime

def convert_file(input_file, output_file=None, for_medium=False):
    """Convert a Markdown file to HTML using the template."""
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.html'

    # Read the Markdown file
    with open(input_file, 'r') as f:
        md_content = f.read()

    # Extract the title (first heading)
    title_match = re.search(r'^# (.+)$', md_content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Blog Post"

    # Extract the date if present (format: *Posted on: [date]*)
    date_match = re.search(r'\*Posted on: (.+)\*', md_content)
    date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

    if for_medium:
        # Prepare content for Medium (just return the markdown with some modifications)
        # Replace local image paths with GitHub URLs
        repo_url = "https://github.com/Lemniscate-world/FoNu_NLP_TG/raw/main/"
        md_content = re.sub(r'\(\.\./(.*?)\)', f'({repo_url}\1)', md_content)
        md_content = re.sub(r'\(\./images/(.*?)\)', f'({repo_url}blog/images/\1)', md_content)

        # Add a footer with link back to GitHub
        md_content += "\n\n---\n\n*This post was originally published on our [GitHub project page](https://github.com/Lemniscate-world/FoNu_NLP_TG/). FoNu_NLP_TG (\"Fo Nu\" means \"speak\" in Ewe, and TG stands for Togo) is a research project focused on transformer models and Ewe-English translation.*"

        # Write the Medium-ready markdown
        medium_file = os.path.splitext(input_file)[0] + '_medium.md'
        with open(medium_file, 'w') as f:
            f.write(md_content)
        print(f"Prepared {input_file} for Medium as {medium_file}")
        return

    # Convert Markdown to HTML for GitHub Pages
    html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])

    # Read the template
    with open('template.html', 'r') as f:
        template = f.read()

    # Replace placeholders in the template
    output_html = template.replace('{{title}}', title)\
                          .replace('{{date}}', date)\
                          .replace('{{content}}', html_content)

    # Write the output HTML file
    with open(output_file, 'w') as f:
        f.write(output_html)

    print(f"Converted {input_file} to {output_file}")

def setup_github_pages():
    """Setup files for GitHub Pages."""
    # Create docs directory if it doesn't exist
    if not os.path.exists('../docs'):
        os.makedirs('../docs')
        print("Created docs directory for GitHub Pages")

    # Copy CSS to docs
    shutil.copy('style.css', '../docs/style.css')

    # Convert all markdown files and place in docs
    for filename in os.listdir('.'):
        if filename.endswith('.md') and filename != 'README.md':
            output_file = '../docs/' + os.path.splitext(filename)[0] + '.html'
            convert_file(filename, output_file)

    # Copy images directory
    if os.path.exists('images'):
        if not os.path.exists('../docs/images'):
            os.makedirs('../docs/images')
        for image in os.listdir('images'):
            shutil.copy(f'images/{image}', f'../docs/images/{image}')

    # Create index.html in the root of docs if it doesn't exist
    if not os.path.exists('../docs/index.html') and os.path.exists('index.md'):
        convert_file('index.md', '../docs/index.html')

    print("GitHub Pages setup complete. Files are in the 'docs' directory.")

def prepare_for_medium():
    """Prepare all blog posts for Medium."""
    # Create medium directory if it doesn't exist
    if not os.path.exists('medium'):
        os.makedirs('medium')
        print("Created medium directory for Medium-ready posts")

    # Convert all markdown files for Medium
    for filename in os.listdir('.'):
        if filename.endswith('.md') and filename != 'README.md' and not filename.endswith('_medium.md'):
            convert_file(filename, for_medium=True)

    print("Medium preparation complete. Files are ready for copy-pasting to Medium.")

def convert_all():
    """Convert all Markdown files in the current directory to HTML."""
    for filename in os.listdir('.'):
        if filename.endswith('.md') and filename != 'README.md' and not filename.endswith('_medium.md'):
            convert_file(filename)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            convert_all()
        elif sys.argv[1] == '--github-pages':
            setup_github_pages()
        elif sys.argv[1] == '--medium':
            prepare_for_medium()
        else:
            convert_file(sys.argv[1])
    else:
        print("Usage: python convert.py [file.md | --all | --github-pages | --medium]")
        print("  file.md: Convert a specific Markdown file to HTML")
        print("  --all: Convert all Markdown files in the current directory to HTML")
        print("  --github-pages: Setup files for GitHub Pages in the docs directory")
        print("  --medium: Prepare all blog posts for Medium")
