#!/usr/bin/env python3
"""
Code Status Analyzer for FoNu_NLP_TG

This script analyzes the codebase to identify works in progress, TODOs, FIXMEs,
and other indicators of incomplete or buggy code.
"""

import os
import sys
import argparse
import re
from pathlib import Path
import json
from datetime import datetime

# Patterns to search for in code
TODO_PATTERN = re.compile(r'(?i)#\s*TODO\s*:?(.*?)($|\n)')
FIXME_PATTERN = re.compile(r'(?i)#\s*FIXME\s*:?(.*?)($|\n)')
BUG_PATTERN = re.compile(r'(?i)#\s*BUG\s*:?(.*?)($|\n)')
WIP_PATTERN = re.compile(r'(?i)#\s*WIP\s*:?(.*?)($|\n)')
HACK_PATTERN = re.compile(r'(?i)#\s*HACK\s*:?(.*?)($|\n)')

# File extensions to analyze
CODE_EXTENSIONS = ['.py', '.ipynb', '.js', '.html', '.css', '.md']

def find_code_files(root_dir, extensions=None):
    """Find all code files in the given directory."""
    if extensions is None:
        extensions = CODE_EXTENSIONS
    
    code_files = []
    for ext in extensions:
        code_files.extend(list(Path(root_dir).glob(f'**/*{ext}')))
    
    return code_files

def analyze_file(file_path):
    """Analyze a single file for TODOs, FIXMEs, etc."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        todos = [(m.group(1).strip(), m.start()) for m in TODO_PATTERN.finditer(content)]
        fixmes = [(m.group(1).strip(), m.start()) for m in FIXME_PATTERN.finditer(content)]
        bugs = [(m.group(1).strip(), m.start()) for m in BUG_PATTERN.finditer(content)]
        wips = [(m.group(1).strip(), m.start()) for m in WIP_PATTERN.finditer(content)]
        hacks = [(m.group(1).strip(), m.start()) for m in HACK_PATTERN.finditer(content)]
        
        # Get line numbers for each match
        line_count = 1
        line_positions = [0]
        for i, char in enumerate(content):
            if char == '\n':
                line_count += 1
                line_positions.append(i + 1)
        
        def get_line_number(pos):
            for i, line_pos in enumerate(line_positions):
                if pos < line_pos:
                    return i
            return line_count
        
        todos = [(text, get_line_number(pos)) for text, pos in todos]
        fixmes = [(text, get_line_number(pos)) for text, pos in fixmes]
        bugs = [(text, get_line_number(pos)) for text, pos in bugs]
        wips = [(text, get_line_number(pos)) for text, pos in wips]
        hacks = [(text, get_line_number(pos)) for text, pos in hacks]
        
        return {
            "todos": todos,
            "fixmes": fixmes,
            "bugs": bugs,
            "wips": wips,
            "hacks": hacks
        }
    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
        return {
            "todos": [],
            "fixmes": [],
            "bugs": [],
            "wips": [],
            "hacks": []
        }

def analyze_codebase(root_dir, extensions=None):
    """Analyze the entire codebase for TODOs, FIXMEs, etc."""
    code_files = find_code_files(root_dir, extensions)
    
    results = {}
    for file_path in code_files:
        rel_path = os.path.relpath(file_path, root_dir)
        file_results = analyze_file(file_path)
        
        # Only include files with issues
        if any(file_results.values()):
            results[rel_path] = file_results
    
    return results

def generate_report(results, output_format='markdown'):
    """Generate a report of the analysis results."""
    if output_format == 'json':
        return json.dumps(results, indent=2)
    
    # Markdown format
    report = "# Code Status Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Summary
    total_todos = sum(len(file_results['todos']) for file_results in results.values())
    total_fixmes = sum(len(file_results['fixmes']) for file_results in results.values())
    total_bugs = sum(len(file_results['bugs']) for file_results in results.values())
    total_wips = sum(len(file_results['wips']) for file_results in results.values())
    total_hacks = sum(len(file_results['hacks']) for file_results in results.values())
    
    report += "## Summary\n\n"
    report += f"- Total TODOs: {total_todos}\n"
    report += f"- Total FIXMEs: {total_fixmes}\n"
    report += f"- Total BUGs: {total_bugs}\n"
    report += f"- Total WIPs: {total_wips}\n"
    report += f"- Total HACKs: {total_hacks}\n\n"
    
    # Details
    report += "## Details\n\n"
    
    for file_path, file_results in sorted(results.items()):
        if not any(file_results.values()):
            continue
        
        report += f"### {file_path}\n\n"
        
        if file_results['todos']:
            report += "#### TODOs\n\n"
            for todo, line in file_results['todos']:
                report += f"- Line {line}: {todo}\n"
            report += "\n"
        
        if file_results['fixmes']:
            report += "#### FIXMEs\n\n"
            for fixme, line in file_results['fixmes']:
                report += f"- Line {line}: {fixme}\n"
            report += "\n"
        
        if file_results['bugs']:
            report += "#### BUGs\n\n"
            for bug, line in file_results['bugs']:
                report += f"- Line {line}: {bug}\n"
            report += "\n"
        
        if file_results['wips']:
            report += "#### WIPs\n\n"
            for wip, line in file_results['wips']:
                report += f"- Line {line}: {wip}\n"
            report += "\n"
        
        if file_results['hacks']:
            report += "#### HACKs\n\n"
            for hack, line in file_results['hacks']:
                report += f"- Line {line}: {hack}\n"
            report += "\n"
    
    return report

def save_report(report, output_file=None):
    """Save the report to a file."""
    if not output_file:
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = Path("tools/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"code_status_report_{date_str}.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved code status report to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Analyze codebase for TODOs, FIXMEs, etc.")
    parser.add_argument("--root-dir", default=".", help="Root directory to analyze")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=['markdown', 'json'], default='markdown', help="Output format")
    args = parser.parse_args()
    
    results = analyze_codebase(args.root_dir)
    report = generate_report(results, args.format)
    output_file = save_report(report, args.output)
    
    print("\nCode status analysis completed!")
    print(f"Report saved to {output_file}")
    print("\nYou can include this report in your weekly updates to track works in progress and known issues.")

if __name__ == "__main__":
    main()
