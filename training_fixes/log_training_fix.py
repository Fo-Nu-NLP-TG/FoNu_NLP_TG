#!/usr/bin/env python3
"""
Script to log training fixes to the README.md file in the training_fixes directory.
This helps maintain a record of all issues encountered and their solutions.
"""

import os
import sys
import datetime
import argparse

def add_fix_to_readme(issue, error_message, root_cause, fix, files_modified):
    """
    Add a new fix entry to the README.md file
    
    Args:
        issue (str): Brief description of the issue
        error_message (str): The error message encountered
        root_cause (str): Explanation of what caused the issue
        fix (str): Description of the fix applied
        files_modified (list): List of files that were modified
    """
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    
    # Create the file if it doesn't exist
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as f:
            f.write("# Training Fixes Documentation\n\n")
            f.write("This document tracks all fixes made to the transformer model training pipeline to address various issues encountered during development.\n\n")
    
    # Read the current content
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Format the new entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = f"## {issue} ({timestamp})\n\n"
    new_entry += f"**Issue:** {issue}\n\n"
    
    if error_message:
        new_entry += "**Error Message:**\n```\n"
        new_entry += error_message + "\n"
        new_entry += "```\n\n"
    
    new_entry += f"**Root Cause:**\n{root_cause}\n\n"
    new_entry += f"**Fix:**\n{fix}\n\n"
    
    if files_modified:
        new_entry += "**Files Modified:**\n"
        for file in files_modified:
            new_entry += f"- `{file}`\n"
        new_entry += "\n"
    
    # Find the position to insert the new entry (after the header)
    header_end = content.find("\n\n") + 2
    updated_content = content[:header_end] + new_entry + content[header_end:]
    
    # Write the updated content
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Added fix entry to {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Log a training fix to the README.md file")
    parser.add_argument("--issue", required=True, help="Brief description of the issue")
    parser.add_argument("--error", help="The error message encountered")
    parser.add_argument("--cause", required=True, help="Explanation of what caused the issue")
    parser.add_argument("--fix", required=True, help="Description of the fix applied")
    parser.add_argument("--files", nargs="+", help="List of files that were modified")
    
    args = parser.parse_args()
    
    add_fix_to_readme(args.issue, args.error, args.cause, args.fix, args.files)

if __name__ == "__main__":
    main()