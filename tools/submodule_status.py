#!/usr/bin/env python3
"""
Submodule Status Reporter for FoNu_NLP_TG

This script generates a report on the status of all submodules in the repository,
which can be included in weekly updates.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import datetime

def get_submodules():
    """Get a list of all submodules in the repository."""
    try:
        result = subprocess.check_output(
            "git submodule status",
            shell=True
        ).decode('utf-8').strip()
        
        submodules = []
        for line in result.split('\n'):
            if not line.strip():
                continue
            
            # Parse the submodule status line
            # Format: [+]<sha1> <path> [(<branch>)]
            parts = line.strip().split()
            if len(parts) >= 2:
                status_and_sha = parts[0]
                path = parts[1]
                
                # Check if there's a status indicator
                status = "current"
                if status_and_sha[0] in ['+', '-', 'U']:
                    status = {
                        '+': "needs update",
                        '-': "not initialized",
                        'U': "merge conflicts"
                    }.get(status_and_sha[0], "unknown")
                    sha = status_and_sha[1:]
                else:
                    sha = status_and_sha
                
                submodules.append({
                    "path": path,
                    "status": status,
                    "sha": sha
                })
        
        return submodules
    except subprocess.CalledProcessError:
        print("Error: Failed to get submodule status. Make sure you're in a git repository.")
        return []

def get_submodule_changes(submodule_path, days=7):
    """Get changes for a specific submodule in the last week."""
    try:
        # Save current directory
        original_dir = os.getcwd()
        
        # Change to submodule directory
        os.chdir(submodule_path)
        
        # Get commit count
        commit_count = subprocess.check_output(
            f"git log --since='{days} days ago' --oneline | wc -l",
            shell=True
        ).decode('utf-8').strip()
        
        # Get files changed
        files_changed = subprocess.check_output(
            f"git log --since='{days} days ago' --name-only --pretty=format: | sort | uniq | wc -l",
            shell=True
        ).decode('utf-8').strip()
        
        # Get last commit info
        last_commit_date = subprocess.check_output(
            "git log -1 --format=%cd --date=short",
            shell=True
        ).decode('utf-8').strip()
        
        # Return to original directory
        os.chdir(original_dir)
        
        return {
            "commit_count": commit_count,
            "files_changed": files_changed,
            "last_commit_date": last_commit_date
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Return to original directory in case of error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        
        return {
            "commit_count": "0",
            "files_changed": "0",
            "last_commit_date": "unknown"
        }

def generate_submodule_report(days=7):
    """Generate a report on all submodules."""
    submodules = get_submodules()
    
    if not submodules:
        return "No submodules found in this repository."
    
    report = "## Submodule Status\n\n"
    report += "| Submodule | Status | Last Commit | Changes (Last Week) |\n"
    report += "|-----------|--------|-------------|---------------------|\n"
    
    for submodule in submodules:
        changes = get_submodule_changes(submodule["path"], days)
        
        report += f"| {submodule['path']} | {submodule['status']} | {changes['last_commit_date']} | "
        report += f"{changes['commit_count']} commits, {changes['files_changed']} files |\n"
    
    return report

def save_report(report, output_file=None):
    """Save the submodule report to a file."""
    if not output_file:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        output_dir = Path("tools/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"submodule_report_{date_str}.md"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Saved submodule report to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate a report on submodule status")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back for changes")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()
    
    report = generate_submodule_report(args.days)
    output_file = save_report(report, args.output)
    
    print("\nSubmodule report generated successfully!")
    print(f"Report saved to {output_file}")
    print("\nYou can include this report in your weekly updates.")

if __name__ == "__main__":
    main()
