#!/usr/bin/env python3
"""
Weekly Update Automation Script for FoNu_NLP_TG

This script helps automate the process of creating and publishing weekly updates
for the FoNu_NLP_TG project across multiple platforms.
"""

import os
import sys
import argparse
import datetime
import subprocess
import shutil
from pathlib import Path

def get_week_number():
    """Get the current ISO week number."""
    return datetime.datetime.now().isocalendar()[1]

def get_date_range():
    """Get the date range for the current week (Sunday to Saturday)."""
    now = datetime.datetime.now()
    start_of_week = now - datetime.timedelta(days=now.weekday() + 1)  # Sunday
    end_of_week = start_of_week + datetime.timedelta(days=6)  # Saturday
    return f"{start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}"

def get_git_changes(days=7):
    """Get a detailed summary of git changes in the last week."""
    try:
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

        # Get commit messages for summary
        commit_messages = subprocess.check_output(
            f"git log --since='{days} days ago' --pretty=format:'%s'",
            shell=True
        ).decode('utf-8').strip().split('\n')

        # Get detailed changes with stats
        detailed_changes = subprocess.check_output(
            f"git log --since='{days} days ago' --stat",
            shell=True
        ).decode('utf-8').strip()

        # Get files by type
        file_types = {}
        try:
            all_files = subprocess.check_output(
                f"git log --since='{days} days ago' --name-only --pretty=format:",
                shell=True
            ).decode('utf-8').strip().split('\n')

            for file in all_files:
                if not file.strip():
                    continue
                ext = os.path.splitext(file)[1]
                if ext in file_types:
                    file_types[ext] += 1
                else:
                    file_types[ext] = 1
        except:
            file_types = {"unknown": 0}

        # Get current branches
        branches = subprocess.check_output(
            "git branch",
            shell=True
        ).decode('utf-8').strip().split('\n')

        # Get open issues if GitHub CLI is available
        open_issues = []
        try:
            issues_output = subprocess.check_output(
                "gh issue list --state open --limit 10",
                shell=True
            ).decode('utf-8').strip()
            if issues_output:
                open_issues = issues_output.split('\n')
        except:
            # GitHub CLI not available or not authenticated
            pass

        # Get submodule status
        submodules = []
        try:
            submodule_output = subprocess.check_output(
                "git submodule status",
                shell=True
            ).decode('utf-8').strip()
            if submodule_output:
                submodules = submodule_output.split('\n')
        except:
            # No submodules or error
            pass

        return {
            "commit_count": commit_count,
            "files_changed": files_changed,
            "commit_messages": commit_messages,
            "detailed_changes": detailed_changes,
            "file_types": file_types,
            "branches": branches,
            "open_issues": open_issues,
            "submodules": submodules
        }
    except subprocess.CalledProcessError:
        print("Error: Failed to get git changes. Make sure you're in a git repository.")
        return {
            "commit_count": "0",
            "files_changed": "0",
            "commit_messages": [],
            "detailed_changes": "",
            "file_types": {},
            "branches": [],
            "open_issues": [],
            "submodules": []
        }

def create_update_file(week_num, date_range, git_changes):
    """Create a new weekly update file from the template."""
    template_path = Path("tools/weekly_update_template.md")
    medium_template_path = Path("tools/medium_post_template.md")

    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        return None

    if not medium_template_path.exists():
        print(f"Warning: Medium template file not found at {medium_template_path}")

    # Create blog directory if it doesn't exist
    blog_dir = Path("blog/updates")
    blog_dir.mkdir(parents=True, exist_ok=True)

    # Create the update file
    update_filename = f"week_{week_num}_{datetime.datetime.now().strftime('%Y%m%d')}.md"
    update_path = blog_dir / update_filename

    # Read template and replace placeholders
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Replace basic placeholders
    content = template_content.replace("[Week Number]", str(week_num))
    content = content.replace("[Date Range]", date_range)

    # Write the file
    with open(update_path, 'w') as f:
        f.write(content)

    print(f"Created update file: {update_path}")

    # Create Medium blog post if template exists
    if medium_template_path.exists():
        medium_filename = f"medium_week_{week_num}_{datetime.datetime.now().strftime('%Y%m%d')}.md"
        medium_path = blog_dir / medium_filename

        # Read Medium template and replace placeholders
        with open(medium_template_path, 'r') as f:
            medium_template_content = f.read()

        # Replace basic placeholders
        medium_content = medium_template_content.replace("[X]", str(week_num))
        medium_content = medium_content.replace("[Title: Week X Update: Key Accomplishment or Focus]",
                                              f"Week {week_num} Update: [Main Focus or Accomplishment]")

        # Write the Medium blog post file
        with open(medium_path, 'w') as f:
            f.write(medium_content)

        print(f"Created Medium blog post template: {medium_path}")

    return update_path

def update_blog_index(update_path):
    """Update the blog index.md to include the new update."""
    index_path = Path("blog/index.md")

    if not index_path.exists():
        print(f"Warning: Blog index not found at {index_path}")
        return

    with open(index_path, 'r') as f:
        index_content = f.readlines()

    # Find the "Latest Posts" section
    latest_posts_line = -1
    for i, line in enumerate(index_content):
        if "## Latest Posts" in line:
            latest_posts_line = i
            break

    if latest_posts_line == -1:
        print("Warning: Could not find '## Latest Posts' section in blog index")
        return

    # Get the update filename without path
    update_filename = os.path.basename(update_path)

    # Create the new entry
    update_title = f"Weekly Update: Week {get_week_number()} - {get_date_range()}"
    new_entry = f"- [{update_title}](updates/{update_filename}) - Weekly progress update.\n"

    # Insert the new entry after the "Latest Posts" heading
    index_content.insert(latest_posts_line + 1, new_entry)

    # Write the updated index
    with open(index_path, 'w') as f:
        f.writelines(index_content)

    print(f"Updated blog index at {index_path}")

def convert_for_github_pages():
    """Convert blog posts for GitHub Pages."""
    convert_script = Path("blog/convert.py")

    if not convert_script.exists():
        print(f"Warning: Convert script not found at {convert_script}")
        return

    try:
        subprocess.run(["python", str(convert_script), "--github-pages"], check=True)
        print("Converted blog posts for GitHub Pages")
    except subprocess.CalledProcessError:
        print("Error: Failed to convert blog posts for GitHub Pages")

def convert_for_medium():
    """Convert blog posts for Medium."""
    convert_script = Path("blog/convert.py")

    if not convert_script.exists():
        print(f"Warning: Convert script not found at {convert_script}")
        return

    try:
        subprocess.run(["python", str(convert_script), "--medium"], check=True)
        print("Converted blog posts for Medium")
    except subprocess.CalledProcessError:
        print("Error: Failed to convert blog posts for Medium")

def main():
    parser = argparse.ArgumentParser(description="Generate weekly updates for FoNu_NLP_TG project")
    parser.add_argument("--publish", action="store_true", help="Publish updates to GitHub Pages and prepare for Medium")
    args = parser.parse_args()

    # Get current week info
    week_num = get_week_number()
    date_range = get_date_range()

    # Get git changes
    git_changes = get_git_changes()

    # Create update file
    update_path = create_update_file(week_num, date_range, git_changes)
    if not update_path:
        sys.exit(1)

    # Update blog index
    update_blog_index(update_path)

    print(f"\nWeekly update file created at {update_path}")
    print("\nNext steps:")
    print("1. Edit the update file to fill in the details")
    print("2. Commit and push the changes")

    if args.publish:
        # Convert for GitHub Pages and Medium
        convert_for_github_pages()
        convert_for_medium()
        print("3. Your GitHub Pages site will update automatically")
        print("4. Copy the Medium-formatted content to Medium")

if __name__ == "__main__":
    main()
