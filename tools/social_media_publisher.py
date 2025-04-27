#!/usr/bin/env python3
"""
Social Media Publisher for FoNu_NLP_TG

This script helps extract social media snippets from weekly update files
and prepares them for posting on various platforms.

Note: This script doesn't directly post to social media (which would require API keys),
but prepares the content in a format ready for posting.
"""

import os
import sys
import argparse
import re
from pathlib import Path
import datetime
import json

def find_latest_update():
    """Find the most recent weekly update file."""
    updates_dir = Path("blog/updates")
    if not updates_dir.exists():
        print(f"Error: Updates directory not found at {updates_dir}")
        return None
    
    update_files = list(updates_dir.glob("week_*.md"))
    if not update_files:
        print(f"Error: No update files found in {updates_dir}")
        return None
    
    # Sort by modification time (most recent first)
    latest_update = max(update_files, key=os.path.getmtime)
    return latest_update

def extract_social_snippets(update_file):
    """Extract social media snippets from the update file."""
    if not update_file.exists():
        print(f"Error: Update file not found at {update_file}")
        return None
    
    with open(update_file, 'r') as f:
        content = f.read()
    
    snippets = {}
    
    # Extract Twitter/X snippet
    twitter_match = re.search(r'### Twitter/X.*?```(.*?)```', content, re.DOTALL)
    if twitter_match:
        snippets['twitter'] = twitter_match.group(1).strip()
    
    # Extract LinkedIn snippet
    linkedin_match = re.search(r'### LinkedIn.*?```(.*?)```', content, re.DOTALL)
    if linkedin_match:
        snippets['linkedin'] = linkedin_match.group(1).strip()
    
    # Extract Medium teaser
    medium_match = re.search(r'### Medium Teaser.*?```(.*?)```', content, re.DOTALL)
    if medium_match:
        snippets['medium'] = medium_match.group(1).strip()
    
    return snippets

def save_snippets(snippets, output_dir=None):
    """Save the snippets to files for easy copying."""
    if not output_dir:
        output_dir = Path("tools/social_media_output")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Save individual platform files
    for platform, content in snippets.items():
        output_file = output_dir / f"{platform}_{date_str}.txt"
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Saved {platform} content to {output_file}")
    
    # Save all snippets to a single JSON file
    json_file = output_dir / f"all_snippets_{date_str}.json"
    with open(json_file, 'w') as f:
        json.dump(snippets, f, indent=2)
    print(f"Saved all snippets to {json_file}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Extract social media snippets from weekly updates")
    parser.add_argument("--update-file", help="Path to specific update file (defaults to most recent)")
    parser.add_argument("--output-dir", help="Directory to save snippets (defaults to tools/social_media_output)")
    args = parser.parse_args()
    
    # Get the update file
    if args.update_file:
        update_file = Path(args.update_file)
    else:
        update_file = find_latest_update()
        if not update_file:
            sys.exit(1)
    
    # Extract snippets
    snippets = extract_social_snippets(update_file)
    if not snippets:
        print("Error: Failed to extract social media snippets")
        sys.exit(1)
    
    # Save snippets
    output_dir = save_snippets(snippets, args.output_dir)
    
    print("\nSocial media snippets extracted successfully!")
    print(f"Files saved to {output_dir}")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Copy and paste the content to your social media platforms")
    print("3. For Medium, use the full blog post from the 'blog/updates' directory")

if __name__ == "__main__":
    main()
