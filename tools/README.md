# FoNu_NLP_TG Tools

This directory contains utility scripts and tools for managing the FoNu_NLP_TG repository.

## Weekly Update and Social Media Tools

These tools help automate the process of creating weekly updates and sharing them on social media platforms:

### Weekly Update Automation

- `update_automation.py` - Script to generate weekly update templates and update the blog
- `weekly_update_template.md` - Template for weekly updates
- `weekly_update_workflow.md` - Documentation of the weekly update workflow

### Social Media Publishing

- `social_media_publisher.py` - Script to extract social media snippets from weekly updates
- `social_media_output/` - Directory where social media content is saved

### Submodule Management

- `submodule_status.py` - Script to generate reports on submodule status
- `reports/` - Directory where submodule reports are saved

## Usage

### Creating a Weekly Update

```bash
python tools/update_automation.py
```

### Publishing to Social Media

```bash
python tools/social_media_publisher.py
```

### Generating Submodule Reports

```bash
python tools/submodule_status.py
```

For detailed instructions, see `weekly_update_workflow.md`.
