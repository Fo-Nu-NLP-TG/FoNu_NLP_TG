# Reports

This directory contains generated reports about the repository, including:

- Submodule status reports
- Weekly activity summaries
- Other automated reports

These reports are generated by the scripts in the `tools` directory and can be included in your weekly updates.

## Submodule Status Reports

Submodule status reports are generated by the `submodule_status.py` script and show:

- The status of each submodule (current, needs update, etc.)
- The last commit date for each submodule
- The number of commits and files changed in the last week

## Usage

To generate a new submodule status report:

```bash
python tools/submodule_status.py
```

This will create a new report in this directory with the current date.
