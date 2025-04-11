# Contributing Guide

Thank you for your interest in contributing to the FoNu NLP TG project! This guide will help you get started with contributing to the repository.

## Table of Contents

- [Code Organization](#code-organization)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code Organization

The repository is organized into the following main directories:

- `Attention_Is_All_You_Need/`: Core transformer implementation
- `data_processing/`: Data preparation and tokenization tools
- `documentation/`: Project documentation files
- `evaluation/`: Model evaluation scripts
- `Research/`: Research papers and reports
- `tools/`: Utility scripts for repository management
- `blog/`: Project blog and updates

When adding new code, please follow these guidelines:

1. **Place files in the appropriate directory** based on their functionality
2. **Create new directories** if your contribution doesn't fit into existing ones
3. **Update documentation** to reflect any changes to the repository structure

## Development Workflow

1. **Set up your environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the coding standards

4. **Test your changes** thoroughly

5. **Submit a pull request** with a clear description of the changes

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use meaningful variable and function names
- Include docstrings for all functions, classes, and modules
- Write unit tests for new functionality
- Keep functions focused on a single responsibility
- Use type hints where appropriate

Example function with proper documentation:

```python
def tokenize_text(text: str, language: str = "ewe") -> List[str]:
    """
    Tokenize text using the appropriate tokenizer for the specified language.
    
    Args:
        text: The input text to tokenize
        language: The language of the text (default: "ewe")
        
    Returns:
        A list of tokens
        
    Raises:
        ValueError: If the language is not supported
    """
    # Implementation here
```

## Documentation Guidelines

- Write documentation in Markdown format
- Place general documentation in the `documentation/` directory
- Update the README.md when adding major features
- Include code examples where appropriate
- Document any environment setup or dependencies
- Use diagrams (Mermaid) for complex workflows or architectures

### Adding Documentation to the Website

To add documentation to the website:

1. Create a new Markdown file in the appropriate directory under `docs/`
2. Add the file to the navigation in `mkdocs.yml`
3. Run `mkdocs serve` to preview the changes locally
4. Submit a pull request with your changes

## Pull Request Process

1. Ensure your code follows the coding standards
2. Update documentation to reflect your changes
3. Add or update tests as necessary
4. Make sure all tests pass
5. Submit the pull request with a clear description of:
   - What changes were made
   - Why the changes were necessary
   - Any potential issues or limitations
   - References to related issues

## Issue Reporting

When reporting issues, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, etc.)
6. Any relevant logs or error messages

## Data Handling

When working with data:

1. **Never commit large data files** to the repository
2. Use `.gitignore` to exclude data files
3. Document how to obtain or generate the data
4. Consider using Git LFS for tracking large files if absolutely necessary

## Contributing to the Blog

To contribute a blog post:

1. Create a new Markdown file in the `docs/blog/` directory
2. Follow the format of existing blog posts
3. Include a title, date, and author
4. Submit a pull request with your blog post

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to FoNu NLP TG!
