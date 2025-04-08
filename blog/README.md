# FoNu_NLP_TG Project Blog

This directory contains the blog posts and related assets for the FoNu_NLP_TG project.

## Structure

- `index.md` - The main index page for the blog
- `*.md` - Individual blog posts
- `images/` - Directory containing images used in blog posts

## How to Add a New Post

1. Create a new Markdown file in this directory with a descriptive name (e.g., `transformer_attention_explained.md`)
2. Add any images to the `images/` directory
3. Update the `index.md` file to include a link to your new post
4. Commit and push your changes

## Formatting Guidelines

- Use Markdown for all posts
- Include a title at the top of each post using a level 1 heading (`# Title`)
- Include the date of the post near the top
- Use appropriate heading levels (H2 for main sections, H3 for subsections)
- Include images where helpful
- Link to relevant resources and references
- End with a conclusion or "what's next" section

## Publishing

### GitHub Pages

We use GitHub Pages to publish the blog. To prepare files for GitHub Pages:

```bash
python convert.py --github-pages
```

This will create a `docs` directory in the project root with all necessary files. Then:

1. Push the changes to GitHub
2. Go to the repository settings
3. Under "GitHub Pages", select "main branch /docs folder" as the source
4. Your blog will be available at `https://[username].github.io/FoNu_NLP_TG/`

### Medium

We also publish our blog posts on Medium. To prepare a post for Medium:

```bash
python convert.py --medium
```

This will create Medium-ready markdown files that you can copy and paste into Medium's editor. The script automatically:

1. Updates image paths to point to GitHub
2. Adds a footer with a link back to the GitHub project
3. Formats the content to work well on Medium

## License

All blog content is licensed under the same license as the main project.
