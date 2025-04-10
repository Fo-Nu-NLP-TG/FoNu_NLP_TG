# Handling Large Files in the FoNu_NLP_TG Project

This document provides guidelines for handling large files in the FoNu_NLP_TG project.

## GitHub File Size Limits

GitHub has the following file size limits:

- **Recommended maximum**: 50 MB
- **Hard limit**: 100 MB

Files larger than these limits will cause issues when pushing to GitHub.

## Large Files in This Project

The following files have been identified as exceeding GitHub's limits:

- `data/processed/clean_ewe_french.csv` (99.09 MB)
- `data/processed/ewe_french.csv` (103.24 MB)

## Solutions for Handling Large Files

### 1. Use Git LFS (Large File Storage)

Git LFS is designed to handle large files in Git repositories. It replaces large files with text pointers inside Git, while storing the file contents on a remote server.

#### Setting Up Git LFS

1. Install Git LFS:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # macOS
   brew install git-lfs
   
   # Windows
   # Download from https://git-lfs.github.com/
   ```

2. Initialize Git LFS:
   ```bash
   git lfs install
   ```

3. Track large file types:
   ```bash
   git lfs track "*.csv"
   git lfs track "data/processed/*.csv"
   ```

4. Add the `.gitattributes` file:
   ```bash
   git add .gitattributes
   git commit -m "Add Git LFS tracking for large files"
   ```

### 2. Exclude Large Files from Version Control

If you don't need to track large files in Git, you can exclude them using `.gitignore`:

```bash
# Add to .gitignore
data/processed/*.csv
```

### 3. Split Large Files

If possible, split large files into smaller chunks:

```bash
# Example: Split a large CSV file into 50MB chunks
split -b 50m data/processed/large_file.csv data/processed/large_file_part_
```

### 4. Use Data Version Control (DVC)

DVC is a tool specifically designed for versioning data files and ML models:

1. Install DVC:
   ```bash
   pip install dvc
   ```

2. Initialize DVC:
   ```bash
   dvc init
   ```

3. Add large files to DVC:
   ```bash
   dvc add data/processed/large_file.csv
   ```

4. Set up a remote storage:
   ```bash
   dvc remote add -d myremote s3://mybucket/path
   ```

5. Push data to remote storage:
   ```bash
   dvc push
   ```

## Removing Large Files from Git History

If you've already committed large files to Git, you'll need to remove them from the history:

### Using the Provided Script

We've created a script to help you remove large files from Git tracking:

```bash
chmod +x remove_large_files.sh
./remove_large_files.sh
```

### Manual Removal

1. Remove the file from Git tracking:
   ```bash
   git rm --cached data/processed/large_file.csv
   ```

2. Commit the change:
   ```bash
   git commit -m "Remove large file from tracking"
   ```

3. Update `.gitignore` to prevent re-adding:
   ```bash
   echo "data/processed/large_file.csv" >> .gitignore
   git add .gitignore
   git commit -m "Update .gitignore to exclude large file"
   ```

### Using BFG Repo-Cleaner

For more complex cases, you can use the BFG Repo-Cleaner:

1. Download BFG: https://rtyley.github.io/bfg-repo-cleaner/

2. Remove large files:
   ```bash
   java -jar bfg.jar --strip-blobs-bigger-than 100M my-repo.git
   ```

3. Clean up and push:
   ```bash
   cd my-repo.git
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   git push -f origin main:main
   ```

## Best Practices

1. **Never commit large binary files directly to Git**
2. **Use Git LFS for large files that need versioning**
3. **Keep datasets in a separate location and document how to obtain them**
4. **Consider using data version control tools like DVC**
5. **Regularly check the size of files before committing**

## Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub Documentation on Large Files](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [DVC Documentation](https://dvc.org/doc)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
