# PR Review Assistant

A self-contained tool that learns from historical PR review patterns and suggests review comments for new code changes. The assistant uses TF-IDF similarity matching to find relevant historical examples and provides contextual suggestions.

## Features

- **XML Data Support**: Supports two XML schemas for historical review data
  - `prReviewTriplets` format for multiple review examples
  - `network_review_triplet` format for single configuration reviews
- **TF-IDF Similarity Matching**: Uses term frequency-inverse document frequency for intelligent code similarity detection
- **CLI Interface**: Simple command-line interface for easy integration
- **Artifact Generation**: Builds downloadable ZIP packages via GitHub Actions
- **Sample Data**: Includes example triplets for Python, JavaScript, and network configuration

## Quick Start

### Basic Usage

```bash
# Run the PR review assistant
python cli.py --triplets-dir demo/triplets --input-file demo/sample_input.py --out suggestions.md

# With custom parameters
python cli.py --triplets-dir demo/triplets --input-file demo/sample_input.js --top-k 3 --out js_suggestions.md
```

### Output

The tool generates two output files:
- **Markdown file** (`suggestions.md`): Human-readable list of suggested review comments
- **JSON file** (`suggestions.json`): Structured data with similarity scores and metadata

## XML Schemas

### Schema 1: Multiple Triplets (`prReviewTriplets`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<prReviewTriplets>
    <triplet>
        <title>Descriptive title</title>
        <before>Original code before review</before>
        <comment>Review comment</comment>
        <after>Improved code after review</after>
        <repo>repository/name</repo>
        <filePath>path/to/file.py</filePath>
        <commenter>reviewer-username</commenter>
    </triplet>
    <!-- More triplets... -->
</prReviewTriplets>
```

### Schema 2: Single Network Configuration (`network_review_triplet`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<network_review_triplet>
    <description>Review description</description>
    <before_config>Original configuration</before_config>
    <review_comment>Review comment</review_comment>
    <after_config>Improved configuration</after_config>
    <metadata>
        <reviewer>reviewer-name</reviewer>
        <priority>high</priority>
        <category>security</category>
    </metadata>
</network_review_triplet>
```

## Architecture

### Core Components

- **`pr_reviewer/models.py`**: Data models for review triplets
- **`pr_reviewer/loader.py`**: XML parsing and loading functionality
- **`pr_reviewer/similarity.py`**: TF-IDF similarity calculation engine
- **`pr_reviewer/reviewer.py`**: Main review logic and CLI interface

### Algorithm

1. **Load Training Data**: Parse XML files containing historical review triplets
2. **Build TF-IDF Model**: Create vocabulary and calculate term importance weights
3. **Similarity Matching**: Compare input code against historical examples using cosine similarity
4. **Aggregation**: Collect and deduplicate review suggestions from top matches
5. **Output Generation**: Create markdown and JSON outputs with suggestions

## Development

### Project Structure

```
├── pr_reviewer/           # Main package
│   ├── __init__.py       # Package initialization
│   ├── models.py         # Data models
│   ├── loader.py         # XML parsing
│   ├── similarity.py     # TF-IDF engine
│   └── reviewer.py       # CLI and main logic
├── demo/                 # Sample data and examples
│   ├── triplets/         # XML training data
│   ├── sample_input.py   # Python example
│   └── sample_input.js   # JavaScript example
├── scripts/              # Build utilities
│   └── build_zip.py      # ZIP artifact builder
├── .github/workflows/    # GitHub Actions
│   └── build.yml         # CI/CD pipeline
└── cli.py               # CLI entry point
```

### Testing

```bash
# Test XML loading
python -c "from pr_reviewer.loader import load_from_dir; print(f'Loaded {len(load_from_dir(\"demo/triplets\"))} triplets')"

# Test similarity engine
python -c "from pr_reviewer.similarity import tokenize; print(tokenize('def hello_world(): pass'))"

# End-to-end test
python cli.py --triplets-dir demo/triplets --input-file demo/sample_input.py
```

### Building Artifacts

```bash
# Create distributable ZIP
python scripts/build_zip.py

# Output: dist/pr_reviewer.zip
```

## GitHub Actions Integration

The repository includes a GitHub Actions workflow that:

1. **Tests Functionality**: Validates XML loading and CLI operations
2. **Builds Artifacts**: Creates a downloadable ZIP package
3. **Uploads Artifacts**: Makes the tool available for download from PR builds

The workflow triggers on:
- Pull requests to `main` branch
- Pushes to `main` branch

## Example Usage

### Python Code Review

Input file (`demo/sample_input.py`):
```python
def process_data(data_file):
    with open(data_file, 'r') as f:
        content = json.load(f)
    
    results = []
    for item in content:
        if item['status'] == 'pending':
            results.append(item['id'])
    
    return results
```

Generated suggestions might include:
- Consider adding error handling for file not found and JSON parsing errors
- This can be simplified using a list comprehension for better readability and performance
- Please add type hints to improve code documentation and enable better IDE support

### JavaScript Code Review

Input file (`demo/sample_input.js`):
```javascript
function saveUserProfile(profileData, callback) {
    database.update('users', profileData.id, profileData, (err, result) => {
        if (err) {
            callback(err);
        } else {
            callback(null, result);
        }
    });
}
```

Generated suggestions might include:
- Consider using async/await pattern for better error handling and readability
- Missing input validation - should validate required fields and data types

## License

This project is part of Hackathon2025.
