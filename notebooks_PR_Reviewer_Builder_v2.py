#!/usr/bin/env python
# coding: utf-8

# # PR Reviewer Builder v2
# 
# This notebook builds a dataset for few-shot learning by identifying:
# - Commit before SME review
# - Commit after SME review  
# - File paths of changes
# - SME comment text
# - Whether the after-commit actually addresses the comment
# 
# The algorithms help detect whether commits after SME reviews actually respond to review comments.

# ## Setup and Imports

# In[ ]:


import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# For fuzzy string matching
try:
    from rapidfuzz import fuzz
except ImportError:
    print("rapidfuzz not installed. Installing...")
    get_ipython().system('pip install rapidfuzz')
    from rapidfuzz import fuzz

# For data processing
import pandas as pd
import numpy as np


# ## Data Structures

# In[ ]:


@dataclass
class CodeSpan:
    """Represents a span of code with line numbers and content."""
    start_line: int
    end_line: int
    content: str
    file_path: str

@dataclass
class ReviewComment:
    """Represents an SME review comment."""
    comment_id: str
    text: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None

@dataclass
class CommitInfo:
    """Represents information about a commit."""
    commit_sha: str
    message: str
    files_changed: List[str]
    timestamp: str

class FollowThroughAdvanced:
    """Advanced follow-through scoring class."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def calculate_score(self, before_code: str, after_code: str, comment: str) -> float:
        """Calculate follow-through score based on code changes and comment."""
        # This will be implemented in the scoring section
        # Implement a simple scoring algorithm based on code changes and comment relevance
        if not before_code and not after_code:
            return 0.0

        # Calculate how much the code changed
        edit_distance = normalized_edit_distance(before_code, after_code)

        # Look for keywords in comment that suggest the type of change expected
        comment_lower = comment.lower()
        action_keywords = ['fix', 'update', 'change', 'modify', 'improve', 'refactor', 'add', 'remove']

        keyword_score = 0.0
        for keyword in action_keywords:
            if keyword in comment_lower:
                keyword_score += 0.1

        keyword_score = min(keyword_score, 1.0)

        # Combine edit distance with keyword relevance
        combined_score = (edit_distance * 0.7) + (keyword_score * 0.3)

        return min(combined_score, 1.0)


# ## Data Harvesting Section
# 
# This section contains the existing harvesting logic that should remain unchanged.

# In[ ]:


def harvest_pr_data(repo_path: str) -> Dict[str, Any]:
    """Harvest PR data from repository.

    This function remains unchanged and handles the data collection.
    """
    # Placeholder for existing harvesting logic
    return {
        'commits': [],
        'reviews': [],
        'file_changes': []
    }

def extract_code_spans(file_path: str, line_ranges: List[Tuple[int, int]]) -> List[CodeSpan]:
    """Extract code spans from file at specified line ranges.

    This function remains unchanged.
    """
    spans = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for start, end in line_ranges:
                content = ''.join(lines[start-1:end]) if start <= len(lines) else ''
                spans.append(CodeSpan(start, end, content, file_path))
    return spans


# ## Scoring Algorithms Section
# 
# This section contains the algorithms that need to be implemented to detect whether commits after SME reviews actually respond to review comments.

# In[ ]:


# TODO: Implement these functions

def concat_snips(spans: List[CodeSpan]) -> str:
    """Concatenate code snippets from spans for analysis.

    Args:
        spans: List of CodeSpan objects containing code snippets

    Returns:
        str: Concatenated code text from all spans
    """
    if not spans:
        return ""

    # Sort spans by file path and then by start line for consistent ordering
    sorted_spans = sorted(spans, key=lambda s: (s.file_path, s.start_line))

    # Concatenate content from all spans with file separators
    result_parts = []
    current_file = None

    for span in sorted_spans:
        # Add file header if we're switching to a new file
        if current_file != span.file_path:
            if current_file is not None:
                result_parts.append("\\n" + "="*50 + "\\n")
            result_parts.append(f"# File: {span.file_path}\\n")
            current_file = span.file_path

        # Add line number information and content
        result_parts.append(f"# Lines {span.start_line}-{span.end_line}:\\n")
        result_parts.append(span.content)

        # Add separator between spans in same file
        if not span.content.endswith("\\n"):
            result_parts.append("\\n")
        result_parts.append("\\n")

    return "".join(result_parts).strip()


def normalized_edit_distance(before_code: str, after_code: str) -> float:
    """Calculate normalized edit distance between before/after code using fuzzy matching.

    Args:
        before_code: Code before the change
        after_code: Code after the change

    Returns:
        float: Normalized edit distance (0.0 = identical, 1.0 = completely different)
    """
    if not before_code and not after_code:
        return 0.0

    if not before_code or not after_code:
        return 1.0

    # Normalize whitespace for better comparison
    before_normalized = ' '.join(before_code.split())
    after_normalized = ' '.join(after_code.split())

    # Calculate similarity ratio using rapidfuzz
    similarity = fuzz.ratio(before_normalized, after_normalized) / 100.0

    # Return distance (1 - similarity)
    return 1.0 - similarity


def simple_locality_score(comment_text: str, file_paths: List[str]) -> float:
    """Compute locality scores based on file path mentions in comments.

    Args:
        comment_text: The review comment text
        file_paths: List of file paths that were changed

    Returns:
        float: Locality score (0.0 = no file path mentions, 1.0 = all files mentioned)
    """
    if not comment_text or not file_paths:
        return 0.0

    # Normalize comment text to lowercase for case-insensitive matching
    comment_lower = comment_text.lower()

    # Count how many file paths are mentioned in the comment
    mentioned_files = 0

    for file_path in file_paths:
        # Check various forms of the file path
        file_name = file_path.split('/')[-1]  # Just filename
        file_path_lower = file_path.lower()
        file_name_lower = file_name.lower()

        # Check if file path or filename is mentioned
        if (file_path_lower in comment_lower or 
            file_name_lower in comment_lower or
            # Also check without extension
            file_name_lower.split('.')[0] in comment_lower):
            mentioned_files += 1

    # Return ratio of mentioned files to total files
    return mentioned_files / len(file_paths)


def score_follow_through(before_commit: CommitInfo, 
                        after_commit: CommitInfo,
                        review_comment: ReviewComment,
                        before_spans: List[CodeSpan],
                        after_spans: List[CodeSpan]) -> Dict[str, float]:
    """Score follow-through using the FollowThroughAdvanced class with proper threshold handling.

    Args:
        before_commit: Commit information before SME review
        after_commit: Commit information after SME review
        review_comment: SME review comment
        before_spans: Code spans before the change
        after_spans: Code spans after the change

    Returns:
        Dict[str, float]: Dictionary containing various scores:
            - 'edit_distance': Normalized edit distance between code
            - 'locality_score': How well comment matches changed files
            - 'follow_through_score': Overall follow-through score
            - 'addresses_comment': Binary score (0/1) if comment is addressed
    """
    # Concatenate code snippets for comparison
    before_code = concat_snips(before_spans)
    after_code = concat_snips(after_spans)

    # Calculate normalized edit distance
    edit_distance = normalized_edit_distance(before_code, after_code)

    # Calculate locality score
    all_file_paths = list(set(after_commit.files_changed + before_commit.files_changed))
    locality_score = simple_locality_score(review_comment.text, all_file_paths)

    # Use FollowThroughAdvanced to calculate overall score
    follow_through = FollowThroughAdvanced(threshold=0.5)
    follow_through_score = follow_through.calculate_score(
        before_code, after_code, review_comment.text
    )

    # Calculate if comment is addressed based on combined factors
    # High edit distance (changes were made) + high locality (relevant files) = good follow-through
    change_significance = min(edit_distance * 2, 1.0)  # Scale edit distance
    combined_score = (change_significance * 0.4 + locality_score * 0.3 + 
                     (follow_through_score or 0.5) * 0.3)

    addresses_comment = 1.0 if combined_score > 0.5 else 0.0

    return {
        'edit_distance': edit_distance,
        'locality_score': locality_score,
        'follow_through_score': follow_through_score or 0.5,
        'addresses_comment': addresses_comment
    }


# ## Export Section
# 
# This section contains the existing export logic that should remain unchanged.

# In[ ]:


def export_dataset(data: Dict[str, Any], output_path: str) -> None:
    """Export the processed dataset to file.

    This function remains unchanged and handles the data export.
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Dataset exported to {output_path}")

def create_few_shot_examples(scored_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create few-shot learning examples from scored data.

    This function remains unchanged.
    """
    examples = []
    for item in scored_data:
        if item.get('addresses_comment', 0) > 0.5:  # Threshold for positive examples
            examples.append({
                'before_code': item.get('before_code', ''),
                'after_code': item.get('after_code', ''),
                'comment': item.get('comment_text', ''),
                'label': 'addresses_comment'
            })
    return examples


# ## Main Processing Pipeline
# 
# This section orchestrates the entire process.

# In[ ]:


def main_pipeline(repo_path: str, output_path: str) -> None:
    """Main pipeline to process PR data and build dataset."""
    print("Starting PR Reviewer dataset building...")

    # Step 1: Harvest data (unchanged)
    print("Harvesting PR data...")
    pr_data = harvest_pr_data(repo_path)

    # Step 2: Process and score data using implemented algorithms
    print("Processing and scoring data...")
    scored_results = []

    # This would iterate through the harvested data and apply scoring
    # Implementation details depend on the harvested data structure

    # Step 3: Export results (unchanged)
    print("Exporting dataset...")
    export_dataset({
        'scored_results': scored_results,
        'metadata': {
            'total_items': len(scored_results),
            'repo_path': repo_path
        }
    }, output_path)

    print("Dataset building complete!")

# Example usage (commented out)
# main_pipeline('/path/to/repo', 'pr_reviewer_dataset.json')


# ## Testing Section
# 
# This section will be used to test the implemented functions.

# In[ ]:


# Test the implemented functions once they are completed
def test_algorithms():
    """Test the implemented algorithm functions."""
    print("Testing implemented algorithms...")

    # Test data
    test_spans = [
        CodeSpan(1, 3, "def hello():\n    print('Hello')\n    return True", "test.py"),
        CodeSpan(5, 7, "def world():\n    print('World')\n    return False", "test.py")
    ]

    # Test concat_snips
    print("Testing concat_snips...")
    result = concat_snips(test_spans)
    # print(f"Concatenated result: {result[:100]}...")  # First 100 chars

    # Test normalized_edit_distance
    print("Testing normalized_edit_distance...")
    distance = normalized_edit_distance("hello world", "hello earth")
    # print(f"Edit distance: {distance}")

    # Test simple_locality_score
    print("Testing simple_locality_score...")
    score = simple_locality_score("Please fix the issue in test.py", ["test.py", "main.py"])
    # print(f"Locality score: {score}")

    print("Algorithm testing complete!")

    # Functions are now implemented, ready to test
test_algorithms()

