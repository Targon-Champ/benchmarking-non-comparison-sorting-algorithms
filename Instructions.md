# Contributing to Benchmarking Non-Comparison Sorting Algorithms

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing to the repository.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Branch Strategy](#branch-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Communication](#communication)

## Getting Started

### Prerequisites

- Git installed on your machine
- GitHub account
- Basic understanding of sorting algorithms
- Familiarity with the programming language used in this project

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub by clicking the "Fork" button

2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/benchmarking-non-comparison-sorting-algorithms.git
   cd benchmarking-non-comparison-sorting-algorithms
   ```

3. **Add the original repository as upstream**:
   ```bash
   git remote add upstream https://github.com/Targon-Champ/benchmarking-non-comparison-sorting-algorithms.git
   ```

4. **Verify your remotes**:
   ```bash
   git remote -v
   ```
   You should see both `origin` (your fork) and `upstream` (original repo)

## How to Contribute

### Types of Contributions We Welcome

- **Algorithm Implementations**: New non-comparison sorting algorithms
- **Performance Optimizations**: Improvements to existing implementations
- **Bug Fixes**: Corrections to existing code
- **Documentation**: Improvements to README, comments, or guides
- **Test Cases**: Additional benchmarking scenarios or test datasets
- **Benchmarking Tools**: Enhancements to measurement and analysis tools

## Branch Strategy

### Protected Branches

- **`main`**: Production-ready code. Direct pushes are **not allowed**.
- All contributions must go through Pull Requests

### Creating a Feature Branch

Always create a new branch for your work:

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming Conventions

- Feature branches: `feature/descriptive-name`
- Bug fixes: `fix/issue-description`
- Documentation: `docs/what-you-are-documenting`
- Performance improvements: `perf/optimization-description`

Examples:
- `feature/radix-sort-implementation`
- `fix/counting-sort-negative-numbers`
- `docs/update-readme`
- `perf/optimize-bucket-sort`

## Commit Guidelines

### Commit Message Format

Write clear, concise commit messages:

```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (formatting, semicolons, etc.)

### Examples

```bash
git commit -m "feat: implement radix sort algorithm"

git commit -m "fix: handle negative numbers in counting sort"

git commit -m "docs: add benchmark results for large datasets"

git commit -m "perf: optimize memory usage in bucket sort"
```

## Pull Request Process

### Before Submitting a Pull Request

1. **Sync with upstream** to avoid conflicts:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Test your changes** thoroughly

3. **Update documentation** if necessary

4. **Commit your changes** following the commit guidelines

### Submitting a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Go to the original repository
   - Click "New Pull Request"
   - Select "compare across forks"
   - Choose your fork and branch
   - Click "Create Pull Request"

3. **Fill out the PR template** with:
   - Clear title describing your changes
   - Detailed description of what you changed and why
   - Reference any related issues (e.g., "Fixes #123")
   - Screenshots or benchmarks if applicable

### PR Review Process

- A maintainer will review your PR within 3-5 business days
- Address any requested changes by pushing new commits to your branch
- Once approved, a maintainer will merge your PR
- **Do not merge your own PRs**

### After Your PR is Merged

1. **Delete your feature branch**:
   ```bash
   git checkout main
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Update your local main**:
   ```bash
   git pull upstream main
   ```

## Code Standards

### General Guidelines

- Write clean, readable, and well-documented code
- Follow existing code style and conventions
- Add comments for complex algorithms or logic
- Use meaningful variable and function names
- Keep functions small and focused on a single task

### Algorithm Implementation Standards

When implementing sorting algorithms:

1. **Function Signature**: Follow the existing pattern
2. **Input Validation**: Check for edge cases (empty arrays, single elements, etc.)
3. **Time Complexity**: Document the time and space complexity
4. **Comments**: Explain the algorithm's approach and key steps

Example:
```python
def radix_sort(arr):
    """
    Sorts an array using Radix Sort algorithm.
    
    Time Complexity: O(d * (n + k)) where d is the number of digits
    Space Complexity: O(n + k)
    
    Args:
        arr: List of non-negative integers
        
    Returns:
        Sorted list
    """
    # Implementation here
```

## Testing Requirements

### Before Submitting

- Test your code with various input sizes
- Include edge cases:
  - Empty arrays
  - Single element arrays
  - Already sorted arrays
  - Reverse sorted arrays
  - Arrays with duplicates
  - Large datasets

### Benchmark Results

If you're adding or modifying algorithms, include benchmark results:
- Dataset sizes tested
- Execution times
- Memory usage
- Comparison with other algorithms

## Communication

### Getting Help

- **Questions**: Open a [Discussion](https://github.com/Targon-Champ/benchmarking-non-comparison-sorting-algorithms/discussions) or create an [issue](https://github.com/Targon-Champ/benchmarking-non-comparison-sorting-algorithms/issues)
- **Bug Reports**: Use the issue template to report bugs
- **Feature Requests**: Open an issue describing the feature

### Code of Conduct

- Be respectful and professional
- Welcome newcomers and help them get started
- Provide constructive feedback in reviews
- Focus on the code, not the person

## Questions?

If you have any questions about contributing, feel free to:
- Open an issue with the "question" label
- Reach out to the maintainers
- Check existing issues and discussions

---
---

Thank you for contributing! Your efforts help make this project better for everyone. 🎉