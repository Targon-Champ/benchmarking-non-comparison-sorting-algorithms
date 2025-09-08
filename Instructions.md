# Collaboration Instructions

This document provides guidelines for contributing to the "Benchmarking Non-Comparison Sorting Algorithms" project. Please follow these instructions to ensure a smooth collaboration process.

## Getting Started

### Prerequisites

- Git installed on your local machine
- A GitHub account
- Appropriate development environment for the project (to be determined)

### Cloning the Repository

1. Navigate to the repository on GitHub: https://github.com/Targon-Champ/benchmarking-non-comparison-sorting-algorithms
2. Click on the "Code" button and copy the repository URL
3. Open your terminal or command prompt
4. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Targon-Champ/benchmarking-non-comparison-sorting-algorithms.git
   ```
5. Navigate to the project directory:
   ```bash
   cd benchmarking-non-comparison-sorting-algorithms
   ```

## Branching Strategy

We follow a feature branching workflow to keep the main branch stable and deployable at all times.

### Creating a New Branch

Before starting work on a new feature or bug fix:

1. Ensure you're on the main branch and it's up to date:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Create a new branch with a descriptive name:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
   For bug fixes, use:
   ```bash
   git checkout -b fix/your-fix-name
   ```

   Examples:
   - `feature/radix-sort-implementation`
   - `fix/benchmark-timing-issue`
   - `docs/update-readme`

## Making Changes

1. Make your changes in your feature branch
2. Add and commit your changes with descriptive commit messages:
   ```bash
   git add .
   git commit -m "Add implementation of Radix Sort algorithm"
   ```

3. Push your changes to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

## Submitting a Pull Request

1. Navigate to the repository on GitHub
2. Switch to your branch using the branch dropdown
3. Click on "New pull request"
4. Set the base branch to `main` and the compare branch to your feature branch
5. Add a descriptive title and detailed description of your changes
6. Submit the pull request

### Pull Request Guidelines

- Ensure your code follows the project's coding standards
- Include tests if applicable
- Update documentation as needed
- Keep pull requests focused on a single feature or bug fix
- Reference any related issues in your pull request description

## Code Review Process

All pull requests must be reviewed and approved by at least one other contributor before merging. During the review process:

1. Address all comments and suggestions
2. Make requested changes in new commits (avoid force pushing unless specifically requested)
3. Request a re-review after addressing feedback

## Keeping Your Branch Updated

To avoid conflicts, regularly sync your feature branch with the main branch:

```bash
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main
```

If there are conflicts, resolve them and commit the changes.

## Best Practices

- Write clear, concise commit messages
- Keep commits focused on a single change
- Test your code before submitting a pull request
- Follow the existing code style and conventions
- Update documentation when making changes to functionality
- Be responsive to feedback during the code review process

## Need Help?

If you have any questions or need assistance:

1. Check existing issues and pull requests
2. Create a new issue describing your question or problem
3. Reach out to the repository maintainers directly

Thank you for contributing to this project!
