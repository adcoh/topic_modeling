# Pre-commit setup complete

This project now has pre-commit hooks configured that will run:

- **Ruff linting**: Checks for code quality issues and applies fixes
- **Ruff formatting**: Ensures consistent code formatting
- **Pyright type checking**: Validates type annotations
- **Basic file checks**: Removes trailing whitespace, fixes end-of-file issues, validates YAML/TOML

The hooks will run automatically on every commit. You can also run them manually:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run
```
