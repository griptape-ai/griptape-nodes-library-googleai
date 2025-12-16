.PHONY: format lint check fix all clean

# Source directory
SRC_DIR = googleai

# Format code with ruff
format:
	uv run ruff format $(SRC_DIR)

# Lint code with ruff
lint:
	uv run ruff check $(SRC_DIR)

# Check formatting without making changes
check-format:
	uv run ruff format --check $(SRC_DIR)

# Check linting and formatting (CI-friendly)
check: check-format lint

# Fix auto-fixable lint issues
fix:
	uv run ruff check --fix $(SRC_DIR)

# Format and fix all issues
all: format fix

# Install dev dependencies
install-dev:
	uv add --dev ruff

# Show ruff version
version:
	uv run ruff --version
