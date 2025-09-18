# Project Template

A minimal template for modern Python projects.

## How to use this template to create a new project repository

1. Select this template under the "Repository template" dropdown when you create a repository in Github UI.
2. Clone the repository and open this README.md in your editor to continue from there.
3. Go over the files in the created repository and understand what each tool/file does.
4. Update this readme by replacing the "MY_AWESOME_PROJECT" name below and briefly describe the project.
5. If you want to use mypy for typing, uncomment mypy related configs in [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml)

## How to contribute to this template

1. Create a descriptive issue in this project.

# MY_AWESOME_PROJECT

## Get started

### Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Requirements

Most of the requirements are installed with the following command

```bash
cd path/to/git-repo
uv venv
make install_dependencies
```

### Other Commands

Other useful commands for the project can be found in the [Makefile](Makefile).
