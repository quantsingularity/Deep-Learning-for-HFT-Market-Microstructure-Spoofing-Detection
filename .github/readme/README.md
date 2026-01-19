# CI Formatting Check

> GitHub Actions workflow to enforce repository-wide formatting and linting for Python, Markdown, and YAML files.

[![CI Formatting Check](https://img.shields.io/badge/ci-formatting--check-blue.svg)]()

---

## Table of contents

- [Overview](#overview)
- [Workflow triggers](#workflow-triggers)
- [What it checks](#what-it-checks)
- [Job steps (summary)](#job-steps-summary)
- [Run locally](#run-locally)
- [Environment variables](#environment-variables)
- [Common failures & fixes](#common-failures--fixes)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository includes a GitHub Actions workflow **CI Formatting Check** that runs on `push` and `pull_request` for the `main` and `develop` branches and can also be run manually via `workflow_dispatch`.

The workflow verifies repository-wide formatting expectations using:

- `autoflake` - to detect & optionally remove unused imports/variables (Python)
- `black` - to check Python code formatting
- `prettier` - to check Markdown and YAML formatting

It is intentionally conservative: autoflake runs against a temporary copy and the job fails only if diffs are found (i.e., the repo is not formatted as expected).

---

## Workflow triggers

- `push` to `main` or `develop`
- `pull_request` targeting `main` or `develop`
- Manual trigger via the Actions UI (`workflow_dispatch`)

---

## What it checks

| Area                              | Tool        |                 Files checked | Mode                                                      |
| --------------------------------- | ----------- | ----------------------------: | --------------------------------------------------------- |
| Python unused imports & variables | `autoflake` | `code/` (or `${BACKEND_DIR}`) | Dry-run via temp copy; job fails if changes would be made |
| Python formatting                 | `black`     |                       `code/` | `--check` (no in-place formatting)                        |
| Markdown formatting               | `prettier`  |                     `**/*.md` | `--check`                                                 |
| YAML formatting                   | `prettier`  |             `**/*.{yml,yaml}` | `--check`                                                 |

---

## Job steps (summary)

| Step                      | Purpose                                | Main action / Command                                                                                                                          |
| ------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Checkout repository       | Get full repo history (fetch-depth: 0) | `actions/checkout@v4`                                                                                                                          |
| Setup Python              | Provide Python runtime for formatters  | `actions/setup-python@v5` (`python-version: '3.10'`)                                                                                           |
| Cache pip                 | Speed up pip install                   | `actions/cache@v4` (cache `~/.cache/pip`)                                                                                                      |
| Install Python formatters | Install `autoflake`, `black`           | `pip install --upgrade pip && pip install autoflake black`                                                                                     |
| Run autoflake             | Detect unused imports/vars (temp copy) | `autoflake --remove-all-unused-imports --remove-unused-variables --recursive --ignore-init-module-imports --in-place $TMP_COPY` then `diff -r` |
| Run black (check)         | Check code style                       | `black ${BACKEND_DIR} --check`                                                                                                                 |
| Setup Node.js             | Provide Node runtime for Prettier      | `actions/setup-node@v4` (`node-version: '18'`)                                                                                                 |
| Run Prettier (.md)        | Check Markdown                         | `npx prettier@latest --check "**/*.md" --ignore-path .prettierignore`                                                                          |
| Run Prettier (YAML)       | Check YAML                             | `npx prettier@latest --check "**/*.{yml,yaml}" --ignore-path .prettierignore`                                                                  |
| Finalize check            | Announcement                           | `echo "All repository-wide formatting checks completed successfully."`                                                                         |

---

## Run locally

You can run the equivalent checks locally to reproduce the CI behavior.

1. Install dependencies (Python & Node):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip autoflake black
npm install --no-save prettier@latest
```

2. Run autoflake against a temporary copy (same logic as CI):

```bash
BACKEND_DIR=code
TMP_COPY=$(mktemp -d)
cp -a "${BACKEND_DIR}/." "$TMP_COPY/"
autoflake --remove-all-unused-imports --remove-unused-variables --recursive --ignore-init-module-imports --in-place "$TMP_COPY"
if ! diff -r "$BACKEND_DIR" "$TMP_COPY" > /dev/null; then
  echo "autoflake found formatting/unused-imports issues in backend."
  echo "Run: autoflake --remove-all-unused-imports --remove-unused-variables --recursive --ignore-init-module-imports --in-place ${BACKEND_DIR}"
  rm -rf "$TMP_COPY"
  exit 1
fi
rm -rf "$TMP_COPY"
```

3. Run black in check mode:

```bash
black "${BACKEND_DIR}" --check
```

4. Run prettier checks:

```bash
npx prettier@latest --check "**/*.md" --ignore-path .prettierignore
npx prettier@latest --check "**/*.{yml,yaml}" --ignore-path .prettierignore
```

---

## Environment variables

| Variable      | Default | Description                                         |
| ------------- | ------- | --------------------------------------------------- |
| `BACKEND_DIR` | `code`  | Directory that contains the Python backend to check |

If your backend lives in a different folder, set `BACKEND_DIR` in the workflow or pass it as an environment variable in local runs.

---

## Common failures & fixes

- **`autoflake` found formatting/unused-imports issues**
  - Solution: run `autoflake --remove-all-unused-imports --remove-unused-variables --recursive --ignore-init-module-imports --in-place ${BACKEND_DIR}` locally, commit changes, and push.

- **`black` failed with non-zero exit**
  - Solution: run `black ${BACKEND_DIR}` locally and commit changed files.

- **`prettier` reports formatting issues**
  - Solution: run `npx prettier@latest --write "**/*.md"` (or for YAML `"**/*.{yml,yaml}"`), verify changes and commit.

---

## Contributing

Please ensure all formatting checks pass locally before opening a pull request. Use the commands above to fix formatting issues. If you need to change the rules (for example, adjust `autoflake` flags or Python version), open a PR against the workflow file and describe the reason for the change.

---

## License

This workflow and README are released under the same license as the repository.

---
