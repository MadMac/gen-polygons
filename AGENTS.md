# AI Agent Instructions for gen-polygons

## Pre-Commit Checks

**ALWAYS run cargo clippy before git commit:**

Before executing any `git commit` command, you MUST:
1. Run: `cargo clippy --all-targets --all-features -- -D warnings`
2. Fix ALL warnings and errors it reports
3. Only then proceed with the commit

This ensures code quality is maintained across all contributions.

## Clippy Configuration

- Target: all targets (`--all-targets`)
- Features: all features (`--all-features`)  
- Warnings as errors: enabled (`-D warnings`)

If clippy fails, do NOT commit. Fix the issues first.
