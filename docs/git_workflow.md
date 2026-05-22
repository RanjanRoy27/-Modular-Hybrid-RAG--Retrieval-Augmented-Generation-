# Git Workflow

This repo should tell a clean story on GitHub: one purpose per commit, short-lived branches, and a mostly linear `main` branch.

## Branches

- `main`: stable, deployable, protected when possible.
- `feature/<short-name>`: new features.
- `fix/<short-name>`: bug fixes.
- `docs/<short-name>`: documentation-only changes.
- `chore/<short-name>`: dependency, tooling, or structure cleanup.

Examples:

```bash
git switch -c feature/hybrid-retrieval
git switch -c fix/streaming-state
git switch -c docs/deployment-guide
```

## Commit Style

Use small commits with imperative messages:

```text
Add session persistence
Fix stream endpoint initialization
Document local setup
Move smoke tests into tests directory
```

Good commits should:

- Explain one logical change.
- Include related tests or docs when practical.
- Avoid mixing formatting, refactors, and feature behavior in one commit.

## Linear History

Preferred flow:

```bash
git fetch origin
git rebase origin/main
git push
```

For pull requests, use **Squash and merge** or **Rebase and merge**. Avoid merge commits unless there is a clear reason.

If your branch already exists on GitHub and you rebased it:

```bash
git push --force-with-lease
```

Use `--force-with-lease`, not plain `--force`.

## Pull Request Checklist

Before merging:

- The branch is up to date with `origin/main`.
- The app starts locally or the limitation is documented.
- Relevant smoke tests or syntax checks pass.
- README/docs are updated for user-visible changes.
- `CHANGELOG.md` has an `Unreleased` entry.

## Versioning

Use Semantic Versioning:

- `v0.1.0`: initial working MVP.
- `v0.2.0`: backwards-compatible feature release.
- `v0.2.1`: bug fix or docs/tooling patch.
- `v1.0.0`: stable public API and deployment contract.

Release command:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Before tagging a release:

- Move `CHANGELOG.md` entries from `Unreleased` to the version section.
- Confirm `.env.example`, README, and deployment docs match the release.
- Confirm generated files, secrets, local databases, and client documents are not committed.

## Current History Note

The repository already contains one historical merge commit from earlier development. Do not rewrite `main` unless everyone using the repo agrees. From this point forward, keep history linear with rebase or squash merges.

