# GitHub integration (design, not enabled)

Today the hands work against a sandbox repository on disk (`workspaces/origin.git`, see
`src/workspace.py`). This is the design for putting a real GitHub repository behind that
same flow. **Nothing here is built or enabled.** Turning it on needs a sandbox repo and an
explicit OK from the owner; no code pushes to GitHub before then.

## What maps to what

The board stays the source of truth: the agents still walk to the board to start, block,
review, approve and finish tasks. GitHub mirrors those decisions; it never makes them.

| On the board / in the sim | Today (local) | With GitHub |
|---|---|---|
| A work session starts on T-3 | branch `task/T-3` from `origin/main` | same, from the GitHub `main` |
| The hands finish T-3 (tests pass) | `publish`: push `task/T-3` to `origin.git` | push `task/T-3` to GitHub |
| Owner moves T-3 to review at the board, with a note | the note is on the card | open a PR `task/T-3` -> `main`, titled `T-3: <title>`, the note as its body |
| Owner marks T-3 blocked at the board | the note is on the card | convert the PR to draft, comment the reason |
| Reviewer's hands review T-3 | check out `origin/task/T-3`, diff, run tests | same, from the PR's head; the summary becomes a review comment |
| Reviewer approves T-3 at the board | `approved_by` on the card | submit a PR review: **approve**, with the review summary |
| Reviewer leaves a changes note at the board | the note on the card reopens the work | submit a PR review: **request changes**, the note as its body |
| Owner moves T-3 to done at the board | `integrate`: merge into `main` | merge the PR (merge commit), delete the branch |

## Where it plugs in

`Workspace` already isolates "where the branch goes" in three methods: `publish`,
`checkout_review` and `integrate`. The integration is a `Remote` with two implementations:

- `LocalRemote`: today's bare repo on disk, the default and the test double.
- `GitHubRemote`: the same three operations, plus `open_pr`, `mark_draft`, `review` and
  `merge`, run from the host with the `gh` CLI.

The board events that GitHub mirrors (to review, blocked, approved, changes noted, done)
are observed the same way `ship_approved_work` observes "done" today: after the agent's
board visit, the agent's hands compare the board with what they last mirrored and act on
the difference. No board code calls GitHub.

**`gh` from the host, not GitHub MCP for the model.** The executor model never sees a token
and never calls GitHub: it only works in the checkout. The host runs a small, fixed set of
`gh` commands, each a direct mirror of a board event. MCP would give the model direct
GitHub access, which isn't needed and widens what can go wrong.

## Identities: two GitHub accounts

GitHub doesn't let an account approve its own pull request, and the approval has to come
from the reviewing teammate. So each agent needs its own identity:

- **Recommended:** two GitHub Apps (or two machine-user accounts), "Adam Beck (sim)" and
  "Eve Smith (sim)", each installed on the sandbox repo only, with *Contents: write* and
  *Pull requests: write*. Each agent's `gh` calls use its own token
  (`GH_TOKEN` per call, from `GA_GITHUB_TOKEN_ADAM` / `GA_GITHUB_TOKEN_EVE`).
- **Fallback (one token):** PRs and merges as one account, with the reviewer's verdict
  posted as a comment ("Approved by Eve Smith in review") instead of a GitHub approval.
  The board still gates the merge, so the rule holds, but GitHub's own review record
  doesn't show it.

## Safety

- **A sandbox repository only.** `GA_GITHUB_REPO=owner/name` must be on an allowlist in
  config, and never this project's own repository.
- **Off by default, then dry-run.** `GA_GITHUB=off` (default) uses `LocalRemote`.
  `GA_GITHUB=dry-run` logs every `gh` command it would run, without running it.
  `GA_GITHUB=on` runs them, only with the allowlisted repo and the tokens set.
- **No force pushes to `main`**, and branch protection on the sandbox `main`: changes only
  through a PR with one approval.
- **Tokens stay on the host:** they are never passed into the Docker sandbox or into a
  prompt.
- **A failed GitHub call never changes the board.** It is logged and remembered by the
  agent ("Eve Smith couldn't open the PR for T-3: ..."), and retried on the next board visit.

## Rollout

1. `Remote` interface, with `LocalRemote` as today's behavior (no change in what runs).
2. `GitHubRemote` in dry-run, with tests that check the `gh` command lines it would run.
3. A sandbox repo, two app identities and the allowlist, set up by the owner.
4. One supervised run with `GA_GITHUB=on`, read back on GitHub.
