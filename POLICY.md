# POLICY — Reconia (ĐA161)

Team: Phí Vương Tường Tâm (52300155) · Vũ Hưng Thịnh (52300159)
Project: Invoice and Inventory Document Reconciliation System

---

## 1. Branches

| Branch | Purpose |
|---|---|
| `main` | Submission builds only. Merged twice: midterm and final. **Never commit directly.** |
| `dev` | Integration branch. **Only ever receives merged PRs. Never commit directly.** |
| `<type>/<scope>-<description>` | Where all work happens. One branch per piece of work. |

Branch naming uses the same types and scopes as commit messages:

```
feat/extraction-xml-parser
feat/matching-tier0-reference
fix/rules-zero-quantity
docs/policy-pr-workflow
chore/infra-docker-compose
```

`main` being behind `dev` is normal, not a problem to fix.

## 2. Commit messages

Conventional Commits:

```
<type>(<scope>): <description>
```

- `<type>` — `feat` `fix` `docs` `style` `refactor` `test` `chore`
- `<scope>` — optional, from the area table in section 3. Omit for cross-cutting changes.
- `<description>` — imperative mood, lowercase first letter, no trailing period, under ~50 chars

```
feat(extraction): add XML e-invoice parser
feat(matching): add tier-0 explicit reference lookup
fix(rules): handle zero quantity in line arithmetic
refactor(db): add direction column for inbound and outbound
docs: add team policy
```

Commit at meaningful units of work, not per file. Each commit should build and run.

### Attribution

Commit messages carry no name — **attribution comes from the git author.** Each person keeps
one fixed name + email on every machine they work from; `git shortlog -sn` groups by name +
email, so using two emails makes one person appear as two contributors. The contribution table
in the report must match this output:

```
git shortlog -sn
```

## 3. Areas

| Scope | Maps to |
|---|---|
| `intake` | Module 1 — ingestion, classification, preprocessing |
| `extraction` | Module 2 — extraction, normalization |
| `matching` | Module 3 — reconciliation, pairing |
| `rules` | Module 4 — constraints, anomaly detection |
| `agent` | Module 5 — agent orchestration, human review |
| `audit` | Module 6 — audit log, correction memory |
| `ui` | Screens, review UI, reports |
| `db` | Schema, migrations |
| `infra` | Docker, configuration |
| `eval` | Test data, measurement, model comparison |
| `docs` | Documentation, report |

Source directories follow this same list.

## 4. Workflow

**Nothing is ever committed straight to `dev`.** Every piece of work — a feature, a fix, even a
docs change — starts as a branch and reaches `dev` through a reviewed pull request.

1. Branch off the latest `dev`:
   ```
   git checkout dev && git pull
   git checkout -b feat/extraction-xml-parser
   ```
2. Work. Commit in meaningful units. **Push at the end of every working day**, including
   unfinished work. This protects against local machine failure, and `git log --since="1 week ago"`
   provides the basis for the weekly progress report.
3. When done (see section 6), open a PR into `dev`:
   ```
   gh pr create --base dev --title "feat(extraction): add XML e-invoice parser" --assignee @me
   ```
4. **Assign the other person as reviewer.** They read it and either approve or request changes.
5. Merge only after approval. Delete the branch afterwards.

### Rules

- **Never merge your own PR without the other person approving it.** The review is the point —
  it is also how each of you can answer questions about the other's code at the defense.
- **Never** `git push --force` to `main` or `dev`.
- `main` is merged **twice only** — midterm submission and final submission, both agreeing.
- A PR that sits unreviewed for more than a day: ping them. Do not merge around it.

> Both members need write access to `github.com/tamir39/reconia` for this to work.
> Add Vũ Hưng Thịnh as a collaborator before the first PR.

## 5. Code conventions

- **Money is always `Decimal`, never `float`.** Database columns use `NUMERIC`.
  Floating point rounding produces differences the reconciliation logic would report as genuine.
- Tax code (MST) is the party identity key. **Never match on address.**
- Never run OCR on a document that already has XML or a usable text layer.
- Table and column names: `snake_case`, English, singular (`invoice`, `stock_document`,
  `product_alias`).
- Code and comments in English. User-facing text in Vietnamese.
- Each validation rule is **one pure function** writing into the `finding` table.

## 6. Definition of done

1. Runs on the author's machine without errors.
2. At least one test for any logic (rules, normalization, matching).
3. Committed in the correct format and pushed to the branch.
4. **PR opened into `dev`, reviewed and approved by the other person, and merged.**
5. No leftover `TODO`s, dead code, or empty functions.

Anything short of all five is still **in progress** and does not count toward the
completion percentage in the plan sheet.

## 7. Never commit

| Never commit | Why |
|---|---|
| **Real** invoices or stock documents before anonymization | Contains MST, prices, counterparty names |
| `.env`, API keys, tokens (including the Gemini key) | A leaked key must be rotated immediately |
| OCR model weights, large binaries | Bloats the repo, re-downloadable |

Real data stays **outside the repo**. Only synthetic data and the data generator get committed.
Always run `git status` and read the diff before committing.
