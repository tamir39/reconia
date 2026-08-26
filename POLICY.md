# POLICY — Reconia (ĐA161)

Team: Phí Vương Tường Tâm (52300155) · Vũ Hưng Thịnh (52300159)
Project: Invoice and Inventory Document Reconciliation System

---

## 1. Branches

| Branch | Purpose |
|---|---|
| `main` | Submission builds only. Merged twice: midterm and final. Never commit directly. |
| `dev` | Main working branch. Everything lands here. |
| `feature/<name>/<area>` | Only for large or risky work. Small changes go straight to `dev`. |

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

There is no name in the commit message — **attribution comes entirely from the git author.**
That makes the identity setup below mandatory, not optional: the contribution table in the
report must match `git shortlog -sn`, and GVPB questions each member individually.

> **Do this on both machines now** (cannot be fixed retroactively):
> ```
> git config user.name "Phi Vuong Tuong Tam"
> git config user.email "<your email>"
> ```
> Any email works — the faculty does not require a specific one. What matters is that
> **each person uses one fixed name and email on every machine**, because `git shortlog -sn`
> groups by name + email. Two different emails make one person show up as two rows.

Check it any time with:
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

## 4. Push & merge

- Push your branch **at the end of every working day**, even if unfinished. A dead laptop
  loses everything, and `git log --since="1 week ago"` is basically your weekly report to thầy.
- **Never** `git push --force` to `main` or `dev`.
- Merging into `dev`: **message the other person first.** No PR needed, but the message is
  mandatory — you will both touch the same files.
- Merging into `main`: **twice only** — midterm submission and final submission, both agreeing.

## 5. Code conventions

- **Money is always `Decimal`, never `float`.** Database columns use `NUMERIC`.
  Get this wrong and every reconciliation result becomes untrustworthy.
- Tax code (MST) is the party identity key. **Never match on address.**
- Never run OCR on a document that already has XML or a usable text layer.
- Table and column names: `snake_case`, English, singular (`invoice`, `stock_document`,
  `product_alias`).
- Code and comments in English. User-facing text in Vietnamese.
- Each validation rule is **one pure function** writing into the `finding` table.

## 6. Definition of done

1. Runs on the author's machine without errors.
2. At least one test for any logic (rules, normalization, matching).
3. Committed in the correct format and pushed to remote.
4. The other person has been told.
5. No leftover `TODO`s, dead code, or empty functions.

Anything short of all five is still **in progress** and does not count toward the
completion percentage in the plan sheet.

## 7. Never commit

| Never commit | Why |
|---|---|
| **Real** invoices or stock documents before anonymization | Contains MST, prices, counterparty names |
| `.env`, API keys, tokens (including the Gemini key) | A leaked key must be rotated immediately |
| OCR model weights, large binaries | Bloats the repo, re-downloadable |
| Faculty `.xlsx` plan and student roster files | Personal data; one is marked *lưu hành nội bộ* |
| `CLAUDE.md`, `QUY_DINH_DU_AN.md` | Personal notes, local only |

Real data stays **outside the repo**. Only synthetic data and the data generator get committed.
Always run `git status` and read the diff before committing.

## 8. Weekly rhythm

1. During the week: small frequent commits, push at end of day.
2. End of week: merge into `dev`, verify the system still runs.
3. **Send the weekly report email to thầy** — subject
   `[BC-DACNTT/KLTN/DATN-261] - 52300155 - 52300159`, replying into the existing thread,
   never a new email.
   Two consecutive weeks without a report and thầy may stop supervising the group.
4. Every two weeks: Google Meet, Saturday afternoon after 13h30.
