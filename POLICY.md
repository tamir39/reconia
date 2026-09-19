# POLICY — Reconia (ĐA161)

Team: Phí Vương Tường Tâm (52300155) · Vũ Hưng Thịnh (52300159)
Project: Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction
(Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin từ hóa đơn)

---

## 1. Branches

| Branch | Purpose |
|---|---|
| `main` | Submission builds only. Merged twice: midterm and final. **Never commit directly.** |
| `dev` | Integration branch. **Only ever receives merged PRs. Never commit directly.** |
| `<type>/<scope>-<description>` | Where all work happens. One branch per piece of work. |

Branch naming uses the same types and scopes as commit messages:

```
feat/data-cord-adapter
feat/graph-spatial-edges
feat/model-gnn-rgcn
feat/train-pipeline
feat/eval-docile-metrics
docs/update-spec
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
feat(data): add CORD-v2 dataset adapter
feat(graph): compute spatial edge relation features
feat(model): implement Relational GCN for KILE
feat(train): add learning rate scheduler with warmup
fix(data): resolve SROIE multi-line bounding box bleeding
test(graph): add unit test for undirected edge conversion
docs: update benchmark specification
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
| `data` | Dataset adapters (DocILE, SROIE, CORD-v2), OCR loaders, normalization, paths |
| `graph` | Document graph builders, nodes, spatial edges, feature extraction |
| `model` | Deep Learning models (GNN, RoBERTa, LayoutLMv3, LiLT) |
| `train` | Training loop, loss functions, optimizers, checkpointing |
| `eval` | Unified evaluation framework (KILE/LIR metrics, entity-level P/R/F1) |
| `demo` | Legacy ERP reconciliation & verification demo (`app/ui/`, `app/models/`, `app/matching/`) |
| `db` | Database schema, migrations (demo layer) |
| `infra` | Docker, environment, dependencies, GPU configuration |
| `docs` | Documentation, specs, progress reports |

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

### Deep Learning & Benchmark Rules
- **Reproducibility first:** All training and splitting scripts must explicitly fix random seeds (default `42`).
- **Portability:** Never hardcode absolute file paths (e.g. `D:/...`). Always resolve dataset directories dynamically using `app.data.paths` or environment variables (`DOCILE_DIR`, `SROIE_DIR`, `CORD_DIR`).
- **Device independence:** PyTorch tensors and models must support agnostic device dispatch via `.to(device)` (`cuda` if available, otherwise `cpu`).
- **Strictly unified evaluation:** Metrics (Precision, Recall, F1) must be computed using identical evaluation logic and format matching the DocILE benchmark standards (evaluating at the entity level with exact/overlap matching).
- **Documentation:** Code and comments in English. Technical documentation and research reports in Vietnamese.

### Legacy ERP Demo Rules
- **Money is always `Decimal`, never `float`.** Database columns use `NUMERIC`.
- Tax code (MST) is the party identity key. **Never match on address.**
- Table and column names: `snake_case`, English, singular (`invoice`, `stock_document`, `product_alias`).
- Each validation rule is **one pure function** writing into the `finding` table.

## 6. Definition of done

1. Runs on the author's machine without errors (or on Kaggle for GPU Transformer training).
2. All unit tests pass (`pytest tests/`).
3. Committed in conventional commit format (`<type>(<scope>): <description>`).
4. **PR opened into `dev`, reviewed and approved by the peer reviewer, and merged.**
5. No leftover `TODO`s, dead code, or hardcoded personal machine paths.

Anything short of all five is still **in progress** and does not count toward the
completion percentage in the plan sheet.

## 7. Never commit

| Never commit | Why |
|---|---|
| Large raw datasets (`docile/`, `sroie/`, `cord/`) | Very large size; datasets are managed externally |
| Model weights (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, checkpoints) | Bloats git history; weights belong on Kaggle/HuggingFace |
| **Real** invoices or stock documents before anonymization | Contains MST, prices, counterparty confidential data |
| `.env`, API keys, tokens (e.g. Kaggle keys, Gemini keys) | Security violation; leaked keys must be rotated immediately |

Raw datasets and trained weights stay **outside the git repo**.
Always run `git status` and read the diff before committing.
