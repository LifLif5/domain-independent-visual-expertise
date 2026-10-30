# SHARING — archiving the code and data for the Nature Communications submission

**Bottom line:** Nature Communications does *not* require Code Ocean. Its
reporting standard only asks that custom code be (i) shared with editors and
reviewers during peer review, (ii) described in a *Code availability*
statement, and (iii) **deposited in a DOI-minting repository such as Zenodo or
Code Ocean** after acceptance, with an OSI-approved license. Source:
<https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards>
(§ "Availability of computer code"). Code Ocean is described there as an
*optional free service*, so a GitHub repository + Zenodo DOI satisfies the
policy on its own. A Code Ocean capsule can still be added later (revision or
post-publication) — see §7.

---

## 1. What to do now, in order (≈1–2 hours of work)

| # | Step | Where |
|---|------|-------|
| 1 | Push this repository to **GitHub** (private while under review) | github.com |
| 2 | Deposit the **data** (5 files, 120 MB) in Zenodo/figshare → **data DOI** | zenodo.org |
| 3 | Fill in the placeholders in `CITATION.cff` and `.zenodo.json` (authors, ORCID, data DOI, repo URL), commit | this repo |
| 4 | Create a **GitHub release `v1.0.0`** with the Zenodo switch on → **code DOI** | zenodo.org ⇄ GitHub |
| 5 | Grant the editor/reviewers access (private repo invite or Zenodo share link) | github / zenodo |
| 6 | Paste the *Code availability* / *Data availability* statements (§5) into the manuscript | — |
| 7 | Mention the reviewer guide in the cover letter (§6) | — |

Steps 4–6 are what the journal actually checks; steps 1–3 are prerequisites.

## 2. Zenodo (recommended DOI route)

* Free, CERN-operated, DOI per version ("concept DOI" for all versions),
  any file size/format, accepted by Nature Portfolio (named in the policy).
* **Metadata comes from this repo**: with `.zenodo.json` present Zenodo uses it
  for the release archive (and *ignores* `CITATION.cff`; keep both — GitHub
  shows `CITATION.cff` as a "Cite this repository" button).
* Setup, once: zenodo.org → *Settings → GitHub* → flip the switch for the repo.
  Then *GitHub → Releases → Draft a new release*, tag `v1.0.0` → Zenodo archives
  that snapshot and mints a DOI. Later revisions: release `v1.0.1` → new DOI,
  same concept DOI.
* The archived snapshot contains **code only** (no data — 120 MB would exceed
  GitHub's 100 MB per-file limit anyway), which is why the data need their own
  record; link the two via `related_identifiers` in `.zenodo.json`.
* Data record: *New upload* → upload the five files → `upload_type: dataset`,
  same creators, license CC BY 4.0 (or CC0), and link the manuscript. Add
  `movement_phase_full.pkl` **plus** a preservation-friendly copy
  (`movement_phase_full.parquet`, ~10–20 MB, produced with
  `pd.read_pickle(...).to_parquet(...)`) — Zenodo only guarantees bit-level
  preservation and pickles are Python-version sensitive.

### Alternatives to Zenodo (all policy-compliant)

| Repository | DOI | Notes |
|---|---|---|
| **figshare** | yes | same role, good UI, 20 GB free |
| **OSF** | yes | one project can hold code + data + manuscript, private during review with a view-only link |
| **Software Heritage** | SWHID | automatic permanent archive of every GitHub push; a *complement*, not a substitute |
| **anonymous.4open.science** | no | anonymises a GitHub repo for double-blind review; do not use if the journal is single-blind |

## 3. Reviewer access during peer review

The policy requires code to be available to editors/reviewers *at submission*;
it does not have to be public yet. Pick one:

1. **Private GitHub repo** + invite reviewer GitHub accounts (needs their
   usernames via the editor) — simplest, but editors often prefer a plain link.
2. **Zenodo restricted record + share link**: publish the record with
   `access_right: restricted`, then *Share → Link* and paste the secret link
   into the Code availability statement. Works for anyone without an account
   and is swapped to public at acceptance (files cannot change after
   publishing — that is the point of a DOI).
3. **Public from day one** (most common for Nature Portfolio submissions, and
   what the statement in §5 assumes). Simplest and safest: nothing to revoke or
   forget at acceptance.

Note: Nature Communications uses single-blind review, so an anonymised mirror
is normally unnecessary.

## 4. Does the reviewer need compute?

No. The full pipeline runs in ≈25 min on a laptop CPU with 8 GB RAM (measured;
see `REVIEWER_GUIDE.md` §4), so no Code Ocean-style compute grant is needed for
peer review. `REVIEWER_GUIDE.md` gives the reviewer a conda **and** a Docker
route, the expected outputs and a table of numbers to compare against.

Optional but nice: if you want a one-command `docker run` image without a local
Docker install, `.github/workflows/docker-publish.yml` builds
`environment/Dockerfile` on GitHub's runners and pushes it to GitHub Container
Registry (manual trigger; untested here — run it once and check the log before
citing the image).

## 5. Statements to paste into the manuscript

Insert **Code availability** directly after **Data availability** and before the
references (that ordering is required by the policy). Fill in the DOIs.

**Code availability (public from submission — recommended)**

> The analysis code that supports the findings of this study is available at
> https://github.com/&lt;user&gt;/EyeTrackingAnalysis and is archived at
> https://doi.org/&lt;code DOI&gt; (release v1.0.0, MIT license). It contains the
> feature extraction, all statistical models (linear mixed models fitted with
> pymer4/lme4) and every figure of the manuscript, together with the pinned
> software environment and a single entry point (`run`) that regenerates all
> results from the deposited data in ≈25 min on a standard CPU. A step-by-step
> guide for reviewers is included (`REVIEWER_GUIDE.md`).

**Code availability (restricted during review — alternative first sentence)**

> The analysis code is available to editors and reviewers during peer review at
> &lt;private GitHub link / Zenodo share link&gt; and will be made publicly
> available at https://doi.org/&lt;code DOI&gt; upon publication. It contains …

**Data availability (draft — adapt to your raw-data situation)**

> The processed data generated in this study have been deposited in Zenodo at
> https://doi.org/&lt;data DOI&gt; and comprise the visual-search trial-level gaze
> and derived feature table (`simple_df_search.csv`), the search accuracy and
> reaction-time table (`search_performance_long.csv`), the MOT trial-level gaze
> samples (`simple_df_mot.csv`), the MOT performance table
> (`mot_long_dataframe.csv`) and the raw MOT gaze trajectories
> (`movement_phase_full.pkl` / `.parquet`). Raw eye-tracking recordings and the
> demographics that could allow re-identification of participants are available
> from the corresponding author on reasonable request, subject to the ethics
> approval under which they were collected.

**Reference-list entry** (the policy asks for code to be *cited*, not just
linked — add it as a numbered reference):

> &lt;Surname, Initials&gt;. EyeTrackingAnalysis: analysis code for visual-search
> and multiple-object-tracking eye-tracking experiments (v1.0.0). Zenodo
> https://doi.org/&lt;code DOI&gt; (2026).

## 6. Cover letter / submission-system notes

* "Code and data availability": point at the Code availability statement and
  state that no proprietary software is needed (Python + R only, both open
  source, installed from conda-forge).
* If the submission system has a "reproducibility / code repository" field,
  give the GitHub URL; put the Zenodo DOI in the statement (the DOI is the
  citable, permanent one).
* Do **not** promise a Code Ocean capsule unless you will actually build one —
  a capsule is optional and reviewers will not be given compute time by the
  journal anyway.

## 7. If you still want a Code Ocean capsule (later)

* Capsules can be created after publication from the same repository
  (*New Capsule → Clone from Git*), so nothing in this plan has to be undone.
  `CAPSULE_SETUP.md` §2 keeps the full recipe; `environment/Dockerfile` and the
  root `run` are exactly what a capsule expects (`code/` → `/code`, `run` →
  Reproducible Run entry point).
* When a capsule exists, add a second sentence to the Code availability
  statement ("… and can be run without installation as a Code Ocean capsule at
  https://doi.org/&lt;capsule DOI&gt;") and cite it as an additional reference.
* Versioning: keep the GitHub release (`v1.0.0`) fixed for the published paper;
  do revision work on `main` and mint `v1.0.1` only if the analysis changes.

## 8. Placeholders to fill before pushing

Search for `<` in these files and replace:

```bash
grep -n '<' CITATION.cff .zenodo.json | grep -v orcid.org
```

* `.zenodo.json` — creator name/affiliation/ORCID, contact person, data DOI,
  manuscript DOI.
* `CITATION.cff` — same, plus repository URL and date-released.
* `LICENSE` — copyright holder line (currently "The authors of
  EyeTrackingAnalysis"); MIT is OSI-approved and is what the policy asks for.
  Swap for `Apache-2.0`/`BSD-3-Clause`/`GPL-3.0` if your institution requires.
