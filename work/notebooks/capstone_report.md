# Capstone Report — Freestyle: Content Decay Risk Prediction

**Author:** Gaurav Navghare

**Lane:** Freestyle — Content Decay / Decline Risk Prediction

**Repo:** [github.com/gaurav443201/flyrank-ml-internship-work/](https://github.com/gaurav443201/flyrank-ml-internship-work/)

**Date:** 25th August 2026

---

## 0. Abstract

A page rarely collapses overnight. Before its clicks fall, there are usually smaller tells first — a rank that has started drifting, a click-through rate that's quietly slipping below what its position should earn, a position that has gotten noisier week to week even if the average hasn't moved much. This project asks whether those tells, read from a short lookback window, can flag a page as being at risk of decline before the drop shows up in the traffic report.

Using the FlyRank ML Internship warehouse, I engineered lookback-window features for each page — how its position is trending and how unstable that position has been, how impressions are moving, how far its click-through rate sits from what a page in that position would normally get — and trained a gradient-boosted classifier to predict decline in the following window. I validated it against a naive-persistence baseline ("if it was already falling, guess it keeps falling") on a split that is grouped by page and separated in time, so nothing about the future leaks into training.

The model reached an **AUC of 0.712** against a baseline of 0.512 (base rate 0.386), and its output became a ranked, reason-coded shortlist an editor could use to decide which pages to rescue first. The claim is deliberately modest: this is a triage tool for where to look first, not an explanation of why any single page will decline.

---

## 1. Problem framing

The unit of analysis is a page, observed at a given cutoff date — the same page can appear more than once, at different points in time, as a separate "moment" the model learns from.

The output is a **decay-risk score** between 0 and 1 — the estimated chance a page's clicks are about to fall — paired with a small set of plain-English reason codes explaining why a page scored the way it did.

The action this supports: every week, an editor opens a ranked list and decides which pages need a rescue pass right now — a title/snippet rewrite, a fresh internal link, an updated statistic or date — while there's still time to arrest the slide, instead of noticing the drop only after the page has already lost most of its traffic.

The two failure modes aren't equally costly:

- A **false positive** sends an editor to refresh a page that was never really at risk — a wasted afternoon, but a cheap one.
- A **false negative** lets a genuinely declining page slip through unflagged, and by the time anyone notices in a monthly report, the traffic (and the revenue or leads behind it) is already gone.

That asymmetry is why this is built as a ranked shortlist for triage rather than a strict alarm — the riskiest pages surface first, and nothing is silently dropped.

Machine learning earns its place here because decay isn't announced by one signal — it's a combination of a softening rank, a CTR that's started underperforming its position, and a position that's grown more volatile, and those only add up to real risk when read together, across thousands of pages, every week. That's a pattern worth weighing consistently at scale, not something an editor can reliably eyeball in a spreadsheet.

---

## 2. Data safety

This project uses the FlyRank ML Internship warehouse's `fact_content_daily_performance` table — daily `report_date`, `gsc_clicks`, `gsc_impressions`, and `gsc_avg_position` — together with `client_hash_id` and `content_hash_id` as pseudonymous identifiers.

- **Release covers:** 2025-01-27 to 2026-06-30
- **Lookback window (features):** 2025-09-01 to 2025-10-31
- **Label window (decline outcome):** 2025-11-14 to 2026-01-09
- **Gap between the two:** 14 days

Some things were left out on purpose, not by accident:

- No client name, domain, or raw URL ever enters the pipeline. `client_hash_id` and `content_hash_id` are combined into a single pseudonymous page identifier (a content hash alone isn't guaranteed unique across different clients), and that identifier is used only to group the train/test split — never as a model feature.
- The pre-built `trend_direction` and `trend_pct` fields were deliberately not used, for the same reason as any label-derived, window-ambiguous column: it's too easy for it to quietly leak the answer into the question. The decline label is rebuilt from raw click counts, under a cutoff date I control myself.
- No query text and no query-level rows appear anywhere — the analysis stays at the page level, so there's no search query to exclude in the first place.

Three leakage risks were checked, not just assumed away:

1. **Page overlap across splits** — an explicit assertion confirms no page ID appears in both the training set and the test set.
2. **A curve fit on data it later gets tested against** — the expected-CTR-by-position curve is fit only on training pages, never on the full dataset.
3. **A label window that isn't really in the future** — a 14-day gap sits between the end of the lookback window and the start of the label window, wider than the reporting lag in the source data, so there's no ambiguous overlap.

Nothing client-identifying appears anywhere in `work/` — only pseudonymous page IDs and aggregated numbers.

---

## 3. Baseline

The baseline is **naive persistence**: predict decline if the page's clicks were already falling within the lookback window itself.

It's a fair comparison precisely because it's unambitious — it uses only one signal the model also has access to, and applies no learning at all. Any lift the model shows over this baseline has to come from genuinely combining position trend, position stability, impressions, and CTR — not from reading the same click trend a simpler rule already reads.

**Baseline AUC on the held-out split: 0.512** — essentially a coin flip, which is what a single noisy signal should look like when clicks are this volatile week to week.

---

## 4. Model / analysis

I used a gradient-boosted tree classifier. Decay felt like a non-linear, combined story — a softening position only really signals risk if the CTR is also starting to underperform, and a stable-looking average position can still be masking real volatility underneath — and boosted trees pick up that kind of interaction better than a linear model, while staying small enough to inspect afterwards.

**Features:**

| Feature | Description |
|---|---|
| `position_slope` | Linear trend of average position across the lookback window |
| `position_volatility` | Standard deviation of daily position within the lookback window — captures rank instability a slope alone would miss |
| `impressions_pct_change` | First half of the window versus the second half |
| `ctr_vs_expected` | Observed CTR minus what a page at that position would typically get, with the expectation curve fit only on training pages |
| `obs_count` | How many observations the trend is actually built from, so a two-point "trend" doesn't get treated the same as a full window's worth |

Left out on purpose: raw click and impression counts, which would let the model shortcut to "small pages look risky" instead of learning decay, and the pre-computed trend fields, for the leakage reasons already covered in Section 2.

**Label definition:** a page is labeled `decay = 1` if its clicks in the label window fall by at least **34.6%** relative to the lookback window (the 40th percentile of all pages' percent-change, read off the real data rather than picked out of thin air). That threshold landed the dataset at a base rate of **38.6%** decaying pages.

---

## 5. Evaluation

The split is both **grouped by page ID**, so no page appears on both sides, and **time-aware**, so the test set's label window comes strictly after anything the model trained on. A grouped split alone wouldn't have been enough — pages could still share calendar time across train and test — so both constraints apply together.

After filtering to pages with at least 3 observations in the lookback window, the split produced **24,918 training rows** and **8,307 test rows**.

| | Base rate | AUC | Average precision |
|---|---|---|---|
| Baseline (naive persistence) | 0.386 | 0.512 | — |
| Model (gradient boosting) | 0.386 | **0.712** | 0.601 |

The model beats the baseline by 0.200 AUC points, and clears the 0.386 base rate by a wide enough margin that the discrimination is real, not just a low base rate dressed up as a score.

**Error analysis:** I pulled out the false negatives — pages that did decline but scored low — and looked at their features instead of just counting them: **531 of the 8,307 test rows (6.4%)** fell into this bucket. Going in, I expected these misses to cluster around a thin lookback history. They didn't: the median `obs_count` among the false negatives was **58**, close to the maximum possible, meaning most of the pages the model missed actually had a full history to learn from.

What they lacked instead was a visible warning sign: the median `position_slope` among misses was **-0.008** (essentially flat to slightly improving) and the median `impressions_pct_change` was **+0.062** (slightly rising). In plain terms, the model's blind spot isn't noisy data — it's pages whose clicks fell despite a stable or even improving position and impression trend, which points to something the four features can't see directly, such as a shift in what people click on once they reach the results page. That's a real and honest limit, not a bug.

---

## 6. Interpretation

Looking at the ranked shortlist rather than just the raw feature-importance numbers tells the same story: almost every page that made the top 25 carried the reason codes `ctr_below_expected` and `position_volatile` together. That pairing — a click-through rate already underperforming its position band, combined with a position that's been bouncing around more than usual — is what the model leaned on most heavily to call a page at risk. `position_slope` and `impressions_pct_change` mattered far less on their own; a page didn't need a visibly worsening rank to score high, it needed to already be earning fewer clicks than its position should support.

The honest negative result sits in Section 5's error analysis: decay risk built from position and impression trends alone missed a meaningful slice of pages that fell anyway — a well-understood gap rather than a hidden one, and a fair place for a future iteration (perhaps a feature that tracks CTR's own volatility, not just its distance from expectation) to start.

---

## 7. Recommendation

The model's output becomes a ranked shortlist, with reason codes an editor can act on without ever opening the notebook:

| Reason code | Fires when | What it means to do |
|---|---|---|
| `position_worsening` | `position_slope` is in the top decile (rank slipping) | Reinforce now — internal links, a freshness pass — before the slide continues |
| `ctr_below_expected` | CTR falls below what its position band would predict | Check for a lost SERP feature or a stale title/snippet; rewrite the meta description |
| `position_volatile` | `position_volatility` is unusually high for the window | Keep watching — this can be ordinary SERP noise, not real decay |
| `thin_history` | `obs_count` is near the minimum | Not enough history to trust the score yet — recheck next window |

In practice, the top of the ranked list is overwhelmingly `ctr_below_expected` combined with `position_volatile` — pages already earning fewer clicks than their rank would predict, on a position that's been unsteady. Those are the ones worth an editor's attention first.

The confidence level here is directional and decision-support, not a guarantee. This list tells an editor where to look first. It makes no claim about why a search engine ranks pages the way it does, and it isn't a promise that any individual page will decline — Section 5's error analysis is proof of that on its own.

---

## 8. Reproducibility

To rebuild everything from a fresh clone:

```bash
git clone https://github.com/gaurav443201/flyrank-ml-internship-work/.git
cd flyrank-ml-internship-work
pip install -r requirements.txt
jupyter notebook work/notebooks/capstone.ipynb
```

The random seed is fixed at **42**, set once and reused for both the train/test split and the model fit. DuckDB version used: **1.3.2**.

The cell that builds the held-out split (a grouped shuffle split, notebook cell `3c`) and the cell that produces the results table (notebook cell `4b`) are both committed exactly as run, inside `work/notebooks/capstone.ipynb`. The claim that this was evaluated once, blind, is something anyone can check by re-running the notebook top to bottom — it doesn't rest on trust.

---

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
