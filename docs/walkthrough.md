# Zeyro Assignment Walkthrough

This repository implements the full assignment end-to-end with synthetic data, click prediction, personalized ranking, and production monitoring design.

## 1) Feature Engineering and EDA

### Dataset design
- Generated `500` users with realistic behavioral and demographic distributions.
- Included all required fields:
  - `age`, `city_tier`, `income_band`
  - `last_app_open_days_ago`
  - `onboarding_step_completed`
  - `linked_bank_account`
  - `sms_parsed_transactions_30d`
  - `past_nudges_shown`
  - `nudge_clicked` (target)
- Added `candidate_nudges` (5 candidates/user) for ranking.
- Injected missingness intentionally:
  - `income_band`: `8.6%`
  - `last_app_open_days_ago`: `5.4%`
  - `sms_parsed_transactions_30d`: `4.8%`

### Feature engineering
- Derived `past_nudges_unique_count` from `past_nudges_shown`.
- Handled missing values in modeling pipeline:
  - Numeric features -> median imputation
  - Categorical features -> most-frequent imputation + one-hot encoding
- Standardized numeric features before classification.

### Key EDA findings
- Overall click-through rate: `55.4%`.
- Click rate rises strongly with onboarding progress:
  - Step 1: `13.16%`
  - Step 5: `68.8%`
- Recency is highly predictive:
  - Opened app in `0-7d`: ~`66-67%` click
  - Opened app in `31-90d`: `15.79%` click
- Engagement depth matters:
  - `0-2` parsed transactions: `32.65%` click
  - `10-20` parsed transactions: `65.48%` click

### Surprising insight
The largest drop appears in high inactivity users (`31-90` days since last open), where click propensity collapses to `15.79%`. This is steeper than the tier/income effects, suggesting timing and reactivation state are much stronger levers than pure demographics.

## 2) Nudge Click Prediction Model

### Model choice
- Used `LogisticRegression(class_weight="balanced")` within a preprocessing pipeline.
- Rationale:
  - Fast, stable baseline for early-stage product
  - Interpretable coefficients for product decisions
  - Good fit for mixed tabular features and limited data size

### Evaluation setup
- Train/test split: `75/25`, stratified by label.
- Threshold tuning:
  - Selected threshold maximizing train F1 over `0.25..0.75`.
  - Chosen threshold: `0.34`.

### Test metrics
- Precision: `0.6500`
- Recall: `0.9420`
- F1: `0.7692`
- ROC-AUC: `0.7117`

### Metric priority
I optimize for **F1 with recall emphasis**. In this use case, missing users who would have engaged (false negatives) is expensive for growth. Slightly lower precision is acceptable early on, because ranking and throttling can control user experience.

### Most predictive signals (absolute coefficient)
1. `onboarding_step_completed` (+)
2. `last_app_open_days_ago` (-)
3. `city_tier=tier_3` (-)
4. `income_band=high` (+)
5. `city_tier=tier_1` (+)
6. `income_band=low` (-)
7. `linked_bank_account` (+)

## 3) Personalized Nudge Ranking

### Ranking strategy
Used a hybrid scorer:
- **Model score**: user-level click probability from classifier
- **Nudge prior**: heuristic prior CTR per nudge ID
- **Repeat penalty**: penalize nudges already shown to that user

For user `u` and candidate nudge `n`:
- Non-cold-start: `score = 0.65 * model_prob(u) + 0.35 * prior(n) - repeat_penalty`
- Cold-start user (no history): `score = 0.30 * model_prob(u) + 0.70 * prior(n)`

This balances personalization with pragmatic priors and avoids repeatedly surfacing stale nudges.

### Cold start handling
- Detect with `past_nudges_shown == []`.
- Shift weight from personalized score to nudge priors.
- Use demographic/behavioral defaults through model imputers for sparse user records.

## 4) Drift and Monitoring Strategy

### What to monitor in production
1. **Data quality**
   - Missing rate by feature
   - Null/invalid spikes by source
2. **Feature drift**
   - PSI for continuous features (`last_app_open_days_ago`, transactions)
   - Categorical distribution drift (`city_tier`, `income_band`)
3. **Prediction drift**
   - Mean score trend
   - Score histogram movement
4. **Outcome and business metrics**
   - CTR, conversion post-click, unsubscribe/negative signals
   - Precision/recall proxy via delayed labels
5. **Segment fairness checks**
   - CTR and score parity by city tier / income band

### Simple drift detection sketch
- Keep 30-day rolling baseline windows from training-serving logs.
- Daily job computes:
  - PSI on top numeric features
  - Jensen-Shannon divergence on categorical distributions
  - Alert if:
    - PSI > `0.2` on any critical feature for 3 consecutive days, or
    - JS divergence > threshold + CTR drops > 10% WoW
- If alert triggers:
  1. Route to monitoring channel
  2. Shadow-train on latest 60-90 days
  3. Compare offline metrics and calibration
  4. If improved, deploy behind canary (10% traffic), then full rollout

## Bonus Direction (not implemented)

### Contextual bandits framing
- Arm = nudge ID, context = user features, reward = click/conversion.
- Start with LinUCB/Thompson Sampling to explore/exploit online.
- Benefit: continual policy learning instead of periodic retraining only.

### Account Aggregator (AA) integration
- Add cashflow stability, obligations, salary periodicity, and spend category volatility as features.
- Use AA-derived financial health signals to:
  - improve user intent prediction,
  - map nudges to immediate financial context,
  - build stronger cold-start personalization from inferred behavior.
