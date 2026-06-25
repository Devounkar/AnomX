# AnomX — Synthetic Forex Behaviour & Anomaly Detection (Midterm Submission)

> This README summarizes everything learned and implemented from Week 1 through Week 4 of the AnomX project: Python/Git foundations, synthetic event generation, and feature engineering for anomaly detection on simulated forex trading behaviour.

---

## Table of Contents

1. [Week 1–2: Python, Git & Foundations](#week-1–2-python-git--foundations)
2. [Week 3: Event Generation](#week-3-event-generation)
3. [Week 4: Feature Engineering & EDA](#week-4-feature-engineering--eda)
4. [Files Submitted in This Repository](#files-submitted-in-this-repository)

---

## Week 1–2: Python, Git & Foundations

### Python basics learned
Over these two weeks I worked through core Python building blocks that the rest of the project depends on: functions, classes, list/dict comprehensions, working with `datetime` objects, random number generation with controllable seeds (for reproducible synthetic data), and writing modular code split across multiple files instead of one long script.

### Git & GitHub workflow
I set up a local Git repository, learned the `add → commit → push` cycle, wrote meaningful commit messages tied to specific milestones (e.g. "add event generator", "add feature engineering script"), and used branches to keep experimental work separate from the working version before merging back to `main`.

### Pandas and NumPy basics
I learned to:
- Build and manipulate `DataFrame`s (filtering, grouping, merging, `groupby().agg()`)
- Use `NumPy` for vectorized numerical operations instead of slow Python loops (especially useful for generating large volumes of synthetic events quickly)
- Handle missing values and type conversions
- Export and re-import tabular data in different formats (CSV and Parquet)

### Understanding anomaly detection
Anomaly detection is the task of identifying data points (here, *users*) whose behaviour deviates significantly from what is normal for the rest of the population — without necessarily having labelled examples of every kind of anomaly in advance. In a financial context, this means picking out behaviour like sudden bursts of trading activity, unusual login patterns, or deposit-then-withdraw cycles that don't match typical trading activity. The key challenge is that "normal" itself has to be learned from the data, since fraud/abuse patterns are rare and varied.

### Project overview and objectives
The goal of this project is to simulate realistic forex trading and account activity (logins, trades, deposits, withdrawals) for many synthetic users, inject a known subset of anomalous behaviour patterns into that simulation, and then build a feature set that can later be used to detect those anomalies. The midterm phase covers everything up through generating the data and engineering features from it — model training and evaluation come in later phases of the project.

### Documentation practices
Each script is documented with docstrings explaining what it does, what parameters it takes, and what it outputs. Configuration values (number of users, number of events, anomaly injection rates) are kept in one place rather than hardcoded throughout the code, so the whole pipeline can be re-run with different settings without editing multiple files.

---

## Week 3: Event Generation

### How synthetic event data was generated
A data generator script creates a population of synthetic users and simulates their activity over a fixed observation window. Most users behave "normally" — a baseline rate of logins, trades, deposits, and withdrawals drawn from realistic distributions. A smaller, labelled subset of users have one or more anomalous patterns deliberately injected into their event stream (e.g. abnormally high trade frequency in a short window, logins from many different regions in quick succession, or a deposit followed shortly by a withdrawal with little trading activity in between). This gives the dataset internal ground truth that can be used later for evaluating detection methods, while keeping the *features* themselves free of any direct label leakage.

### Types of events used
- **Logins** — timestamp, IP address, region
- **Trades** — timestamp, currency pair, trade size, leverage, PnL
- **Deposits** — timestamp, amount
- **Withdrawals** — timestamp, amount
- **Account/profile changes** — e.g. updates to bank details, contact info, or leverage settings

### Dataset structure and key insights
The generated event data is written as two separate files — `portal_events.parquet` (logins, account/profile changes, support activity) and `trade_events.parquet` (trades, deposits, withdrawals) — each with one row per event, identifying the user, event type, timestamp, and event-specific details (e.g. trade size for a trade event, amount for a deposit/withdrawal). A separate small reference file (`user_labels.parquet`) keeps track of which users were given injected anomalous behaviour, purely for later evaluation — this is *not* used as a feature.

Key observations from generating and inspecting the data:
- Event volume per user varies naturally even among "normal" users, which means raw counts on their own are a weak signal — they need to be normalized by observation time.
- Anomalous patterns are easier to simulate convincingly when they're injected as *relative* deviations from a user's own baseline rather than fixed absolute thresholds, since real anomalous behaviour also looks different person to person.

**A note on file format:** the event and feature datasets are submitted as **`.parquet`** files (`portal_events.parquet`, `trade_events.parquet`, `features.parquet`) rather than `.csv`. Parquet was chosen deliberately over plain CSV because:
- It preserves column data types (timestamps stay as timestamps, floats stay as floats) instead of flattening everything to text, which avoids re-parsing/casting errors on reload.
- It is a columnar, compressed format, so both file size and read/write time are significantly smaller than CSV once the event count grows into the tens of thousands — relevant since this generator produces a large volume of events.
- It's the standard format used by `pandas`/`pyarrow` pipelines in practice, so working with it now is also good preparation for the modelling stages later in the project.

The trade-off is that Parquet files aren't human-readable by opening them in a text editor the way a CSV is — but they open instantly with `pd.read_parquet()`, which is how the rest of the pipeline (and the grader, if needed) is expected to read them.

---

## Week 4: Feature Engineering & EDA

### Features created from raw event logs
The raw event log is aggregated into one feature row per user. Features fall into a few natural groups:

**Login / access behaviour**
- Login frequency (rate-normalized per day)
- Number of distinct IP addresses and regions used
- Ratio of logins happening at unusual hours (e.g. late night)
- Average and minimum time between consecutive events (very small gaps can suggest automated/bot-like activity)

**Trading behaviour**
- Trade frequency per hour of observed activity
- Maximum trades within a short rolling window (bursty activity)
- Leverage and margin usage statistics
- Variability of profit/loss across trades

**Deposit / withdrawal behaviour**
- Ratio of total withdrawn to total deposited
- A "churn" ratio comparing deposit + withdrawal activity to actual trading volume

### Why these features are useful for anomaly detection
Raw event counts alone don't separate normal from anomalous users well, because activity volume naturally varies across a population. Turning the raw log into *rate-normalized* and *relative* features (per day, per hour, vs. the user's own history) makes the dataset comparable across users with very different overall activity levels. Features like inter-event timing and geographic switching speed are useful because they capture behaviour that's hard to fake convincingly even when overall volume looks normal — which is exactly the kind of signal an anomaly detector needs.

### Exploratory data analysis (EDA) performed
Before finalizing the feature set, I checked:
- Distribution plots for each feature to spot heavy skew or extreme outliers
- Correlation between related features (e.g. trade frequency vs. margin usage) to avoid redundant signals
- Comparison of feature distributions between the labelled "normal" and "anomalous" subsets, purely to sanity-check that the injected anomalies actually showed up differently — this was for validation only, not used inside the feature engineering logic itself

### Key observations and learnings
- A handful of raw counts (e.g. number of trades) were highly skewed, which is why most count-based features were converted to rates instead of being used directly.
- Some anomalous patterns showed up clearly in a single feature (e.g. trade bursts), while others only became visible when combining two features together (e.g. deposit/withdrawal churn relative to trading volume) — this was a useful lesson in why a single threshold rule isn't enough and why a broader feature set matters for the modelling phase ahead.
- Keeping the feature engineering script separate from the data generation script made it easy to re-run feature extraction on new data without regenerating events from scratch.

---

## Files Submitted in This Repository

```
AnomX/
├── README.md                     # this file
├── requirements.txt              # project dependencies
│
├── src/
│   ├── data_generator.py         # Week 3 — synthetic event generation script
│   └── feature_engineering.py    # Week 4 — feature engineering script
│
├── data/
│   ├── portal_events.parquet     # Week 3 — synthetic portal/access events (logins, profile changes, support)
│   ├── trade_events.parquet      # Week 3 — synthetic trading events (trades, deposits, withdrawals)
│   ├── user_labels.parquet       # Week 3 — ground-truth anomaly labels (evaluation only, not a feature)
│   └── features.parquet          # Week 4 — generated per-user feature dataset
│
└── notebooks/ (optional)
    └── eda.ipynb                 # Week 4 — exploratory data analysis / observations
```

**Minimum required for this submission:**
- `README.md` (this file)
- `src/data_generator.py`
- `data/portal_events.parquet`
- `data/trade_events.parquet`
- `data/user_labels.parquet`
- `src/feature_engineering.py`
- `data/features.parquet`
- Any EDA notes/observations (can live in this README, a notebook, or a separate `EDA.md`)

> Submitted in Parquet format rather than CSV — see the format note under **Week 3: Event Generation** above for the reasoning.
> Note: `features.parquet` is the full, unsplit per-user feature set produced by `build_features()`. The train/test split (`train_features.parquet` / `test_features.parquet`) happens at the model-training stage, which is outside the scope of this midterm (Weeks 1–4).
