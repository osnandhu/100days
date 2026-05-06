# Spending Analyst Agent - Deep Dive

## What Does This Agent Do?

The spending agent answers: **"Is anything weird in my recent transactions?"**

It pulls your transactions from the database and runs two anomaly detection
algorithms to flag unusual spending with severity levels.

---

## Core Concepts

### What is Anomaly Detection?

Finding data points that don't fit the normal pattern. In finance:
- A $500 grocery bill when you usually spend $80 → anomaly
- Buying fuel in a country you don't live in → anomaly
- Normal amount but on a weird day → might be anomaly

### Why Two Methods?

No single algorithm catches everything. We use two complementary approaches:

**Z-Score** (statistical) — Great for: "Is this amount unusual for THIS category?"
- Simple, fast, interpretable
- Limited to single-variable analysis (just the amount)

**Isolation Forest** (machine learning) — Great for: "Is this combination unusual?"
- Catches multivariate patterns
- Can detect unusual category + amount + timing combinations
- Less interpretable (it says "anomaly" but not clearly why)

Using both gives us higher confidence when they agree.

---

## How Z-Score Works

The Z-score measures how many standard deviations a value is from the mean.

```
z = (x - mean) / standard_deviation
```

Example: Your grocery transactions are `[80, 85, 90, 75, 500]`

```
mean = 166, std = 178
z-score of 500 = (500 - 166) / 178 = 1.88
```

Wait — that's not flagged? The problem: one extreme value inflates the
mean and std. With more data and consistent patterns, Z-score works well.

Our severity mapping:
```
Z > 2  → MEDIUM  (this happens ~5% of the time by chance)
Z > 3  → HIGH    (this happens ~0.3% of the time by chance)
Z > 4  → CRITICAL (this is exceptionally rare)
```

**Why we group by category**: A $200 restaurant bill isn't anomalous if you
dine out regularly. But $200 at a grocery store might be. Grouping ensures
each transaction is compared against similar transactions.

---

## How Isolation Forest Works

Imagine randomly drawing vertical and horizontal lines to split data points.
Normal points are clustered together — they need MANY splits to isolate.
Anomalies are far from the cluster — they need FEW splits to isolate.

```
Normal point:  needs 8-10 random splits to isolate → normal
Anomaly:       needs 2-3 random splits to isolate  → flagged
```

The algorithm builds 100 random trees (forest) and averages the results.

Key parameter: `contamination=0.1` tells the algorithm "expect about 10%
of transactions to be unusual." This is a tuning dial:
- Set too low (0.01) → misses real anomalies
- Set too high (0.3) → flags too much, causing alert fatigue

10% is a reasonable starting point for personal finance.

---

## How the Code Works (Step by Step)

### Step 1: Fetch Transactions (FetchTransactionsTool)

```python
with get_db_session() as db:
    txns = db.query(Transaction) \
        .filter(Transaction.user_id == user_id) \
        .filter(Transaction.date >= cutoff) \
        .order_by(Transaction.date.desc()) \
        .all()
```

Uses your existing SQLAlchemy models. The context manager (`with`) ensures
the database session is properly closed even if an error occurs.

### Step 2: Z-Score Pass

```python
for category, group in df.groupby("category"):
    z_scores = np.abs(stats.zscore(group["amount"].values))
    # Flag anything with z > 2
```

`groupby("category")` splits transactions by type, then computes z-scores
within each group independently.

### Step 3: Isolation Forest Pass

```python
features = df[["amount"]].copy()
features["category_code"] = pd.Categorical(df["category"]).codes

iso_forest = IsolationForest(contamination=0.1, random_state=42)
predictions = iso_forest.fit_predict(features)
# -1 = anomaly, 1 = normal
```

We encode categories as numbers so the algorithm can use them as features.
`random_state=42` makes results reproducible.

### Step 4: Combine and Severity Upgrade

If BOTH methods flag the same transaction, severity gets upgraded:
```
Z-score only        → MEDIUM
IsolationForest only → MEDIUM
Both agree           → HIGH (or CRITICAL if z > 4)
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Two detection methods | Higher confidence when both agree |
| Category grouping for Z-score | $200 means different things for groceries vs rent |
| 10% contamination rate | Balance between catching issues and avoiding false alarms |
| Severity levels (not just flag/no-flag) | Users need triage guidance, not just a dump of flagged items |

---

## Common Pitfalls

1. **Too few transactions** — Z-score needs at least 2-3 data points per
   category to compute meaningful statistics. The code handles this by
   skipping categories with < 2 transactions.

2. **Seasonal spending looks anomalous** — Holiday shopping in December will
   flag as anomalous if your model only sees Jan-Nov data. Solution: use
   more historical data (12+ months).

3. **Isolation Forest with small datasets** — Below 20-30 data points,
   Isolation Forest is unreliable. The code requires at least 5 transactions.

---

## Practice Exercises (Weekend)

### Exercise 1: Add Time-Based Features
Currently we use amount + category. Add `day_of_week` and `day_of_month`
as features for Isolation Forest. Does it catch more patterns?

### Exercise 2: Tune the Contamination Rate
Try contamination values of 0.05, 0.1, 0.15, and 0.2 on the same data.
Plot how the number of flagged anomalies changes. Where's the sweet spot?

### Exercise 3: Add a Third Method
Implement Local Outlier Factor (LOF) from sklearn as a third detection
method. LOF considers the density of surrounding points, making it good
at finding anomalies in clusters. Compare with Isolation Forest.

### Exercise 4: Build a Category Trend Tracker
Instead of one-off anomaly detection, track each category's monthly total
over time. Flag when a category's spending jumps by more than 30% month-
over-month. This catches gradual lifestyle inflation.
