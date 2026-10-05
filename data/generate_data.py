"""Generate a synthetic A/B-test dataset: one-page checkout redesign in a Baku food-delivery app.

The numbers are SYNTHETIC (fixed seed) but shaped on realistic experiment patterns:
- 50/50 user-level randomisation over 21 days (2026-09-01 .. 2026-09-21)
- control conversion ~11-12%; the new one-page checkout lifts conversion, mostly on mobile
  and for new users, and is slightly worse on the web
- a novelty bump in the first days of the test
- pre-experiment activity (orders in the 30 days before) correlates with outcomes -> CUPED
- a few data-quality issues on purpose: duplicate exposure rows, users seen in both groups,
  missing order values, and internal QA/test accounts
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)
N = 48_000

start = pd.Timestamp("2026-09-01")
days = 21
user_id = rng.choice(np.arange(100_000, 999_999), N, replace=False)
group = rng.choice(["control", "treatment"], N, p=[0.5, 0.5])
assign_day = rng.integers(0, days, N)
assign_date = start + pd.to_timedelta(assign_day, unit="D")

device = rng.choice(["android", "ios", "web"], N, p=[0.52, 0.31, 0.17])
user_type = rng.choice(["new", "returning"], N, p=[0.38, 0.62])
city = rng.choice(["Baku", "Sumgait", "Ganja"], N, p=[0.78, 0.13, 0.09])

# Pre-period engagement (orders in the 30 days before the test)
lam = np.where(user_type == "new", 0.15, 1.9)
pre_orders = rng.poisson(lam * rng.gamma(2.0, 0.5, N))

# --- Conversion probability (log-odds model)
base = -3.55
logit = (base
         + 1.55 * np.log1p(pre_orders)
         + np.select([device == "ios", device == "web"], [0.12, -0.25], 0.0)
         + np.where(city == "Baku", 0.05, -0.08)
         + np.where(user_type == "new", -0.15, 0.0))
is_t = group == "treatment"
effect = np.select(
    [device == "android", device == "ios", device == "web"], [0.08, 0.06, -0.09], 0.0)
effect = effect + np.where(user_type == "new", 0.06, 0.0)
novelty = np.where(assign_day < 4, 0.08, 0.0)          # extra lift in the first 4 days
logit = logit + is_t * (effect + novelty)
p_conv = 1 / (1 + np.exp(-logit))
converted = rng.random(N) < p_conv

# --- Orders and order value (AZN) for converters
n_orders = np.where(converted, 1 + rng.poisson(0.25 + 0.12 * np.log1p(pre_orders)), 0)
aov = rng.lognormal(mean=np.log(21.5), sigma=0.45, size=N)
aov = aov * np.where(device == "web", 1.12, 1.0) * np.where(city == "Baku", 1.04, 0.92)
aov = aov * np.where(is_t, 0.99, 1.0)                  # tiny basket-size drop (no upsell page)
revenue = np.where(converted, n_orders * aov, 0.0)

# --- Guardrails / UX metrics
cancel_p = np.where(is_t, 0.052, 0.048)
cancelled = converted & (rng.random(N) < cancel_p)
checkout_sec = np.where(
    converted,
    rng.gamma(4.0, np.where(is_t, 17.0, 22.5)) * np.where(device == "web", 1.15, 1.0),
    np.nan)
sessions = 1 + rng.poisson(1.6 + 0.6 * np.log1p(pre_orders))

df = pd.DataFrame({
    "user_id": user_id,
    "assign_date": assign_date.date,
    "group": group,
    "device": device,
    "user_type": user_type,
    "city": city,
    "pre_orders_30d": pre_orders,
    "sessions": sessions,
    "converted": converted.astype(int),
    "orders": n_orders,
    "revenue_azn": revenue.round(2),
    "cancelled": cancelled.astype(int),
    "checkout_time_sec": np.round(checkout_sec, 1),
})

# --- Inject data-quality issues
# 1) duplicate exposure rows (logging fired twice)
dups = df.sample(220, random_state=SEED)
# 2) users exposed to both variants (cookie reset / device switch)
both = df.sample(140, random_state=SEED + 1).copy()
both["group"] = np.where(both["group"] == "control", "treatment", "control")
# 3) internal QA accounts hammering the treatment
qa = pd.DataFrame({
    "user_id": np.arange(1, 61), "assign_date": start.date(), "group": "treatment",
    "device": "android", "user_type": "returning", "city": "Baku", "pre_orders_30d": 40,
    "sessions": rng.integers(30, 90, 60), "converted": 1, "orders": rng.integers(5, 15, 60),
    "revenue_azn": 0.0, "cancelled": 1, "checkout_time_sec": 9.0,
})
df = pd.concat([df, dups, both, qa], ignore_index=True)
# 4) missing order values for some converters (payment-callback gaps)
conv_idx = df.index[(df.converted == 1) & (df.user_id > 100)]
df.loc[rng.choice(conv_idx, 35, replace=False), "revenue_azn"] = np.nan

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
out = Path(__file__).parent / "checkout_ab_test.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows -> {out}")
