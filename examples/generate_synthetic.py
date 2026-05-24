"""
generate_synthetic.py
---------------------
Write a small synthetic transactions CSV used by configs/sample_synthetic.yaml.
Column names differ from the UCI dataset so that the YAML schema mapping
is exercised.

Output: data/synthetic_transactions.csv
Columns: user_id, order_id, order_ts, units, price_inr, country_code
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic_transactions.csv"
SEED = 42


def main(n_customers: int = 2_000, mean_orders: int = 6) -> None:
    rng = np.random.default_rng(SEED)

    rows = []
    today = pd.Timestamp("2025-12-31")
    for cust in range(1, n_customers + 1):
        n_orders = max(1, rng.poisson(mean_orders))
        days_active = rng.integers(1, 700)
        order_days = rng.integers(0, days_active, size=n_orders)
        for d in order_days:
            ts = today - pd.Timedelta(days=int(d))
            units = max(1, int(rng.poisson(3)))
            price = float(np.round(rng.lognormal(2.5, 0.6), 2))
            rows.append({
                "user_id": f"U{cust:05d}",
                "order_id": f"O{rng.integers(1, 9_999_999):07d}",
                "order_ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "units": units,
                "price_inr": price,
                "country_code": rng.choice(["IN", "US", "GB", "DE", "AU"]),
            })

    df = pd.DataFrame(rows)
    cancel_idx = rng.choice(len(df), size=len(df) // 50, replace=False)
    df.loc[cancel_idx, "units"] *= -1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows for {n_customers:,} customers to {OUT}")


if __name__ == "__main__":
    main()
