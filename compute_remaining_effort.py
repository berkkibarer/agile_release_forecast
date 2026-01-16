#!/usr/bin/env python3
"""Compute total completed effort and reasonable remaining effort for each validation dataset"""
import pandas as pd
import json

datasets = [
    "validation_datasets/small_low_cv.csv",
    "validation_datasets/small_high_cv.csv",
    "validation_datasets/medium_mixed.csv",
    "validation_datasets/large_stable.csv",
    "validation_datasets/large_volatile.csv",
    "validation_datasets/xlarge_realistic.csv"
]

for ds in datasets:
    df = pd.read_csv(ds)
    total_velocity = df['velocity'].sum()
    # Total release = completed + remaining (where remaining is 25% of completed work)
    remaining = round(total_velocity * 0.25, 1)
    total_release = round(total_velocity + remaining, 1)
    print(f"{ds.split('/')[-1]}: completed={total_velocity:.1f}, remaining={remaining:.1f}, total_release={total_release:.1f}")
