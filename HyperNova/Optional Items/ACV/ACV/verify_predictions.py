#!/usr/bin/env python3
"""Score predict.py's actual output against the labelled Train cases.

python verify_predictions.py --data_dir data --labels data/Train_Labels.csv [--model acv_model.json]
Defaults: data_dir="data", labels="<data_dir>/Train_Labels.csv", model="acv_model.json" next to this file.

Runs predict.py's real read/score/rank pipeline (not train.py's in-memory LOCO check)
on each labelled file and scores it with (n-(r-1))/n.
"""
import argparse
import os

import pandas as pd

from predict import load_model, read_table, rank_cars, score_file_df


def file_score(rank, n):
    return (n - (rank - 1)) / n


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--model", default=os.path.join(here, "acv_model.json"))
    a = ap.parse_args()
    if a.labels is None:
        a.labels = os.path.join(a.data_dir, "Train_Labels.csv")

    lab = pd.read_csv(a.labels, dtype=str)
    lab.columns = [c.strip().lower() for c in lab.columns]
    labels = dict(zip(lab["filename"].str.strip(), lab["faulty_car"].str.strip()))

    weights, cfg = load_model(a.model)

    scores = []
    for fname, true_car in labels.items():
        path = os.path.join(a.data_dir, fname)
        if not os.path.exists(path):
            print(f"{fname} not found in {a.data_dir}, skipped")
            continue
        df = read_table(path)
        sc, info = score_file_df(df, weights, cfg)
        ranked = rank_cars(sc)

        try:
            rank = next(i for i, c in enumerate(ranked, 1) if int(c) == int(true_car))
        except (ValueError, StopIteration):
            rank = ranked.index(true_car) + 1 if true_car in ranked else None

        n = len(ranked)
        s = file_score(rank, n) if rank else 0.0
        scores.append(s)
        flag = "" if rank == 1 else "  not top pick" if rank else "  missing / not found"
        print(f"{fname:<28} true={true_car:<4} rank={rank}/{n}  score={s:.3f}{flag}  [{info}]")

    if scores:
        print(f"\nmean score across {len(scores)} file(s): {sum(scores)/len(scores):.3f}")
    else:
        print("no files scored, check --data_dir / --labels")


if __name__ == "__main__":
    main()
