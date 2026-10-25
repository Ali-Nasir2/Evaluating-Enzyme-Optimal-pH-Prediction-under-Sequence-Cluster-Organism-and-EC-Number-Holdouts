"""Audit published pHopt label concerns and quantify a flagged-record sensitivity analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "pHopt_data.csv"
PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions" / "combined_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "source_label_audit.csv"
SENSITIVITY_PATH = PROJECT_ROOT / "outputs" / "metrics" / "source_label_sensitivity.csv"

# Zhou and Fang (2026) specifically discuss this accession and report disagreement
# between the pHopt label and the experimental values in the cited source.
FLAGGED_RECORDS = {
    "B5LXD7": {
        "reported_label": 11.5,
        "reported_source_values": "1.0-1.5 and 4.0",
        "source_status": "Published discrepancy reported; source-level correction not applied",
        "reference": "Zhou and Fang (2026), doi:10.1093/bib/bbag543",
    }
}


def main() -> None:
    raw = pd.read_csv(RAW_DATA)
    predictions = pd.read_csv(PREDICTIONS, low_memory=False)

    audit_rows: list[dict[str, object]] = []
    for accession, concern in FLAGGED_RECORDS.items():
        matches = raw.loc[raw["Accession"].eq(accession)]
        audit_rows.append(
            {
                "accession": accession,
                "raw_match_count": len(matches),
                "raw_pHopt_values": ";".join(str(value) for value in sorted(matches["pHopt"].dropna().unique())),
                "raw_organisms": ";".join(sorted(matches["Organism"].dropna().astype(str).unique())),
                "raw_ec_numbers": ";".join(sorted(matches["EC Number"].dropna().astype(str).unique())),
                "reported_label": concern["reported_label"],
                "reported_source_values": concern["reported_source_values"],
                "source_status": concern["source_status"],
                "reference": concern["reference"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(REPORT_PATH, index=False)

    test_predictions = predictions.loc[predictions["evaluation_split"].eq("test")].copy()
    rows: list[dict[str, object]] = []
    for keys, group in test_predictions.groupby(["split_strategy", "feature_group", "model"]):
        if keys[2] == "mean_predictor" or not group["Accession"].isin(FLAGGED_RECORDS).any():
            continue
        retained = group.loc[~group["Accession"].isin(FLAGGED_RECORDS)]
        rows.append(
            {
                "split_strategy": keys[0],
                "feature_group": keys[1],
                "model": keys[2],
                "flagged_accessions": ";".join(sorted(set(group["Accession"]) & set(FLAGGED_RECORDS))),
                "n_test_before": len(group),
                "n_test_after": len(retained),
                "mae_original": group["absolute_error"].mean(),
                "mae_excluding_flagged": retained["absolute_error"].mean(),
                "mae_change": retained["absolute_error"].mean() - group["absolute_error"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(SENSITIVITY_PATH, index=False)


if __name__ == "__main__":
    main()
