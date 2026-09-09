#!/usr/bin/env python3
"""Calculate the tune-depression ratios quoted for the three Fig. 9 cases.

The depressed tunes are the maximum-space-charge-shift endpoints already
quoted in the manuscript.  For a tune footprint, stating this convention is
important because there is no unique single tune for the whole distribution.
"""

from __future__ import annotations

import csv
from pathlib import Path


BARE_TUNE = (3.23, 2.81)
EMITTANCES_PI_MM_MRAD = (40, 60, 80)

# Values already quoted in the Fig. 9 discussion of pyffa_sc.tex.
PIC_DEPRESSED_TUNES = (
    (2.935, 2.478),
    (3.015, 2.575),
    (3.077, 2.643),
)
THEORY_DEPRESSED_TUNES = (
    (2.922, 2.477),
    (3.023, 2.594),
    (3.074, 2.645),
)


def rows():
    qx0, qz0 = BARE_TUNE
    for emittance, pic, theory in zip(
        EMITTANCES_PI_MM_MRAD,
        PIC_DEPRESSED_TUNES,
        THEORY_DEPRESSED_TUNES,
    ):
        for source, (qx, qz) in (("PIC", pic), ("theory", theory)):
            yield {
                "emittance_pi_mm_mrad": emittance,
                "source": source,
                "Qx0": qx0,
                "Qz0": qz0,
                "Qx_depressed": qx,
                "Qz_depressed": qz,
                "Qx_over_Qx0": qx / qx0,
                "Qz_over_Qz0": qz / qz0,
            }


def main():
    output = Path(__file__).resolve().parent / "results" / "fig9_tune_depression.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table = list(rows())
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)

    print("Definition: tune depression = Q_depressed / Q_bare")
    print("Representative Q_depressed: maximum-SC-shift endpoint of the footprint")
    print(f"Bare tune: (Qx0, Qz0) = {BARE_TUNE}")
    print("PIC values for direct use in the response/caption:")
    for row in table:
        if row["source"] == "PIC":
            print(
                f"  {row['emittance_pi_mm_mrad']:>2} pi mm mrad: "
                f"(Qx/Qx0, Qz/Qz0) = "
                f"({row['Qx_over_Qx0']:.3f}, {row['Qz_over_Qz0']:.3f})"
            )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
