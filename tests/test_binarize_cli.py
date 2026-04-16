from pathlib import Path

import pandas as pd

from wildlife_strikes.binarize_cli import binarize_submission


def test_binarize_submission_threshold(tmp_path: Path):
    inp = tmp_path / "submission.csv"
    out = tmp_path / "submission_binary.csv"
    pd.DataFrame(
        {
            "INDEX_NR": [1, 2, 3, 4],
            "INDICATED_DAMAGE": [0.1, 0.5, 0.50001, 0.2],
        }
    ).to_csv(inp, index=False)

    rows, positives = binarize_submission(inp, out, threshold=0.5)
    assert rows == 4
    assert positives == 2

    got = pd.read_csv(out)
    assert got["INDICATED_DAMAGE"].tolist() == [0, 1, 1, 0]
