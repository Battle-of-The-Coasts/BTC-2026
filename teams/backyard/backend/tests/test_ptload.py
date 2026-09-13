import os
import numpy as np
import pytest
from trustscore.ptload import load_pt

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


@pytest.mark.skipif(not os.path.exists(os.path.join(RAW, "MGTAB", "MGTAB", "labels_bot.pt")), reason="MGTAB not downloaded")
def test_mgtab_labels():
    y = load_pt(os.path.join(RAW, "MGTAB", "MGTAB", "labels_bot.pt"))
    assert y.shape == (10199,) and set(np.unique(y)) == {0, 1}
    ei = load_pt(os.path.join(RAW, "MGTAB", "MGTAB", "edge_index.pt"))
    assert ei.shape[0] == 2 and ei.max() < 10199
