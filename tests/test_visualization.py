from __future__ import annotations

import matplotlib.pyplot as plt

from src.visualization import clean_label, save_figure


def test_clean_label_replaces_underscores():
    assert clean_label("cluster_safe") == "cluster safe"


def test_save_figure_writes_png_and_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    save_figure(fig, tmp_path, "test_plot")

    assert (tmp_path / "test_plot.png").exists()
    assert (tmp_path / "test_plot.pdf").exists()

