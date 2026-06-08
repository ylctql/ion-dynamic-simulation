"""Plotter 图例：--species 与 Ba 同位素参杂。"""
import numpy as np

from Plotter.color import get_legend_entries, has_mass_variation


def test_has_mass_variation_detects_doping():
    m = np.array([133 / 135, 1.0, 138 / 135])
    assert has_mass_variation(m)


def test_has_mass_variation_uniform():
    assert not has_mass_variation(np.ones(10))


def test_legend_species_label_yb171():
    m = np.ones(50)
    labels, colors = get_legend_entries(m, species_label="Yb171+")
    assert labels == ["Yb171+"]
    assert colors == ["red"]


def test_legend_ba_doping_ignores_species_label():
    m = np.array([133 / 135, 1.0, 138 / 135])
    labels, colors = get_legend_entries(m, species_label="Yb171+")
    assert "Ba133" in labels
    assert "Ba135" in labels
    assert "Ba138" in labels
    assert "Yb171+" not in labels
    assert len(colors) == len(labels)


def test_legend_isotope_mode_default_ba135():
    m = np.ones(5)
    labels, _ = get_legend_entries(m, color_mode="isotope")
    assert labels == ["Ba135"]


def test_legend_y_pos_no_species_no_legend():
    m = np.ones(5)
    labels, colors = get_legend_entries(m, color_mode="y_pos")
    assert labels == []
    assert colors == []
