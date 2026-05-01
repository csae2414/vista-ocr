"""Sanity checks: package imports cleanly and SpatialGrid math is correct."""
import vista_ocr
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid


def test_package_imports():
    assert vista_ocr.__version__ == "0.0.1"


def test_spatial_grid_paper_default():
    g = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    assert g.n_x == 248
    assert g.n_y == 351
    toks = g.all_tokens()
    assert len(toks) == 248 + 351
    assert toks[0] == "<x_0>"
    assert toks[247] == "<x_247>"
    assert toks[248] == "<y_0>"
    assert toks[-1] == "<y_350>"


def test_spatial_grid_clamps():
    g = SpatialGrid(canvas_h=100, canvas_w=100, quantizer_px=10, scheme="original")
    assert g.x_token(-5) == "<x_0>"
    assert g.x_token(99) == "<x_9>"
    assert g.x_token(99999) == "<x_9>"


def test_spatial_grid_3px_ablation():
    g = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=3, scheme="original")
    assert g.n_x == 827
    assert g.n_y == 1170


def test_spatial_grid_unified_scheme():
    g = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="unified")
    toks = g.all_tokens()
    assert all(t.startswith("<xy_") for t in toks)
    assert len(toks) == max(248, 351)
