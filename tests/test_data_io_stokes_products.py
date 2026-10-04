"""Stokes products: the file's own, and I, Q, U, V as the IAU defines them,
for circular and linear feeds (no factor of 1/2)."""

import numpy as np
import pytest

from data_io.stokes_products import DERIVED, apply_products, derivable, feed_basis, plan_products, recipe_text
from data_io.visibility_data import VisibilityBlock


def _block(products: dict, weights: dict | None = None) -> VisibilityBlock:
    """One row, two channels, the given products on the STOKES axis (the file's order), as
    a block holding every product read: axes STOKES, FREQ (data shape [1, nstokes, 2])."""
    labels = list(products)
    data = np.array([[products[label] for label in labels]], dtype=np.complex64)[..., None].repeat(2, axis=-1)
    weight = np.ones(data.shape, dtype=np.float32)
    for label, value in (weights or {}).items():
        weight[0, labels.index(label), :] = value
    one = np.zeros(1)
    return VisibilityBlock(row_indices=np.array([0]), data=data, weight=weight, axis_types=["STOKES", "FREQ"],
                           axis_indices={"STOKES": np.arange(len(labels)), "FREQ": np.arange(2)},
                           ant1=np.array([1]), ant2=np.array([2]), source_id=np.array([1]), jd=one, uu_sec=one,
                           vv_sec=one, ww_sec=one, chan_freqs_hz=np.array([1e8, 2e8]), stokes_labels=labels)


def _shown(products: dict, wanted: str, weights: dict | None = None):
    labels = list(products)
    plan = plan_products(wanted.split(","), labels)
    block = _block({label: products[label] for label in (labels[i] for i in plan.read)}, weights)
    out = apply_products(block, plan)
    return plan, {label: complex(out.data[0, k, 0]) for k, label in enumerate(out.stokes_labels)}, out


def test_circular_feeds_give_i_q_u_v_with_no_factor_of_a_half():
    products = {"RR": 3 + 1j, "LL": 1 - 1j, "RL": 0.5 + 0.25j, "LR": 0.5 - 0.75j}
    plan, shown, out = _shown(products, "I,Q,U,V")
    assert plan.labels == ("I", "Q", "U", "V") and feed_basis(products) == "circular"
    assert shown["I"] == pytest.approx(4 + 0j)  # RR + LL
    assert shown["V"] == pytest.approx(2 + 2j)  # RR - LL
    assert shown["Q"] == pytest.approx(1 - 0.5j)  # RL + LR
    assert shown["U"] == pytest.approx(1 + 0j)  # -i(RL - LR) = -i(1j)
    assert out.data.shape == (1, 4, 2) and out.axis_indices["STOKES"].tolist() == [0, 1, 2, 3]


def test_linear_feeds_give_i_q_u_v():
    products = {"XX": 2 + 0j, "YY": 1 + 0j, "XY": 0.5 + 0.5j, "YX": 0.5 - 0.5j}
    plan, shown, _ = _shown(products, "V,I,U,Q")  # shown in I, Q, U, V order whatever the order asked
    assert plan.labels == ("I", "Q", "U", "V") and feed_basis(products) == "linear"
    assert [shown[label] for label in plan.labels] == pytest.approx([3, 1, 1, 1])  # V = -i(XY - YX) = 2 Im XY


def test_a_derived_product_is_flagged_where_any_product_it_uses_is():
    products = {"RR": 1 + 0j, "LL": 1 + 0j, "RL": 0j, "LR": 0j}
    _, _, out = _shown(products, "I,Q,V", weights={"RR": -1.0})
    weight = dict(zip(out.stokes_labels, out.weight[0, :, 0]))
    assert weight["I"] <= 0 and weight["V"] <= 0 and weight["Q"] > 0


def test_native_products_pass_through_and_read_only_what_is_needed():
    file_labels = ["RR", "LL", "RL", "LR"]
    native = plan_products(["ll", "RR"], file_labels)
    assert native.labels == ("RR", "LL") and native.read == (0, 1) and not native.derived  # the file's order
    mixed = plan_products(["RR", "V"], file_labels)
    assert mixed.labels == ("RR", "V") and mixed.read == (0, 1)  # V needs RR and LL alone
    _, shown, _ = _shown({"RR": 3 + 1j, "LL": 1 - 1j, "RL": 0j, "LR": 0j}, "RR,V")
    assert shown == pytest.approx({"RR": 3 + 1j, "V": 2 + 2j})
    held = plan_products(["I"], ["I", "Q", "U", "V"])  # a file holding I itself: read as it is
    assert held.labels == ("I",) and held.read == (0,) and not held.derived


def test_what_cannot_be_derived_is_refused_naming_what_is_missing():
    with pytest.raises(ValueError, match=r"Stokes V of linear feeds needs XY, YX, which this file does not hold"):
        plan_products(["V"], ["XX", "YY"])
    with pytest.raises(ValueError, match=r"Stokes Q of circular feeds needs RL, LR"):
        plan_products(["Q"], ["RR", "LL"])
    with pytest.raises(ValueError, match=r"Stokes ZZ: not among this file's products \(RR, LL\) nor the derived"):
        plan_products(["ZZ"], ["RR", "LL"])
    with pytest.raises(ValueError, match=r"neither linear .* nor circular"):
        plan_products(["V"], ["I"])
    with pytest.raises(ValueError, match=r"no Stokes product named \(this file holds RR, LL\)"):
        plan_products(["", " "], ["RR", "LL"])  # e.g. every box of a form unticked


def test_what_a_file_can_derive_and_how_it_is_written():
    assert derivable(["RR", "LL"]) == ("I", "V") and derivable(["XX", "YY"]) == ("I", "Q")
    assert derivable(["RR", "LL", "RL", "LR"]) == DERIVED and derivable(["I", "Q"]) == ()
    assert [recipe_text(label, ["RR", "LL", "RL", "LR"]) for label in DERIVED] == [
        "I = RR + LL", "Q = RL + LR", "U = -i(RL - LR)", "V = RR - LL"]
    assert [recipe_text(label, ["XX", "YY", "XY", "YX"]) for label in DERIVED] == [
        "I = XX + YY", "Q = XX - YY", "U = XY + YX", "V = -i(XY - YX)"]
    assert recipe_text("RR", ["RR", "LL"]) == "RR: as the file holds it" and recipe_text("v", ["rr", "ll"]) == "V = RR - LL"
    with pytest.raises(ValueError, match="neither held nor derivable"):
        recipe_text("RR", ["XX", "YY"])
