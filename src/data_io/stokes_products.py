"""Stokes products: a file's own correlation products, and I, Q, U, V derived
from them as the IAU defines them, for linear and circular feeds alike
(T26's flagging design, step 1).

The IAU (1974) definition as Hamaker & Bregman (1996, A&AS 117, 161) interpret
it, summarised by van Straten et al. (2010, PASA 27, 104; section 2.1, eqs.
9-12): in a right-handed frame, x North, y East, z toward the observer, I =
<|e_x|^2 + |e_y|^2>, Q = <|e_x|^2 - |e_y|^2>, U = <2 Re(e_x e_y*)>, V = <2 Im(e_x
e_y*)>. With the products as coherencies (XX = <e_x e_x*>, XY = <e_x e_y*>, ...)
and the circular receptors of that paper's eq. 15:

    linear:    I = XX + YY   Q = XX - YY   U = XY + YX       V = -i(XY - YX)
    circular:  I = RR + LL   Q = RL + LR   U = -i(RL - LR)   V = RR - LL

with no factor of 1/2 (the user, 2026-10-03). The signs of U and V also depend
on whether the instrument's R/L labels follow the IEEE handedness and on
lower-sideband conjugation; |V| does not.

A derived product is flagged where any product it is made from is: its weight
is the least of theirs (the AIPS convention: <= 0 is flagged).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from data_io.visibility_data import VisibilityBlock

DERIVED = ("I", "Q", "U", "V")
CIRCULAR = ("RR", "LL", "RL", "LR")
LINEAR = ("XX", "YY", "XY", "YX")
# Each derived product as (native product, coefficient) pairs, by feed basis.
RECIPES = {
    "circular": {"I": (("RR", 1), ("LL", 1)), "Q": (("RL", 1), ("LR", 1)),
                 "U": (("RL", -1j), ("LR", 1j)), "V": (("RR", 1), ("LL", -1))},
    "linear": {"I": (("XX", 1), ("YY", 1)), "Q": (("XX", 1), ("YY", -1)),
               "U": (("XY", 1), ("YX", 1)), "V": (("XY", -1j), ("YX", 1j))},
}


@dataclass(frozen=True)
class ProductPlan:
    """The products a selection shows (`labels`) and how they come from the
    file's: the STOKES pixels to read (`read`, in the file's order), and, when
    any product is derived, `matrix` [len(labels) x len(read)], each row the
    coefficients of one product (None: the products read are those shown)."""

    labels: tuple[str, ...]
    read: tuple[int, ...]
    matrix: np.ndarray | None = None

    @property
    def derived(self) -> bool:
        return self.matrix is not None


def feed_basis(file_labels) -> str | None:
    """"circular" or "linear" from the products a file holds; None when it
    holds neither kind (e.g. I, Q, U, V themselves)."""
    labels = set(file_labels)
    if labels & set(CIRCULAR):
        return "circular"
    if labels & set(LINEAR):
        return "linear"
    return None


def plan_products(wanted, file_labels) -> ProductPlan:
    """The plan for the products `wanted` (labels, any case), from a file
    holding `file_labels`: the file's own first, in its order, then derived
    ones in I, Q, U, V order. Raises ValueError naming what is missing."""
    file_labels = [str(label).upper() for label in file_labels]
    wanted = {str(label).strip().upper() for label in wanted if str(label).strip()}
    known = set(file_labels) | set(DERIVED)
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Stokes {', '.join(unknown)}: not among this file's products ({', '.join(file_labels)}) "
                         f"nor the derived {', '.join(DERIVED)}")
    native = [label for label in file_labels if label in wanted]
    derived = [label for label in DERIVED if label in wanted and label not in file_labels]
    if not derived:
        return ProductPlan(tuple(native), tuple(file_labels.index(label) for label in native))
    basis = feed_basis(file_labels)
    if basis is None:
        raise ValueError(f"Stokes {', '.join(derived)}: this file holds {', '.join(file_labels)}, neither linear "
                         f"({', '.join(LINEAR)}) nor circular ({', '.join(CIRCULAR)}) products to derive them from")
    needed = {product for label in derived for product, _ in RECIPES[basis][label]}
    missing = sorted(needed - set(file_labels), key=(CIRCULAR + LINEAR).index)
    if missing:
        raise ValueError(f"Stokes {', '.join(derived)} of {basis} feeds needs {', '.join(missing)}, which this file "
                         f"does not hold ({', '.join(file_labels)})")
    read = [i for i, label in enumerate(file_labels) if label in set(native) | needed]
    position = {file_labels[i]: k for k, i in enumerate(read)}
    matrix = np.zeros((len(native) + len(derived), len(read)), dtype=np.complex64)
    for row, label in enumerate(native):
        matrix[row, position[label]] = 1
    for row, label in enumerate(derived, start=len(native)):
        for product, coefficient in RECIPES[basis][label]:
            matrix[row, position[product]] = coefficient
    return ProductPlan(tuple(native + derived), tuple(read), matrix)


def apply_products(block: VisibilityBlock, plan: ProductPlan) -> VisibilityBlock:
    """`block` (its STOKES axis holding `plan.read`) with the products
    `plan.labels`: each a combination of those read, flagged where any it uses
    is. Unchanged when nothing is derived."""
    if not plan.derived:
        return block
    dim = block.axis_types.index("STOKES") + 1  # data's first dimension is the row
    data = weight = None
    if block.data is not None:
        data = np.moveaxis(np.moveaxis(block.data, dim, -1) @ plan.matrix.T, -1, dim)
    if block.weight is not None:
        read_weight = np.moveaxis(block.weight, dim, -1)
        uses = plan.matrix != 0
        weight = np.stack([read_weight[..., uses[k]].min(axis=-1) for k in range(len(plan.labels))], axis=-1)
        weight = np.moveaxis(weight, -1, dim)
    axis_indices = {**block.axis_indices, "STOKES": np.arange(len(plan.labels))}
    return replace(block, data=data, weight=weight, axis_indices=axis_indices, stokes_labels=list(plan.labels))
