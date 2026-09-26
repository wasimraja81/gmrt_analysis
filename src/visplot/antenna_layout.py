"""Antenna-array layout plot: positions in a local East-North-Up frame
relative to the array's own reference position, not raw ECEF -- ECEF x/y/z
don't align with any direction a plot of a physical array layout should
show (east, north).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from data_io.antenna_table import Antenna

def _directions_deg(angles_deg) -> list[tuple[float, float]]:
    return [(float(np.cos(np.radians(a))), float(np.sin(np.radians(a)))) for a in angles_deg]


# 16 directions (22.5 degrees apart) to search for a label placement in,
# when no narrower preferred set applies. Finer than a plain 8-point
# compass: a label's own on-screen footprint often subtends more than 45
# degrees, so 8 directions alone can leave adjacent candidates overlapping.
_LABEL_DIRECTIONS = _directions_deg(np.arange(0.0, 360.0, 22.5))
# A small clearance around the dot itself, in points, so a label never sits
# on top of (and hides) the marker it belongs to.
_MARKER_CLEARANCE_PT = 5.0


def _ecef_offsets_to_enu_m(dx_m, dy_m, dz_m, lat_rad, lon_rad):
    """Standard ECEF-delta-to-local-ENU rotation, given the reference point's
    own geodetic latitude/longitude."""
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    east_m = -sin_lon * dx_m + cos_lon * dy_m
    north_m = -sin_lat * cos_lon * dx_m - sin_lat * sin_lon * dy_m + cos_lat * dz_m
    return east_m, north_m


def _nearest_neighbor_distances_m(east_m: np.ndarray, north_m: np.ndarray) -> np.ndarray:
    """Each antenna's distance to its closest other antenna -- the local
    density that distinguishes a compact core (many close neighbours) from
    an isolated antenna on a long arm, regardless of how far that antenna
    sits from the array's overall centroid."""
    diff_e = east_m[:, np.newaxis] - east_m[np.newaxis, :]
    diff_n = north_m[:, np.newaxis] - north_m[np.newaxis, :]
    dist_m = np.hypot(diff_e, diff_n)
    np.fill_diagonal(dist_m, np.inf)
    return dist_m.min(axis=1)


def _largest_relative_gap_threshold(
    values: np.ndarray, min_count: int = 3, min_gap_ratio: float = 1.4,
) -> float | None:
    """The value at the largest *ratio* jump in sorted `values`, if that
    jump is at least `min_gap_ratio` and leaves at least `min_count` values
    below it; None if no such jump exists. A ratio test, not an absolute
    gap against the overall range, so it applies regardless of the values'
    own scale."""
    if len(values) < min_count + 1:
        return None
    sorted_v = np.sort(values)
    if sorted_v[-1] <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = sorted_v[1:] / sorted_v[:-1]
    ratios = np.where(np.isfinite(ratios), ratios, 0.0)
    ratios[: min_count - 1] = 0.0  # candidate index i needs i+1 >= min_count values below the gap
    largest_gap_idx = int(np.argmax(ratios))
    if ratios[largest_gap_idx] < min_gap_ratio:
        return None
    return float((sorted_v[largest_gap_idx] + sorted_v[largest_gap_idx + 1]) / 2.0)


def _detect_compact_core_mask(
    east_m: np.ndarray, north_m: np.ndarray, min_core_count: int = 3, min_gap_ratio: float = 1.4,
) -> np.ndarray | None:
    """Which antennas belong to a compact, densely packed core, if one
    exists -- based on nearest-neighbour spacing (a density concept), not
    distance from the array's centroid. Centroid distance alone cannot
    separate a genuine core from an individual antenna at the near end of
    a long arm: confirmed directly against GMRT's own layout, where the
    first antenna on each of the E/W/S arms sits at a centroid-distance
    comparable to the farthest core antennas (the core itself spans a
    continuous ~25m-1050m range from centroid, no gap there at all), even
    though that arm antenna's nearest OTHER antenna is 2-4x farther away
    than any two core antennas ever are from each other.

    Returns None if no antenna is meaningfully denser-packed than the rest
    (nothing to zoom into)."""
    nn_m = _nearest_neighbor_distances_m(east_m, north_m)
    threshold_m = _largest_relative_gap_threshold(nn_m, min_core_count, min_gap_ratio)
    if threshold_m is None:
        return None
    return nn_m <= threshold_m


_INSET_CORNER_CANDIDATES = [(0.52, 0.52), (0.02, 0.52), (0.52, 0.02), (0.02, 0.02)]
_INSET_SIZE_FRACTION = 0.46


def _choose_inset_corner(ax, outside_e: np.ndarray, outside_n: np.ndarray) -> tuple[float, float]:
    """Which corner (as an axes-fraction (x, y) origin) to put the inset in --
    whichever has the fewest non-core antennas underneath it, so the inset
    doesn't cover real data or force those antennas' own labels to compete
    with it for space. Not fixed to any one corner, since which corner is
    empty depends on the array's own layout (a Y-shaped array like GMRT's
    versus any other arrangement)."""
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    best_corner, best_count = _INSET_CORNER_CANDIDATES[0], None
    for fx, fy in _INSET_CORNER_CANDIDATES:
        e_lo, e_hi = xlo + fx * (xhi - xlo), xlo + (fx + _INSET_SIZE_FRACTION) * (xhi - xlo)
        n_lo, n_hi = ylo + fy * (yhi - ylo), ylo + (fy + _INSET_SIZE_FRACTION) * (yhi - ylo)
        count = int(np.sum((outside_e >= e_lo) & (outside_e <= e_hi) & (outside_n >= n_lo) & (outside_n <= n_hi)))
        if best_count is None or count < best_count:
            best_corner, best_count = (fx, fy), count
    return best_corner


def _corner_center_angle_deg(corner_x: float, corner_y: float) -> float:
    """The compass angle (0=east, 90=north, ...) of the diagonal quadrant an
    inset corner (an axes-fraction origin) sits in."""
    east_sign = 1.0 if corner_x > 0.25 else -1.0
    north_sign = 1.0 if corner_y > 0.25 else -1.0
    return float(np.degrees(np.arctan2(north_sign, east_sign)) % 360.0)


def _exclude_directions_near(
    directions: list[tuple[float, float]], excluded_center_deg: float | None, half_width_deg: float = 45.0,
) -> list[tuple[float, float]]:
    """Drop any direction pointing into the `half_width_deg`-wide arc
    centered on `excluded_center_deg` (e.g. the inset's own quadrant) --
    a leader line into that quadrant would run under or into the inset
    panel. A no-op if `excluded_center_deg` is None."""
    if excluded_center_deg is None:
        return directions
    kept = []
    for dx, dy in directions:
        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360.0
        diff = abs(angle_deg - excluded_center_deg) % 360.0
        if min(diff, 360.0 - diff) > half_width_deg:
            kept.append((dx, dy))
    return kept


def _perpendicular_offset_directions(
    center_e: float, center_n: float, e: np.ndarray, n: np.ndarray,
    angle_gap_deg: float = 25.0, excluded_center_deg: float | None = None,
) -> list[list[tuple[int, int]]]:
    """For each point, its preferred label-offset directions: perpendicular
    to the local arm's own extension direction, since offsetting along an
    arm's own line just stacks a label into the next antenna along it,
    while offsetting across the arm does not. The arm direction is found by
    clustering points angularly around `(center_e, center_n)` -- not tied to
    any array's own named arms -- and checking whether each cluster runs
    closer to horizontal or vertical."""
    angles_deg = np.degrees(np.arctan2(n - center_n, e - center_e)) % 360.0
    order = np.argsort(angles_deg)
    sorted_angles = angles_deg[order]
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 360.0]]))

    group_id = np.empty(len(angles_deg), dtype=int)
    gid = 0
    for k, idx in enumerate(order):
        if k > 0 and gaps[k - 1] > angle_gap_deg:
            gid += 1
        group_id[idx] = gid
    if group_id.max() > 0 and gaps[-1] <= angle_gap_deg:
        group_id[group_id == group_id.max()] = 0  # first and last groups wrap around into one arm

    def _angular_distance(a: float, b: float) -> float:
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)

    directions = []
    for i in range(len(angles_deg)):
        member_angles = np.radians(angles_deg[group_id == group_id[i]])
        horizontal = abs(np.mean(np.cos(member_angles))) >= abs(np.mean(np.sin(member_angles)))
        # Exactly north/south (horizontal arm) or exactly east/west (vertical
        # arm) -- no diagonal -- ordered so whichever heading points farther
        # from the inset's own quadrant is tried first (e.g. a south-going
        # arm prefers west over east when the inset sits in the SE corner).
        cardinal_pair = [90.0, 270.0] if horizontal else [0.0, 180.0]
        if excluded_center_deg is not None:
            cardinal_pair = sorted(cardinal_pair, key=lambda a: -_angular_distance(a, excluded_center_deg))
        directions.append(_directions_deg(cardinal_pair))
    return directions


def _min_clear_radius_px(anchor_px, direction, half_w_px, half_h_px, obstacles, max_radius_px):
    """The smallest radius along `direction` (a unit vector, display pixels)
    such that a `2*half_w_px` x `2*half_h_px` box centered at
    `anchor_px + radius*direction` overlaps none of `obstacles` -- computed
    directly from actual bounding boxes (an exponential search for an
    overlap-free radius, then binary search down to it), not guessed from a
    fixed list of candidate distances."""
    def box_at(radius):
        cx, cy = anchor_px[0] + direction[0] * radius, anchor_px[1] + direction[1] * radius
        return Bbox.from_bounds(cx - half_w_px, cy - half_h_px, 2 * half_w_px, 2 * half_h_px)

    def overlaps_at(radius):
        box = box_at(radius)
        return any(box.overlaps(obstacle) for obstacle in obstacles)

    if not overlaps_at(0.0):
        return 0.0
    radius = max(half_w_px, half_h_px, 1.0)
    while overlaps_at(radius):
        radius *= 2.0
        if radius > max_radius_px:
            return None
    lo, hi = radius / 2.0, radius
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if overlaps_at(mid):
            lo = mid
        else:
            hi = mid
    return hi


def _place_labels_without_overlap(
    ax, renderer, labels, e, n, fontsize, avoid_bboxes=(),
    preferred_directions=None, exclude_center_deg=None, strict_directions=False,
):
    """Place each label as close to its point as it can be while overlapping
    neither the point's own marker nor any already-placed label (or one of
    `avoid_bboxes`, e.g. an inset panel drawn over this same axes) --
    computed exactly via `_min_clear_radius_px`, not by trying a fixed list
    of candidate distances and hoping one is far enough. Each label gets a
    thin leader line back to its point and a light box behind its text.

    `preferred_directions`, if given, is one direction list per label (see
    `_perpendicular_offset_directions`); the direction requiring the
    smallest clearing radius among them is used, so a label still favours
    those headings without needing to guess how far out is enough.
    `strict_directions` restricts the search to exactly that list, never the
    full compass set (e.g. arm labels that must stay exactly
    north/south/east/west, never diagonal). `exclude_center_deg`, if given,
    drops directions pointing into that quadrant (e.g. the inset's own) from
    the full compass search too, not just the preferred set. Returns the
    placed label bboxes, so a caller can chain further placement calls that
    avoid them too."""
    fallback_directions = _exclude_directions_near(_LABEL_DIRECTIONS, exclude_center_deg) or _LABEL_DIRECTIONS
    fig = ax.figure
    marker_clearance_px = _MARKER_CLEARANCE_PT * fig.dpi / 72.0
    axes_bbox = ax.get_window_extent(renderer=renderer)
    max_radius_px = 4.0 * max(axes_bbox.width, axes_bbox.height)  # always enough to clear any obstacle on this axes

    placed_bboxes = list(avoid_bboxes)
    for i, (label, ei, ni) in enumerate(zip(labels, e, n)):
        directions = fallback_directions
        if preferred_directions is not None:
            preferred = preferred_directions[i]
            directions = preferred if strict_directions else preferred + [
                d for d in fallback_directions if d not in preferred
            ]

        anchor_px = ax.transData.transform((ei, ni))
        marker_bbox = Bbox.from_bounds(
            anchor_px[0] - marker_clearance_px, anchor_px[1] - marker_clearance_px,
            2 * marker_clearance_px, 2 * marker_clearance_px,
        )
        obstacles = [marker_bbox, *placed_bboxes]

        probe = ax.annotate(
            label, (ei, ni), textcoords="offset points", xytext=(0, 0), fontsize=fontsize, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.85),
        )
        probe_bbox = probe.get_window_extent(renderer=renderer)
        half_w_px, half_h_px = probe_bbox.width / 2.0, probe_bbox.height / 2.0
        probe.remove()

        best_direction, best_radius = None, None
        for direction in directions:
            radius = _min_clear_radius_px(anchor_px, direction, half_w_px, half_h_px, obstacles, max_radius_px)
            if radius is not None and (best_radius is None or radius < best_radius):
                best_direction, best_radius = direction, radius
        if best_radius is None:
            best_direction, best_radius = directions[0], max_radius_px

        radius_pt = best_radius * 72.0 / fig.dpi
        text = ax.annotate(
            label, (ei, ni), textcoords="offset points",
            xytext=(best_direction[0] * radius_pt, best_direction[1] * radius_pt),
            fontsize=fontsize, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6, shrinkA=0, shrinkB=3),
        )
        placed_bboxes.append(text.get_window_extent(renderer=renderer))
    return placed_bboxes


def antenna_layout(
    antennas: list[Antenna], array_location: EarthLocation,
    telescope: str | None = None, source_path: str | Path | None = None,
) -> Figure:
    """Antenna positions in local East-North metres, relative to
    `array_location`, labeled by name. Adds a zoomed inset over the compact
    core when `_detect_compact_core_mask` finds one -- a plain scatter at
    full-array scale leaves closely packed core antennas indistinguishable.

    `telescope` (e.g. the primary header's TELESCOP) and `source_path` (the
    UVFITS file this was read from) are both optional and purely for the
    title -- this function still does no file I/O of its own; a caller
    reads these once from the file it already opened and passes them in."""
    x_m = np.array([a.x_m for a in antennas])
    y_m = np.array([a.y_m for a in antennas])
    z_m = np.array([a.z_m for a in antennas])

    ref_x_m, ref_y_m, ref_z_m = (v.to_value("m") for v in array_location.to_geocentric())
    east_m, north_m = _ecef_offsets_to_enu_m(
        x_m - ref_x_m, y_m - ref_y_m, z_m - ref_z_m,
        array_location.lat.rad, array_location.lon.rad,
    )
    names = [a.name for a in antennas]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(east_m, north_m, s=30, c="tab:blue")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    title = f"Antenna layout: {telescope}" if telescope else "Antenna layout"
    if source_path is not None:
        title += f"\n(file: {Path(source_path).name})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    axins = None
    core_mask = _detect_compact_core_mask(east_m, north_m)
    if core_mask is not None:
        core_e, core_n = east_m[core_mask], north_m[core_mask]
        pad_m = 0.2 * max(core_e.max() - core_e.min(), core_n.max() - core_n.min(), 1.0)
        center_e, center_n = (core_e.max() + core_e.min()) / 2.0, (core_n.max() + core_n.min()) / 2.0
        half_span_m = max(core_e.max() - core_e.min(), core_n.max() - core_n.min()) / 2.0 + pad_m

        fig.canvas.draw()  # finalize ax's autoscaled xlim/ylim before picking a corner in data space
        corner_x, corner_y = _choose_inset_corner(ax, east_m[~core_mask], north_m[~core_mask])
        axins = ax.inset_axes([corner_x, corner_y, _INSET_SIZE_FRACTION, _INSET_SIZE_FRACTION])
        axins.scatter(core_e, core_n, s=40, c="tab:blue")
        axins.set_xlim(center_e - half_span_m, center_e + half_span_m)
        axins.set_ylim(center_n - half_span_m, center_n + half_span_m)
        axins.set_aspect("equal")
        # Arm labels approach the inset from whichever side faces the plot's
        # own center -- keep the inset's y-ticks on its outer side instead,
        # away from that traffic.
        if corner_x > 0.25:
            axins.yaxis.tick_right()
        axins.tick_params(labelsize=6)
        axins.grid(True, alpha=0.3)
        axins.patch.set_facecolor("aliceblue")
        axins.patch.set_alpha(1.0)  # fully opaque -- a partial alpha let the main plot's own grid show through and clash with the inset's
        ax.indicate_inset_zoom(axins, edgecolor="black")

    # Labels are placed only once every axes limit is final (inset zoom
    # included), and only after one draw so get_window_extent reflects the
    # real, final pixel layout rather than stale/default extents.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    avoid_bboxes = [axins.get_window_extent(renderer=renderer)] if axins is not None else []

    if core_mask is None:
        _place_labels_without_overlap(ax, renderer, names, east_m, north_m, fontsize=7, avoid_bboxes=avoid_bboxes)
    else:
        excluded_center_deg = _corner_center_angle_deg(corner_x, corner_y)

        # Arm (non-core) antennas: perpendicular to their own arm's
        # extension direction, exactly as far out as clears the marker and
        # every other placed label (computed exactly, not guessed), and
        # never pointing into the inset's own quadrant -- a leader line
        # there would run under the inset panel.
        outside_e, outside_n = east_m[~core_mask], north_m[~core_mask]
        outside_names = [name for name, keep in zip(names, core_mask) if not keep]
        arm_directions = _perpendicular_offset_directions(
            core_e.mean(), core_n.mean(), outside_e, outside_n, excluded_center_deg=excluded_center_deg,
        )
        _place_labels_without_overlap(
            ax, renderer, outside_names, outside_e, outside_n, fontsize=7, avoid_bboxes=avoid_bboxes,
            preferred_directions=arm_directions, exclude_center_deg=excluded_center_deg, strict_directions=True,
        )

        # Core antennas are not labeled in the main plot at all -- at this
        # scale their dots sit within a few hundred metres of each other, so
        # any label placed here just adds clutter or contends with the arm
        # labels for space; the inset already shows every core antenna
        # clearly labeled, which is what it is for.
        core_names = [name for name, keep in zip(names, core_mask) if keep]
        _place_labels_without_overlap(axins, renderer, core_names, core_e, core_n, fontsize=6)

    return fig
