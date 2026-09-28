import os

import numpy as np
import pytest

from conftest import make_scratch_dir
from data_io.visibility_data import VisibilityBlock
from visplot.plot_spec import PlotSpec
from visplot.quantities import QuantityContext
from visplot.range_cache import CACHE_PREFIX, RangeCache, clear_cache
from visplot.stream import run_stream
from visplot.xy_session import resolve_extents

CTX = QuantityContext(time_reference_jd=2459421.0)


def _fits_stand_in(scratch):
    path = scratch / "obs.fits"
    path.write_bytes(b"stand-in")
    return path


def _key(cache, path, rows=np.arange(3), quantity="amp", **kw):
    options = dict(apply_flags=True, show_flagged=False, mirror=False, log_axis=False)
    options.update(kw)
    return cache.key(path, rows, {"STOKES": np.array([0])}, quantity, **options)


def test_key_changes_with_everything_the_range_depends_on():
    scratch = make_scratch_dir("range_cache_key")
    path = _fits_stand_in(scratch)
    cache = RangeCache(scratch / "cache")
    base = _key(cache, path)
    assert _key(cache, path) == base
    assert _key(cache, path, rows=np.arange(4)) != base
    assert _key(cache, path, quantity="real") != base
    assert _key(cache, path, mirror=True) != base
    assert _key(cache, path, log_axis=True) != base
    os.utime(path, ns=(0, 12345))
    assert _key(cache, path) != base  # the file changed


def test_save_and_load_leave_one_prefixed_file_and_no_partial():
    scratch = make_scratch_dir("range_cache_save")
    path = _fits_stand_in(scratch)
    cache = RangeCache(scratch / "cache")
    key = _key(cache, path)
    counts = np.arange(10, dtype=np.int64)
    written = cache.save(path, key, 1.5, 9.5, counts, {"quantity": "amp"})
    assert written.name.startswith(f"{CACHE_PREFIX}obs_") and written.suffix == ".npz"
    assert [f.name for f in (scratch / "cache").iterdir()] == [written.name]
    lo, hi, loaded = cache.load(path, key)
    assert (lo, hi) == (1.5, 9.5)
    np.testing.assert_array_equal(loaded, counts)
    assert cache.load(path, "0" * 20) is None


def test_stale_partials_are_removed_and_a_live_writers_partial_is_kept():
    scratch = make_scratch_dir("range_cache_partials")
    cache = RangeCache(scratch / "cache")
    dead = scratch / "cache" / f"{CACHE_PREFIX}obs_abc.npz.999999999.partial"
    live = scratch / "cache" / f"{CACHE_PREFIX}obs_def.npz.{os.getpid()}.partial"
    dead.write_bytes(b"x")
    live.write_bytes(b"x")
    assert cache.remove_stale_partials() == [dead]
    assert not dead.exists() and live.exists()


def test_clear_cache_removes_only_visplot_cache_files():
    scratch = make_scratch_dir("range_cache_clear")
    directory = scratch / "cache"
    directory.mkdir()
    ours = [directory / f"{CACHE_PREFIX}obs_1.npz", directory / f"{CACHE_PREFIX}obs_2.npz.123.partial"]
    other = directory / "my_notes.txt"
    for f in ours + [other]:
        f.write_bytes(b"x")
    assert sorted(clear_cache(directory)) == sorted(ours)
    assert [f.name for f in directory.iterdir()] == ["my_notes.txt"]


class _Source:
    """Streams synthetic chunks; counts passes. amp of row r is r + 1."""

    def __init__(self, fits_path):
        self.fits_path = fits_path
        self.row_indices = np.arange(6)
        self.axis_selection = None
        self.n_rows = 6
        self.passes = 0

    def stream(self, reducers, read_data, on_chunk=None):
        self.passes += 1
        rows = self.row_indices
        data = (rows[:, None, None] + 1.0) * np.ones((1, 1, 2)) + 0j
        block = VisibilityBlock(
            row_indices=rows, data=data, weight=np.ones(data.shape), axis_types=["STOKES", "FREQ"],
            axis_indices={"STOKES": np.array([0]), "FREQ": np.arange(2)}, ant1=rows, ant2=rows,
            source_id=np.ones(6, dtype=int), jd=2459421.0 + rows / 24.0, uu_sec=rows * 1.0,
            vv_sec=rows * 0.0, ww_sec=rows * 0.0, chan_freqs_hz=np.array([1e9, 1.1e9]), stokes_labels=["RR"],
        )
        return run_stream([block], CTX, reducers, on_chunk)


def test_a_second_run_takes_the_ranges_from_the_cache_without_a_pass():
    scratch = make_scratch_dir("range_cache_reuse")
    source = _Source(_fits_stand_in(scratch))
    cache = RangeCache(scratch / "cache")
    plot = PlotSpec(y="amp", x="time_h")

    first = resolve_extents(source, [plot], cache=cache)
    assert source.passes == 1 and len(cache.written) == 2  # x and y ranges

    percentile_plot = PlotSpec(y="amp", x="time_h", y_range_mode="percentile", range_percentiles=(20.0, 80.0))
    again = resolve_extents(source, [plot], cache=RangeCache(scratch / "cache"))
    narrowed = resolve_extents(source, [percentile_plot], cache=RangeCache(scratch / "cache"))
    assert source.passes == 1  # both answered from the cache
    assert again == first
    assert narrowed[percentile_plot][1][1] < first[plot][1][1]  # the cached histogram serves percentiles


def test_an_interrupted_pass_saves_nothing():
    scratch = make_scratch_dir("range_cache_interrupted")
    source = _Source(_fits_stand_in(scratch))
    cache = RangeCache(scratch / "cache")
    resolve_extents(source, [PlotSpec(y="amp", x="time_h")], on_chunk=lambda rows: False, cache=cache)
    assert cache.written == [] and cache.usage() == (0, 0)
