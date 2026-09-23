import os

import pytest

from instruments.gmrt.antenna_table import Antenna, read_antenna_table, resolve_active_antennas

REAL_GWB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS"
REAL_GSB_FITS = "/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_gsb.FITS"


def _sample_antennas():
    return [
        Antenna(station_number=1, name="C00:01"),
        Antenna(station_number=2, name="C01:02"),
        Antenna(station_number=8, name="C07:08"),
        Antenna(station_number=20, name="S01:20"),
        Antenna(station_number=24, name="S05:24"),
        Antenna(station_number=25, name="W01:25"),
    ]


def test_resolve_active_antennas_excludes_prefix_matched_duds():
    result = resolve_active_antennas(_sample_antennas(), dud_names=["C07", "S05"])

    active_names = {a.name for a in result.active_antennas}
    dud_names_found = {a.name for a in result.dud_antennas}

    assert active_names == {"C00:01", "C01:02", "S01:20", "W01:25"}
    assert dud_names_found == {"C07:08", "S05:24"}
    assert result.unmatched_dud_names == []


def test_resolve_active_antennas_with_no_duds_returns_everything_active():
    antennas = _sample_antennas()
    result = resolve_active_antennas(antennas, dud_names=[])

    assert result.active_antennas == antennas
    assert result.dud_antennas == []
    assert result.unmatched_dud_names == []


def test_resolve_active_antennas_reports_unmatched_names_instead_of_ignoring_them():
    result = resolve_active_antennas(_sample_antennas(), dud_names=["C07", "Z99"])

    assert result.unmatched_dud_names == ["Z99"]
    assert {a.name for a in result.dud_antennas} == {"C07:08"}


def test_resolve_active_antennas_matches_an_exact_full_name_too():
    antennas = _sample_antennas() + [Antenna(station_number=30, name="NOCOLON")]
    result = resolve_active_antennas(antennas, dud_names=["NOCOLON"])

    assert {a.name for a in result.dud_antennas} == {"NOCOLON"}
    assert result.unmatched_dud_names == []


def test_resolve_active_antennas_does_not_loosely_match_without_a_colon_boundary():
    # "C0" should NOT match "C00:01" or "C01:02" -- only an exact name or a
    # "<name>:" prefix counts, unlike the archived engine's looser fallback.
    result = resolve_active_antennas(_sample_antennas(), dud_names=["C0"])

    assert result.unmatched_dud_names == ["C0"]
    assert result.dud_antennas == []


@pytest.mark.skipif(not os.path.exists(REAL_GWB_FITS), reason="real GWB raw data file not present on this host")
def test_read_antenna_table_against_the_real_gwb_file():
    # Verified 2026-09-23: unlike the GSB file (32 AN-table rows, including
    # placeholder entries C07:31 and S05:32 for the two DUD antennas), the
    # GWB file's own AN table only has 30 rows -- GWB's correlator doesn't
    # emit entries for C07/S05 at all. So on GWB, resolving the DUD list
    # against the table correctly finds no match for either name -- that's
    # expected and benign here, not a stale-config warning sign the way it
    # would be on GSB.
    antennas = read_antenna_table(REAL_GWB_FITS)

    assert len(antennas) == 30
    names = {a.name for a in antennas}
    assert not any(n.startswith("C07:") for n in names)
    assert not any(n.startswith("S05:") for n in names)

    result = resolve_active_antennas(antennas, dud_names=["C07", "S05"])
    assert len(result.active_antennas) == 30
    assert result.dud_antennas == []
    assert set(result.unmatched_dud_names) == {"C07", "S05"}


@pytest.mark.skipif(not os.path.exists(REAL_GSB_FITS), reason="real GSB raw data file not present on this host")
def test_read_antenna_table_against_the_real_gsb_file():
    # Same code, the other correlator's file: here the AN table genuinely
    # does carry placeholder entries for the two DUD antennas (C07:31,
    # S05:32), appended after the 30 real stations -- so unlike GWB, they
    # resolve as matched duds here, not unmatched names. Proves the same
    # resolve_active_antennas logic handles both correlators' different
    # table layouts correctly without any GSB/GWB-specific branching.
    antennas = read_antenna_table(REAL_GSB_FITS)

    assert len(antennas) == 32
    names = {a.name for a in antennas}
    assert "C07:31" in names
    assert "S05:32" in names

    result = resolve_active_antennas(antennas, dud_names=["C07", "S05"])
    assert len(result.active_antennas) == 30
    assert {a.name for a in result.dud_antennas} == {"C07:31", "S05:32"}
    assert result.unmatched_dud_names == []
