import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_io.source_table import Source
from visplot.source_listing import source_listing


def _make_source(sid, name, ra, dec, calcode, flux_i):
    return Source(
        id=sid, name=name, ra_epoch_deg=ra, dec_epoch_deg=dec,
        ra_apparent_deg=ra, dec_apparent_deg=dec, epoch_year=2000.0,
        calcode=calcode, flux_i_jy=flux_i, flux_q_jy=(), flux_u_jy=(), flux_v_jy=(),
    )


def test_source_listing_sorts_by_id_and_shows_key_fields():
    # Deliberately inserted out of id order -- the table must still come out
    # sorted by id (1, then 2), not dict insertion order.
    source_table = {
        2: _make_source(2, "3C286", 202.7845, 30.5091, "", (0.0, 0.0)),
        1: _make_source(1, "3C48", 24.4222, 33.1598, "E", (16.5, 16.5)),
    }

    fig = source_listing(source_table)
    ax = fig.axes[0]
    table = ax.tables[0]

    def cell(row, col):
        return table[(row, col)].get_text().get_text()

    assert cell(1, 1) == "3C48"
    assert cell(1, 2) == "24.422200"
    assert cell(1, 4) == "E"
    assert cell(1, 5) == "16.5"
    assert cell(2, 1) == "3C286"
    assert cell(2, 4) == "--"  # empty CALCODE shown as a placeholder, not blank
    assert cell(2, 5) == "0"
    plt.close(fig)


def test_source_listing_handles_an_empty_source_table():
    fig = source_listing({})
    plt.close(fig)
