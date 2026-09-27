import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from conftest import make_scratch_dir
from cli.visplot_args import NAMED_PLOTS
from visplot.derived_quantities import QUANTITY_NAMES
from cli.visplot import HIGHRES_DPI, LOWRES_DPI, _save_combined_pdf, build_arg_parser


def _dense_figure(n_points=50_000):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    ax.scatter(rng.normal(size=n_points), rng.normal(size=n_points), s=1.0, linewidths=0)
    return fig


def test_save_combined_pdf_rasterizes_every_scatter_collection():
    scratch = make_scratch_dir("cli_visplot_pdf_rasterization")
    fig = _dense_figure(n_points=100)

    _save_combined_pdf(scratch / "a.pdf", [("f", fig)], dpi=LOWRES_DPI)

    assert all(c.get_rasterized() for c in fig.axes[0].collections)
    plt.close(fig)


def test_highres_pdf_embeds_a_larger_image_than_lowres():
    scratch = make_scratch_dir("cli_visplot_pdf_sizes")
    figures = [("dense", _dense_figure())]

    lowres = scratch / "lowres.pdf"
    highres = scratch / "highres.pdf"
    _save_combined_pdf(lowres, figures, dpi=LOWRES_DPI)
    _save_combined_pdf(highres, figures, dpi=HIGHRES_DPI)

    assert HIGHRES_DPI > LOWRES_DPI
    assert highres.stat().st_size > lowres.stat().st_size
    plt.close(figures[0][1])


def test_highres_pdf_is_written_by_default_and_can_be_suppressed():
    parser = build_arg_parser()
    assert parser.parse_args(["x.fits", "--plots", "antenna-layout"]).no_highres_pdf is False
    assert parser.parse_args(["x.fits", "--plots", "antenna-layout", "--no-highres-pdf"]).no_highres_pdf is True


def test_point_size_defaults_to_automatic():
    args = build_arg_parser().parse_args(["x.fits", "--plots", "amp-vs-freq_mhz"])
    assert args.point_size is None


def test_every_argument_has_help_text():
    parser = build_arg_parser()
    missing = [a.dest for a in parser._actions if not a.help]
    assert missing == []


def test_help_plot_names_section_lists_every_plot_and_quantity():
    epilog = build_arg_parser().epilog
    section = epilog.split("examples:")[0]
    for name in sorted(NAMED_PLOTS | QUANTITY_NAMES):
        assert re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", section), name
