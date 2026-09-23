"""Trivial smoke test proving pytest collection and pythonpath wiring work."""


def test_pytest_is_wired():
    assert True


def test_project_modules_importable():
    from modules import ugmrt_query as q  # noqa: F401
