from gdpr_gap import __version__, ping


def test_package_imports() -> None:
    assert ping() == "ok"
    assert __version__ == "0.1.0"
