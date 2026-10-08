import relay


def test_package_imports_with_version() -> None:
    assert relay.__version__ == "0.1.0"
