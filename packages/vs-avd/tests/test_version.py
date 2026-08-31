import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_semver_to_pep440_converts_a_prerelease():
    from _version import semver_to_pep440

    assert semver_to_pep440("0.4.0-alpha2") == "0.4.0a2"
    assert semver_to_pep440("0.4.0-beta1") == "0.4.0b1"
    assert semver_to_pep440("0.4.0-rc1") == "0.4.0rc1"
    assert semver_to_pep440("0.4.0") == "0.4.0"


def test_version_is_read_from_cargo_toml():
    """The wheel's version must match the plugin it carries."""
    import tomllib

    from _version import package_version, semver_to_pep440

    root = pathlib.Path(__file__).resolve().parents[3] / "Cargo.toml"
    assert root.is_file(), f"expected the repo root Cargo.toml at {root}"
    cargo = tomllib.loads(root.read_text())["workspace"]["package"]["version"]

    assert package_version() == semver_to_pep440(cargo)


def test_an_unparseable_version_raises():
    import pytest
    from _version import semver_to_pep440

    with pytest.raises(ValueError):
        semver_to_pep440("not-a-version")
