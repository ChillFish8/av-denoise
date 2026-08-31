"""
Resolves the wheel's version from the workspace Cargo.toml

Should not be shipped with wheels.
"""

from __future__ import annotations

import os
import pathlib
import re
import tomllib


def semver_to_pep440(version: str) -> str:
    """ Converts a Cargo semver string into a PEP 440 release identifier. """
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.?(\d+))?", version)
    if match is None:
        raise ValueError(f"cannot convert {version!r} to a PEP 440 version")

    base, kind, number = match.groups()
    if kind is None:
        return base
    suffix = {"alpha": "a", "beta": "b", "rc": "rc"}[kind]
    return f"{base}{suffix}{number}"


def package_version() -> str:
    override = os.environ.get("VSAVD_VERSION")
    if override:
        return override

    root = pathlib.Path(__file__).resolve().parents[2] / "Cargo.toml"
    cargo = tomllib.loads(root.read_text())["workspace"]["package"]["version"]
    return semver_to_pep440(cargo)
