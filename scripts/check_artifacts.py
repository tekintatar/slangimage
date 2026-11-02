#!/usr/bin/env python3
"""Validate every archive member and compare shipped sources byte-for-byte."""

from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


SDIST_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", ".gitignore")


def _check_members(entries: list[tuple[str, bool]], expected: set[str]) -> None:
    """Require exactly the allowed files; optional directories must be parents."""
    parents = {str(parent) for name in expected for parent in PurePosixPath(name).parents}
    seen: set[str] = set()
    files: set[str] = set()
    for name, is_dir in entries:
        normalized = name.removesuffix("/") if is_dir else name
        parts = normalized.split("/")
        if (
            not normalized or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise RuntimeError(f"unsafe archive path: {name!r}")
        if normalized in seen:
            raise RuntimeError(f"duplicate archive path: {name!r}")
        seen.add(normalized)
        if is_dir:
            if normalized not in parents:
                raise RuntimeError(f"unexpected archive directory: {name!r}")
        else:
            files.add(name)
    if files != expected:
        raise RuntimeError(
            f"archive file set mismatch: missing={sorted(expected - files)}, "
            f"extra={sorted(files - expected)}"
        )


def _source_payload(source_root: Path) -> dict[str, bytes]:
    package_root = source_root / "src" / "slangimage"
    payload: dict[str, bytes] = {}
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if path.suffix not in {".py", ".slang"} and relative.as_posix() != "py.typed":
            continue
        payload[f"slangimage/{relative.as_posix()}"] = path.read_bytes()
    payload["slangimage/playground.py"] = (source_root / "playground" / "app.py").read_bytes()
    return payload


def _project_metadata(source_root: Path) -> tuple[str, str]:
    project = tomllib.loads((source_root / "pyproject.toml").read_text())["project"]
    return project["name"], project["version"]


def _one(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {dist_dir}, found {matches}")
    return matches[0]


def _compare_payload(
    artifact: str,
    actual: dict[str, bytes],
    expected: dict[str, bytes],
) -> None:
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"{artifact} payload mismatch: missing={expected.keys() - actual.keys()}, "
            f"extra={actual.keys() - expected.keys()}"
        )
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise RuntimeError(f"{artifact} contains stale payload bytes: {changed}")


def _check_record(archive: zipfile.ZipFile, record_name: str, files: set[str]) -> None:
    rows = csv.reader(io.StringIO(archive.read(record_name).decode()))
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise RuntimeError("invalid RECORD row")
        name, digest, size = row
        if name not in files or name in seen:
            raise RuntimeError(f"unexpected or duplicate RECORD entry: {name}")
        seen.add(name)
        if name == record_name:
            if digest or size:
                raise RuntimeError("RECORD must leave its own hash and size empty")
            continue
        if not digest or not size:
            raise RuntimeError(f"missing RECORD hash or size: {name}")
        algorithm, encoded = digest.split("=", 1)
        if algorithm != "sha256":
            raise RuntimeError(f"unsupported RECORD hash for {name}: {algorithm}")
        data = archive.read(name)
        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if actual != encoded or len(data) != int(size):
            raise RuntimeError(f"invalid RECORD entry for {name}")
    if seen != files:
        raise RuntimeError(f"missing RECORD entries: {sorted(files - seen)}")


def check_wheel(
    wheel: Path,
    expected: dict[str, bytes],
    project_name: str,
    project_version: str,
    source_root: Path,
) -> None:
    with zipfile.ZipFile(wheel) as archive:
        dist_info = f"{project_name.replace('-', '_')}-{project_version}.dist-info"
        metadata_files = {
            f"{dist_info}/{name}"
            for name in ("METADATA", "WHEEL", "entry_points.txt", "RECORD",
                         "licenses/LICENSE", "licenses/NOTICE")
        }
        allowed = set(expected) | metadata_files
        for member in archive.infolist():
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in {0, stat.S_IFDIR if member.is_dir() else stat.S_IFREG}:
                raise RuntimeError(f"unsupported archive member type: {member.filename}")
        _check_members([(m.filename, m.is_dir()) for m in archive.infolist()], allowed)
        names = tuple(archive.namelist())
        actual = {name: archive.read(name) for name in names if name.startswith("slangimage/")}
        actual = {name: data for name, data in actual.items() if not name.endswith("/")}
        _compare_payload("wheel", actual, expected)

        metadata_name = f"{dist_info}/METADATA"
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata["Name"] != project_name or metadata["Version"] != project_version:
            raise RuntimeError(
                f"wheel metadata mismatch: {metadata['Name']}=={metadata['Version']}"
            )
        for name in ("LICENSE", "NOTICE"):
            if archive.read(f"{dist_info}/licenses/{name}") != (source_root / name).read_bytes():
                raise RuntimeError(f"wheel contains stale {name}")
        record_name = f"{dist_info}/RECORD"
        _check_record(archive, record_name, allowed)


def check_sdist(
    sdist: Path, expected: dict[str, bytes], source_root: Path,
    project_name: str, project_version: str,
) -> None:
    source_expected = {
        f"src/{name}": data for name, data in expected.items() if name != "slangimage/playground.py"
    }
    source_expected["playground/app.py"] = expected["slangimage/playground.py"]
    source_expected.update({name: (source_root / name).read_bytes() for name in SDIST_FILES})
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        root = f"{project_name.replace('-', '_')}-{project_version}"
        allowed = {f"{root}/{name}" for name in source_expected} | {f"{root}/PKG-INFO"}
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported archive member type: {member.name}")
        _check_members([(m.name, m.isdir()) for m in members], allowed)
        actual: dict[str, bytes] = {}
        for member in members:
            if member.isdir():
                continue
            relative = member.name.removeprefix(f"{root}/")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read {member.name}")
            data = extracted.read()
            if relative == "PKG-INFO":
                metadata = BytesParser().parsebytes(data)
                if metadata["Name"] != project_name or metadata["Version"] != project_version:
                    raise RuntimeError("sdist metadata mismatch")
            else:
                actual[relative] = data
    _compare_payload("sdist", actual, source_expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", default="dist", type=Path)
    parser.add_argument("--source-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    expected = _source_payload(source_root)
    if not expected:
        raise RuntimeError("no source package payload found")
    project_name, project_version = _project_metadata(source_root)
    wheel = _one(args.dist_dir, "*.whl")
    sdist = _one(args.dist_dir, "*.tar.gz")
    check_wheel(wheel, expected, project_name, project_version, source_root)
    check_sdist(sdist, expected, source_root, project_name, project_version)
    print(f"artifacts OK: all archive members validated; "
          f"{len(expected)} runtime files match source byte-for-byte")


if __name__ == "__main__":
    main()
