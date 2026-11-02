"""Reject contaminated archives, even when runtime payload bytes are correct."""

import base64
import csv
import importlib.util
import io
import hashlib
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest

tomllib = pytest.importorskip(
    "tomllib", reason="artifact validation tooling requires Python 3.11+"
)

spec = importlib.util.spec_from_file_location(
    "check_artifacts", Path(__file__).parents[1] / "scripts" / "check_artifacts.py",
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

VERSION = "0.1.1"
ROOT = f"slangimage-{VERSION}"
INFO = f"{ROOT}.dist-info"
PAYLOAD = {"slangimage/__init__.py": b"", "slangimage/playground.py": b"# playground\n"}
METADATA = f"Name: slangimage\nVersion: {VERSION}\n".encode()
# Repository and local notes that are not part of the distributable archives.
# The allowlist-based archives must reject them like any other stray file.
LOCAL_NOTES = (
    "AGENTS.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
    "maintainer/notes.md",
    "notes/LOCAL.md",
)


@pytest.fixture
def source(tmp_path):
    for name in checker.SDIST_FILES:
        (tmp_path / name).write_bytes(name.encode())
    return tmp_path


def wheel_files(source):
    return {
        **PAYLOAD,
        f"{INFO}/METADATA": METADATA,
        f"{INFO}/WHEEL": b"Wheel-Version: 1.0\n",
        f"{INFO}/entry_points.txt": b"[console_scripts]\n",
        **{f"{INFO}/licenses/{n}": (source / n).read_bytes() for n in ("LICENSE", "NOTICE")},
    }


def write_wheel(source, files, alter_record=None, extra_member=None):
    rows = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, f"sha256={digest}", str(len(data))])
    rows.append([f"{INFO}/RECORD", "", ""])
    if alter_record:
        alter_record(rows)
    record = io.StringIO()
    csv.writer(record).writerows(rows)
    path = source / "test.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(f"{INFO}/RECORD", record.getvalue())
        if extra_member:
            archive.writestr(extra_member, b"extra")
    return path


def sdist_files(source):
    return {
        f"{ROOT}/src/slangimage/__init__.py": PAYLOAD["slangimage/__init__.py"],
        f"{ROOT}/playground/app.py": PAYLOAD["slangimage/playground.py"],
        f"{ROOT}/PKG-INFO": METADATA,
        **{f"{ROOT}/{n}": (source / n).read_bytes() for n in checker.SDIST_FILES},
    }


def write_sdist(source, files, extra_member=None):
    path = source / "test.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if extra_member:
            archive.addfile(extra_member)
    return path


def test_valid_archives(source):
    checker.check_wheel(write_wheel(source, wheel_files(source)), PAYLOAD,
                        "slangimage", VERSION, source)
    checker.check_sdist(write_sdist(source, sdist_files(source)), PAYLOAD,
                        source, "slangimage", VERSION)


def test_torch_extra_pins_validated_autograd_stack():
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text())["project"]
    assert set(project["optional-dependencies"]["torch"]) == {
        "torch>=2.13,<2.14",
        "slangpy[torch]==0.42.0",
    }


@pytest.mark.parametrize("name", [".env", *LOCAL_NOTES, "experiments/private.py",
                                  "src/slangimage/secret.json"])
def test_sdist_rejects_unexpected_files(source, name):
    files = sdist_files(source)
    files[f"{ROOT}/{name}"] = b"unexpected"
    with pytest.raises(RuntimeError, match="file set mismatch"):
        checker.check_sdist(write_sdist(source, files), PAYLOAD, source, "slangimage", VERSION)


@pytest.mark.parametrize("name", [".env", *LOCAL_NOTES, f"{INFO}/private.txt",
                                  "other.dist-info/METADATA"])
def test_wheel_rejects_unexpected_files_even_with_valid_record(source, name):
    files = wheel_files(source)
    files[name] = b"unexpected"
    with pytest.raises(RuntimeError, match="file set mismatch"):
        checker.check_wheel(write_wheel(source, files), PAYLOAD, "slangimage", VERSION, source)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b", "a//b"])
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_unsafe_paths(source, name, kind):
    with pytest.raises(RuntimeError, match="unsafe archive path"):
        if kind == "wheel":
            checker.check_wheel(write_wheel(source, wheel_files(source), extra_member=name),
                                PAYLOAD, "slangimage", VERSION, source)
        else:
            checker.check_sdist(write_sdist(source, sdist_files(source), tarfile.TarInfo(name)),
                                PAYLOAD, source, "slangimage", VERSION)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_duplicate_members(source, kind):
    with pytest.raises(RuntimeError, match="duplicate archive path"):
        if kind == "wheel":
            with pytest.warns(UserWarning, match="Duplicate name"):
                path = write_wheel(source, wheel_files(source), extra_member=f"{INFO}/METADATA")
            checker.check_wheel(path, PAYLOAD, "slangimage", VERSION, source)
        else:
            path = write_sdist(source, sdist_files(source), tarfile.TarInfo(f"{ROOT}/PKG-INFO"))
            checker.check_sdist(path, PAYLOAD, source, "slangimage", VERSION)


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_sdist_rejects_links_and_special_files(source, member_type):
    member = tarfile.TarInfo(f"{ROOT}/link")
    member.type = member_type
    member.linkname = "README.md"
    with pytest.raises(RuntimeError, match="unsupported archive member type"):
        checker.check_sdist(write_sdist(source, sdist_files(source), member), PAYLOAD,
                            source, "slangimage", VERSION)


def test_wheel_rejects_symlink(source):
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(RuntimeError, match="unsupported archive member type"):
        checker.check_wheel(write_wheel(source, wheel_files(source), extra_member=member),
                            PAYLOAD, "slangimage", VERSION, source)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unhashed", "wrong_hash"])
def test_wheel_requires_complete_record(source, mutation):
    def alter(rows):
        if mutation == "missing":
            rows.pop(0)
        elif mutation == "duplicate":
            rows.append(rows[0])
        elif mutation == "unhashed":
            rows[0][1:] = ["", ""]
        else:
            rows[0][1] = "sha256=incorrect"

    with pytest.raises(RuntimeError, match="RECORD"):
        checker.check_wheel(write_wheel(source, wheel_files(source), alter),
                            PAYLOAD, "slangimage", VERSION, source)


@pytest.mark.parametrize("name", ["README.md", "NOTICE", "pyproject.toml"])
def test_sdist_rejects_stale_support_files(source, name):
    files = sdist_files(source)
    files[f"{ROOT}/{name}"] = b"stale"
    with pytest.raises(RuntimeError, match="stale payload"):
        checker.check_sdist(write_sdist(source, files), PAYLOAD, source, "slangimage", VERSION)
