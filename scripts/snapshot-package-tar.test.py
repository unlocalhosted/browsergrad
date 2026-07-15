#!/usr/bin/env python3
"""Adversarial tests for the bounded npm package tarball snapshotter."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).with_name("snapshot-package-tar.py")
SPEC = importlib.util.spec_from_file_location("snapshot_package_tar", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
snapshot_package_tar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_package_tar)


def regular_file(
    name: str, payload: bytes, *, mode: int = 0o644
) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = mode
    return member, payload


def directory(name: str, *, mode: int = 0o755) -> tuple[tarfile.TarInfo, None]:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    member.mode = mode
    return member, None


def special_member(
    name: str, member_type: bytes, *, linkname: str = ""
) -> tuple[tarfile.TarInfo, None]:
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = linkname
    return member, None


class SnapshotPackageTarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.tarball_index = 0

    def write_tarball(
        self,
        members: list[tuple[tarfile.TarInfo, bytes | None]],
        *,
        archive_format: int = tarfile.PAX_FORMAT,
    ) -> Path:
        self.tarball_index += 1
        tarball = self.root / f"fixture-{self.tarball_index}.tgz"
        with tarfile.open(tarball, mode="w:gz", format=archive_format) as archive:
            for member, payload in members:
                source = io.BytesIO(payload) if payload is not None else None
                archive.addfile(member, source)
        return tarball

    def assert_snapshot_rejected(
        self, members: list[tuple[tarfile.TarInfo, bytes | None]], message: str
    ) -> None:
        tarball = self.write_tarball(members)
        with self.assertRaisesRegex(ValueError, message):
            snapshot_package_tar.snapshot(tarball)

    def test_snapshots_valid_package_files_canonically(self) -> None:
        alpha = b"alpha\x00payload"
        zeta = b"zeta\n"
        package_json = json.dumps(
            {"name": "@unlocalhosted/test-package", "version": "1.2.3"},
            separators=(",", ":"),
        ).encode()
        tarball = self.write_tarball(
            [
                directory("package/lib"),
                regular_file("package/package.json", package_json),
                regular_file("package/z.txt", zeta, mode=0o644),
                regular_file("package/lib/a.bin", alpha, mode=0o600),
            ]
        )

        result = snapshot_package_tar.snapshot(tarball)

        self.assertEqual(result["schema"], "browsergrad.packed-package-snapshot@1")
        self.assertEqual(result["compressedBytes"], tarball.stat().st_size)
        self.assertEqual(result["memberCount"], 4)
        self.assertEqual(
            result["totalFileBytes"], len(alpha) + len(package_json) + len(zeta)
        )
        self.assertEqual(result["packageName"], "@unlocalhosted/test-package")
        self.assertEqual(result["packageVersion"], "1.2.3")
        self.assertEqual(
            result["entries"],
            [
                (
                    "lib/a.bin",
                    f"file:600:{len(alpha)}:{hashlib.sha512(alpha).hexdigest()}",
                ),
                (
                    "package.json",
                    f"file:644:{len(package_json)}:{hashlib.sha512(package_json).hexdigest()}",
                ),
                (
                    "z.txt",
                    f"file:644:{len(zeta)}:{hashlib.sha512(zeta).hexdigest()}",
                ),
            ],
        )

    def test_rejects_parent_traversal(self) -> None:
        self.assert_snapshot_rejected(
            [regular_file("package/../escape.txt", b"escape")],
            "non-canonical archive path",
        )

    def test_rejects_absolute_path(self) -> None:
        self.assert_snapshot_rejected(
            [regular_file("/package/escape.txt", b"escape")],
            "absolute archive path",
        )

    def test_rejects_links_and_special_files(self) -> None:
        cases = {
            "symbolic link": special_member(
                "package/symlink", tarfile.SYMTYPE, linkname="target"
            ),
            "hard link": special_member(
                "package/hardlink", tarfile.LNKTYPE, linkname="package/target"
            ),
            "character device": special_member("package/character", tarfile.CHRTYPE),
            "block device": special_member("package/block", tarfile.BLKTYPE),
            "fifo": special_member("package/fifo", tarfile.FIFOTYPE),
        }
        for label, member in cases.items():
            with self.subTest(label=label):
                self.assert_snapshot_rejected(
                    [member],
                    "extension metadata, links, and special files are not allowed",
                )

    def test_rejects_symlinked_tarball_input(self) -> None:
        tarball = self.write_tarball(
            [regular_file("package/package.json", b'{"name":"pkg","version":"1.0.0"}')]
        )
        link = self.root / "linked.tgz"
        link.symlink_to(tarball)
        with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
            snapshot_package_tar.snapshot(link)

    def test_rejects_duplicate_canonical_path(self) -> None:
        self.assert_snapshot_rejected(
            [
                regular_file("package/data.txt", b"first"),
                regular_file("package/data.txt", b"second"),
            ],
            "duplicate archive member: package/data.txt",
        )

    def test_rejects_file_as_ancestor_in_both_member_orders(self) -> None:
        cases = [
            [
                regular_file("package/data", b"file"),
                regular_file("package/data/child", b"child"),
            ],
            [
                regular_file("package/data/child", b"child"),
                regular_file("package/data", b"file"),
            ],
        ]
        for members in cases:
            with self.subTest(order=[member.name for member, _payload in members]):
                self.assert_snapshot_rejected(members, "archive file")

    def test_rejects_unicode_nfc_and_casefold_path_collisions(self) -> None:
        cases = [
            [
                regular_file("package/README.txt", b"upper"),
                regular_file("package/readme.txt", b"lower"),
            ],
            [
                regular_file(
                    "package/caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", b"nfc"
                ),
                regular_file(
                    "package/cafe\N{COMBINING ACUTE ACCENT}.txt", b"nfd"
                ),
            ],
        ]
        for members in cases:
            with self.subTest(paths=[member.name for member, _payload in members]):
                tarball = self.write_tarball(members, archive_format=tarfile.GNU_FORMAT)
                with self.assertRaisesRegex(
                    ValueError, "Unicode NFC/casefold normalization"
                ):
                    snapshot_package_tar.snapshot(tarball)

    def test_rejects_portable_normalized_file_ancestor(self) -> None:
        tarball = self.write_tarball(
            [
                regular_file("package/Data", b"file"),
                regular_file("package/data/child", b"child"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "portable-normalized ancestor"):
            snapshot_package_tar.snapshot(tarball)

    def test_rejects_member_outside_package_root(self) -> None:
        self.assert_snapshot_rejected(
            [regular_file("other/data.txt", b"outside")],
            "archive member outside package root",
        )

    def test_requires_valid_package_identity_manifest(self) -> None:
        cases = [
            (
                "missing",
                [regular_file("package/data.txt", b"x")],
                "has no package/package.json",
            ),
            (
                "invalid UTF-8",
                [regular_file("package/package.json", b"\xff")],
                "not valid UTF-8 JSON",
            ),
            (
                "non-object",
                [regular_file("package/package.json", b"[]")],
                "must contain an object",
            ),
            (
                "missing name",
                [regular_file("package/package.json", b'{"version":"1.0.0"}')],
                "non-empty string name",
            ),
            (
                "missing version",
                [regular_file("package/package.json", b'{"name":"pkg"}')],
                "non-empty string version",
            ),
        ]
        for label, members, message in cases:
            with self.subTest(label=label):
                self.assert_snapshot_rejected(members, message)

    def test_enforces_per_file_size_bound(self) -> None:
        tarball = self.write_tarball([regular_file("package/data.bin", b"1234")])
        with mock.patch.object(snapshot_package_tar, "MAX_FILE_BYTES", 3):
            with self.assertRaisesRegex(ValueError, "archive member exceeds 3 bytes"):
                snapshot_package_tar.snapshot(tarball)

    def test_enforces_total_file_size_bound(self) -> None:
        tarball = self.write_tarball(
            [
                regular_file("package/one.bin", b"123"),
                regular_file("package/two.bin", b"456"),
            ]
        )
        with mock.patch.object(snapshot_package_tar, "MAX_TOTAL_FILE_BYTES", 5):
            with self.assertRaisesRegex(ValueError, "archive expands beyond 5 bytes"):
                snapshot_package_tar.snapshot(tarball)

    def test_enforces_member_count_bound(self) -> None:
        tarball = self.write_tarball(
            [
                regular_file("package/one", b"1"),
                regular_file("package/two", b"2"),
                regular_file("package/three", b"3"),
            ]
        )
        with mock.patch.object(snapshot_package_tar, "MAX_MEMBER_COUNT", 2):
            with self.assertRaisesRegex(ValueError, "archive exceeds 2 members"):
                snapshot_package_tar.snapshot(tarball)

    def test_enforces_utf8_path_byte_bound(self) -> None:
        tarball = self.write_tarball(
            [
                regular_file(
                    "package/" + "\N{LATIN SMALL LETTER E WITH ACUTE}" * 6 + ".txt",
                    b"x",
                )
            ],
            archive_format=tarfile.GNU_FORMAT,
        )
        with mock.patch.object(snapshot_package_tar, "MAX_PATH_UTF8_BYTES", 20):
            with self.assertRaisesRegex(
                ValueError, "archive path exceeds UTF-8 byte limit"
            ):
                snapshot_package_tar.snapshot(tarball)

    def test_enforces_path_depth_and_component_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 64 components"):
            snapshot_package_tar.validated_parts(
                "package/" + "/".join(["a"] * 64)
            )

        with self.assertRaisesRegex(ValueError, "component exceeds 255"):
            snapshot_package_tar.validated_parts("package/" + ("a" * 256))

    def test_enforces_raw_decompressed_tar_bound(self) -> None:
        tarball = self.write_tarball(
            [
                regular_file(
                    "package/package.json", b'{"name":"pkg","version":"1.0.0"}'
                )
            ]
        )
        with mock.patch.object(
            snapshot_package_tar, "MAX_DECOMPRESSED_TAR_BYTES", 1_024
        ):
            with self.assertRaisesRegex(ValueError, "decompressed tar exceeds 1024"):
                snapshot_package_tar.snapshot(tarball)

    def test_enforces_decompressed_bound_after_tar_end_marker(self) -> None:
        tarball = self.write_tarball(
            [
                regular_file(
                    "package/package.json", b'{"name":"pkg","version":"1.0.0"}'
                )
            ]
        )
        raw_tar = gzip.decompress(tarball.read_bytes())
        tarball.write_bytes(gzip.compress(raw_tar + (b"\0" * 4_096)))
        with mock.patch.object(
            snapshot_package_tar,
            "MAX_DECOMPRESSED_TAR_BYTES",
            len(raw_tar) + 1_024,
        ):
            with self.assertRaisesRegex(ValueError, "decompressed tar exceeds"):
                snapshot_package_tar.snapshot(tarball)

    def test_rejects_compressed_pax_metadata_bomb_before_expansion(self) -> None:
        member, payload = regular_file(
            "package/package.json", b'{"name":"pkg","version":"1.0.0"}'
        )
        metadata_bytes = 2 * 1024 * 1024
        member.pax_headers = {"comment": "A" * metadata_bytes}
        tarball = self.write_tarball(
            [(member, payload)], archive_format=tarfile.PAX_FORMAT
        )
        self.assertLess(tarball.stat().st_size, metadata_bytes // 100)

        with mock.patch.object(
            snapshot_package_tar, "MAX_DECOMPRESSED_TAR_BYTES", 1_024
        ):
            with self.assertRaisesRegex(
                ValueError,
                "extension metadata, links, and special files are not allowed",
            ):
                snapshot_package_tar.snapshot(tarball)


if __name__ == "__main__":
    unittest.main()
