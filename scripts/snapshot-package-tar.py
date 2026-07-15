#!/usr/bin/env python3
"""Emit a bounded canonical snapshot of an npm package tarball without extracting it."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tarfile
import unicodedata
from pathlib import Path
from typing import NoReturn

MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_MEMBER_COUNT = 10_000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 256 * 1024 * 1024
MAX_PATH_UTF8_BYTES = 1_024
MAX_PATH_COMPONENTS = 64
MAX_COMPONENT_UTF8_BYTES = 255
MAX_DECOMPRESSED_TAR_BYTES = 320 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
TAR_BLOCK_BYTES = 512
ALLOWED_RAW_MEMBER_TYPES = frozenset({b"\0", tarfile.REGTYPE, tarfile.DIRTYPE})


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def validated_parts(name: str) -> tuple[str, ...]:
    if "\x00" in name or "\\" in name:
        fail(f"unsafe archive path: {name!r}")
    if len(name.encode("utf-8")) > MAX_PATH_UTF8_BYTES:
        fail("archive path exceeds UTF-8 byte limit")
    if name.startswith("/"):
        fail(f"absolute archive path: {name!r}")
    parts = tuple(name.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        fail(f"non-canonical archive path: {name!r}")
    if len(parts) > MAX_PATH_COMPONENTS:
        fail(f"archive path exceeds {MAX_PATH_COMPONENTS} components")
    if any(len(part.encode("utf-8")) > MAX_COMPONENT_UTF8_BYTES for part in parts):
        fail(f"archive path component exceeds {MAX_COMPONENT_UTF8_BYTES} UTF-8 bytes")
    if parts[0] != "package":
        fail(f"archive member outside package root: {name!r}")
    return parts


def portable_path_key(parts: tuple[str, ...]) -> str:
    normalized_parts = []
    for part in parts:
        folded = unicodedata.normalize("NFC", part).casefold()
        normalized_parts.append(unicodedata.normalize("NFC", folded))
    return "/".join(normalized_parts)


def parent_paths(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("/".join(parts[:end]) for end in range(1, len(parts)))


def parse_raw_tar_octal(field: bytes, label: str) -> int:
    if field and field[0] & 0x80:
        fail(f"base-256 {label} is not allowed in package tarballs")
    value = field.rstrip(b"\0 ").lstrip(b" ")
    if not value:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in value):
        fail(f"invalid octal {label} in raw tar header")
    return int(value, 8)


def read_bounded_exact(
    source: gzip.GzipFile, length: int, decompressed_bytes: int
) -> tuple[bytes, int]:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = source.read(min(READ_CHUNK_BYTES, length - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
        decompressed_bytes += len(chunk)
        if decompressed_bytes > MAX_DECOMPRESSED_TAR_BYTES:
            fail(
                f"decompressed tar exceeds {MAX_DECOMPRESSED_TAR_BYTES} bytes"
            )
    return bytes(chunks), decompressed_bytes


def consume_bounded_exact(
    source: gzip.GzipFile, length: int, decompressed_bytes: int
) -> tuple[int, int]:
    bytes_read = 0
    while bytes_read < length:
        chunk = source.read(min(READ_CHUNK_BYTES, length - bytes_read))
        if not chunk:
            break
        bytes_read += len(chunk)
        decompressed_bytes += len(chunk)
        if decompressed_bytes > MAX_DECOMPRESSED_TAR_BYTES:
            fail(
                f"decompressed tar exceeds {MAX_DECOMPRESSED_TAR_BYTES} bytes"
            )
    return bytes_read, decompressed_bytes


def preflight_raw_tar(tarball: Path) -> tuple[int, int]:
    """Reject unsafe raw headers before tarfile can parse extension payloads."""
    member_count = 0
    total_file_bytes = 0
    decompressed_bytes = 0
    with tarball.open("rb") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as archive:
            while True:
                header, decompressed_bytes = read_bounded_exact(
                    archive, TAR_BLOCK_BYTES, decompressed_bytes
                )
                if len(header) != TAR_BLOCK_BYTES:
                    fail("raw tar ended before the end-of-archive marker")
                if header == b"\0" * TAR_BLOCK_BYTES:
                    second, decompressed_bytes = read_bounded_exact(
                        archive, TAR_BLOCK_BYTES, decompressed_bytes
                    )
                    if second != b"\0" * TAR_BLOCK_BYTES:
                        fail("raw tar must end with two zero blocks")
                    while True:
                        trailing = archive.read(READ_CHUNK_BYTES)
                        if not trailing:
                            break
                        decompressed_bytes += len(trailing)
                        if decompressed_bytes > MAX_DECOMPRESSED_TAR_BYTES:
                            fail(
                                f"decompressed tar exceeds {MAX_DECOMPRESSED_TAR_BYTES} bytes"
                            )
                        if trailing.strip(b"\0"):
                            fail("raw tar contains non-zero data after the end marker")
                    return member_count, total_file_bytes

                stored_checksum = parse_raw_tar_octal(
                    header[148:156], "header checksum"
                )
                computed_checksum = (
                    sum(header[:148]) + (ord(" ") * 8) + sum(header[156:])
                )
                if stored_checksum != computed_checksum:
                    fail("raw tar header checksum mismatch")

                member_count += 1
                if member_count > MAX_MEMBER_COUNT:
                    fail(f"archive exceeds {MAX_MEMBER_COUNT} members")
                member_type = header[156:157]
                if member_type not in ALLOWED_RAW_MEMBER_TYPES:
                    fail(
                        f"unsupported raw tar header type {member_type!r}; "
                        "extension metadata, links, and special files are not allowed"
                    )
                member_size = parse_raw_tar_octal(header[124:136], "member size")
                if member_type == tarfile.DIRTYPE:
                    if member_size != 0:
                        fail("raw tar directory header cannot carry a payload")
                else:
                    if member_size > MAX_FILE_BYTES:
                        fail(
                            f"archive member exceeds {MAX_FILE_BYTES} bytes in raw tar header"
                        )
                    total_file_bytes += member_size
                    if total_file_bytes > MAX_TOTAL_FILE_BYTES:
                        fail(f"archive expands beyond {MAX_TOTAL_FILE_BYTES} bytes")

                padded_size = (
                    (member_size + TAR_BLOCK_BYTES - 1)
                    // TAR_BLOCK_BYTES
                    * TAR_BLOCK_BYTES
                )
                payload_bytes, decompressed_bytes = consume_bounded_exact(
                    archive, padded_size, decompressed_bytes
                )
                if payload_bytes != padded_size:
                    fail("raw tar member payload is truncated")


def record_member_path(
    parts: tuple[str, ...],
    is_directory: bool,
    seen: set[str],
    portable_seen: dict[str, str],
    file_paths: set[str],
    ancestor_paths: set[str],
    portable_file_paths: set[str],
    portable_ancestor_paths: set[str],
) -> None:
    canonical_name = "/".join(parts)
    if canonical_name in seen:
        fail(f"duplicate archive member: {canonical_name}")

    portable_name = portable_path_key(parts)
    portable_collision = portable_seen.get(portable_name)
    if portable_collision is not None:
        fail(
            "archive paths collide after Unicode NFC/casefold normalization: "
            f"{portable_collision!r} and {canonical_name!r}"
        )

    parents = parent_paths(parts)
    portable_parts = tuple(portable_path_key((part,)) for part in parts)
    portable_parents = parent_paths(portable_parts)
    for parent in parents:
        if parent in file_paths:
            fail(f"archive file is an ancestor of another member: {parent}")
    for parent in portable_parents:
        if parent in portable_file_paths:
            fail(
                "archive file is a portable-normalized ancestor of another member: "
                f"{parent}"
            )
    if not is_directory:
        if canonical_name in ancestor_paths:
            fail(
                "archive file would replace an existing member ancestor: "
                f"{canonical_name}"
            )
        if portable_name in portable_ancestor_paths:
            fail(
                "archive file would replace a portable-normalized member ancestor: "
                f"{canonical_name}"
            )

    seen.add(canonical_name)
    portable_seen[portable_name] = canonical_name
    ancestor_paths.update(parents)
    portable_ancestor_paths.update(portable_parents)
    if not is_directory:
        file_paths.add(canonical_name)
        portable_file_paths.add(portable_name)


def snapshot(tarball: Path) -> dict[str, object]:
    stat = tarball.lstat()
    if not tarball.is_file() or tarball.is_symlink():
        fail("tarball must be a regular non-symlink file")
    if stat.st_size > MAX_COMPRESSED_BYTES:
        fail(f"compressed tarball exceeds {MAX_COMPRESSED_BYTES} bytes")

    raw_member_count, raw_total_file_bytes = preflight_raw_tar(tarball)

    seen: set[str] = set()
    portable_seen: dict[str, str] = {}
    file_paths: set[str] = set()
    ancestor_paths: set[str] = set()
    portable_file_paths: set[str] = set()
    portable_ancestor_paths: set[str] = set()
    entries: list[tuple[str, str]] = []
    total_file_bytes = 0
    member_count = 0
    package_json_bytes: bytes | None = None
    with tarfile.open(tarball, mode="r:gz") as archive:
        for member in archive:
            member_count += 1
            if member_count > MAX_MEMBER_COUNT:
                fail(f"archive exceeds {MAX_MEMBER_COUNT} members")
            parts = validated_parts(member.name)
            canonical_name = "/".join(parts)
            is_directory = member.isdir()
            if not is_directory and not member.isfile():
                fail(
                    f"archive member is not a regular file or directory: {canonical_name}"
                )
            if not is_directory and len(parts) == 1:
                fail("package root cannot be a regular file")
            record_member_path(
                parts,
                is_directory,
                seen,
                portable_seen,
                file_paths,
                ancestor_paths,
                portable_file_paths,
                portable_ancestor_paths,
            )

            if is_directory:
                continue
            if member.size < 0 or member.size > MAX_FILE_BYTES:
                fail(f"archive member exceeds {MAX_FILE_BYTES} bytes: {canonical_name}")
            total_file_bytes += member.size
            if total_file_bytes > MAX_TOTAL_FILE_BYTES:
                fail(f"archive expands beyond {MAX_TOTAL_FILE_BYTES} bytes")

            relative_path = "/".join(parts[1:])
            source = archive.extractfile(member)
            if source is None:
                fail(f"archive member has no readable payload: {canonical_name}")
            digest = hashlib.sha512()
            captured = bytearray() if relative_path == "package.json" else None
            bytes_read = 0
            while True:
                chunk = source.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > member.size:
                    fail(f"archive member exceeds declared size: {canonical_name}")
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
            if bytes_read != member.size:
                fail(f"archive member size mismatch: {canonical_name}")
            if captured is not None:
                package_json_bytes = bytes(captured)
            mode = f"{member.mode & 0o777:03o}"
            entries.append(
                (
                    relative_path,
                    f"file:{mode}:{member.size}:{digest.hexdigest()}",
                )
            )

    if member_count != raw_member_count or total_file_bytes != raw_total_file_bytes:
        fail("raw tar preflight and parsed archive disagree")

    if member_count == 0 or not entries:
        fail("archive has no package files")
    if package_json_bytes is None:
        fail("archive has no package/package.json")
    try:
        package_json = json.loads(package_json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"package/package.json is not valid UTF-8 JSON: {error}")
    if not isinstance(package_json, dict):
        fail("package/package.json must contain an object")
    package_name = package_json.get("name")
    package_version = package_json.get("version")
    if not isinstance(package_name, str) or not package_name:
        fail("package/package.json must contain a non-empty string name")
    if not isinstance(package_version, str) or not package_version:
        fail("package/package.json must contain a non-empty string version")
    entries.sort(key=lambda entry: entry[0].encode("utf-8"))
    return {
        "schema": "browsergrad.packed-package-snapshot@1",
        "compressedBytes": stat.st_size,
        "memberCount": member_count,
        "totalFileBytes": total_file_bytes,
        "packageName": package_name,
        "packageVersion": package_version,
        "entries": entries,
    }


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: snapshot-package-tar.py <package.tgz>")
    result = snapshot(Path(sys.argv[1]))
    json.dump(result, sys.stdout, ensure_ascii=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, tarfile.TarError, ValueError) as error:
        print(f"snapshot-package-tar: {error}", file=sys.stderr)
        raise SystemExit(1)
