import hashlib
import json
import os
import re
import stat
import sys

expected = json.load(sys.stdin)


def refuse(code):
    sys.exit(code)


def identity(item):
    return (str(item.st_dev), str(item.st_ino))


def same_identity(item, value):
    return identity(item) == (value["device"], value["inode"])


def open_parent(relative):
    parts = relative.split("/")
    descriptor = os.dup(3)
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except Exception:
        os.close(descriptor)
        raise


def read_regular(relative, expected_item, maximum=None):
    parent, name = open_parent(relative)
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or identity(before) != expected_item["identity"]:
                refuse(63)
            if maximum is not None and before.st_size > maximum:
                refuse(63)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if identity(after) != identity(before) or after.st_size != before.st_size:
                refuse(63)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


parent_root = os.fstat(4)
if identity(parent_root) != (expected["rootIdentity"]["device"], expected["rootIdentity"]["inode"]):
    refuse(60)
opened = os.fstat(3)
if not stat.S_ISDIR(opened.st_mode) or not same_identity(opened, expected["candidateIdentity"]):
    refuse(60)
if opened.st_uid != expected["currentUid"] or stat.S_IMODE(opened.st_mode) != 0o700:
    refuse(60)
try:
    named = os.stat(expected["basename"], dir_fd=4, follow_symlinks=False)
except (FileNotFoundError, NotADirectoryError):
    refuse(60)
if not stat.S_ISDIR(named.st_mode) or not same_identity(named, expected["candidateIdentity"]):
    refuse(60)

files = {}
directories = {}
children = {}
entry_count = 0


def inspect_directory(descriptor, prefix):
    global entry_count
    names = sorted(os.listdir(descriptor))
    direct = {}
    for name in names:
        entry_count += 1
        if entry_count > expected["entryLimit"]:
            refuse(61)
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if item.st_uid != expected["currentUid"]:
            refuse(61)
        relative = name if prefix == "" else prefix + "/" + name
        item_identity = identity(item)
        if stat.S_ISDIR(item.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                opened_child = os.fstat(child)
                if identity(opened_child) != item_identity:
                    refuse(61)
                directories[relative] = {"identity": item_identity}
                direct[name] = {"kind": "directory", "identity": item_identity}
                inspect_directory(child, relative)
            finally:
                os.close(child)
        elif stat.S_ISREG(item.st_mode):
            files[relative] = {
                "identity": item_identity,
                "mode": stat.S_IMODE(item.st_mode),
                "size": item.st_size,
            }
            direct[name] = {"kind": "file", "identity": item_identity}
        else:
            refuse(61)
    children[prefix] = direct


inspect_directory(3, "")
snapshot_marker = expected["snapshotMarker"]
reservation_marker = expected["reservationMarker"]
failed_reservation_marker = expected["failedReservationMarker"]
allowed_markers = {
    snapshot_marker,
    reservation_marker,
    failed_reservation_marker,
    ".git/" + reservation_marker,
    ".git/" + failed_reservation_marker,
}
markers = sorted(set(files).intersection(allowed_markers))
if len(markers) != 1:
    refuse(62)
marker_relative = markers[0]
marker_item = files[marker_relative]
if marker_item["mode"] != 0o600:
    refuse(62)
try:
    marker_text = read_regular(marker_relative, marker_item, 8192).decode("utf-8")
    marker = json.loads(marker_text)
except (UnicodeDecodeError, json.JSONDecodeError):
    refuse(62)
required_keys = {
    "kind", "version", "type", "ownerUid", "ownerPid", "token",
    "targetPathSha256", "rootDevice", "rootInode", "commit", "basename",
    "createdAtMs",
}
if set(marker) != required_keys:
    refuse(62)
canonical = json.dumps(marker, separators=(",", ":"), ensure_ascii=True) + "\n"
if marker_text != canonical:
    refuse(62)
if marker.get("kind") != "browsergrad-corpus-residue-owner" or marker.get("version") != 1:
    refuse(62)
if marker.get("type") != expected["residueType"] or marker.get("ownerUid") != expected["currentUid"]:
    refuse(62)
if not isinstance(marker.get("ownerPid"), int) or marker["ownerPid"] <= 0:
    refuse(62)
if not isinstance(marker.get("token"), str) or re.fullmatch(r"[0-9a-f]{64}", marker["token"]) is None:
    refuse(62)
if marker.get("targetPathSha256") != expected["targetPathSha256"]:
    refuse(62)
if (marker.get("rootDevice"), marker.get("rootInode")) != (
    expected["rootIdentity"]["device"], expected["rootIdentity"]["inode"]
):
    refuse(62)
if marker.get("basename") != expected["basename"]:
    refuse(62)
if not isinstance(marker.get("createdAtMs"), int) or marker["createdAtMs"] <= 0:
    refuse(62)
if marker["createdAtMs"] > expected["nowMs"] + 300000:
    refuse(62)

if expected["residueType"] == "snapshot":
    if marker_relative != snapshot_marker or marker.get("commit") != expected["commit"]:
        refuse(62)
    expected_files = {item["relative"]: item["objectId"] for item in expected["snapshotFiles"]}
    expected_directories = set()
    for relative in expected_files:
        parts = relative.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            expected_directories.add("/".join(parts[:index]))
    actual_files = set(files) - {marker_relative}
    if not actual_files.issubset(expected_files) or not set(directories).issubset(expected_directories):
        refuse(64)
    for relative in sorted(actual_files):
        data = read_regular(relative, files[relative])
        digest = hashlib.sha1()
        digest.update(b"blob " + str(len(data)).encode() + b"\0")
        digest.update(data)
        if digest.hexdigest() != expected_files[relative]:
            refuse(64)
else:
    if marker.get("commit") is not None:
        refuse(62)
    valid_marker_locations = {
        reservation_marker,
        failed_reservation_marker,
        ".git/" + reservation_marker,
        ".git/" + failed_reservation_marker,
    }
    if marker_relative not in valid_marker_locations:
        refuse(62)
    if marker_relative.startswith(".git/"):
        if ".git" not in directories:
            refuse(64)
        if any(relative != ".git" and not relative.startswith(".git/") for relative in directories):
            refuse(64)
        if any(not relative.startswith(".git/") for relative in files):
            refuse(64)
    elif set(files) != {marker_relative} or directories:
        refuse(64)

failed_marker = marker_relative.endswith(failed_reservation_marker)
if not failed_marker:
    try:
        os.kill(marker["ownerPid"], 0)
        refuse(65)
    except ProcessLookupError:
        pass
    except PermissionError:
        refuse(65)


def delete_contents(descriptor, prefix):
    expected_children = children[prefix]
    if set(os.listdir(descriptor)) != set(expected_children):
        refuse(66)
    for name in sorted(expected_children):
        expected_child = expected_children[name]
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if identity(item) != expected_child["identity"]:
            refuse(66)
        if expected_child["kind"] == "directory":
            if not stat.S_ISDIR(item.st_mode):
                refuse(66)
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                if identity(os.fstat(child)) != expected_child["identity"]:
                    refuse(66)
                child_prefix = name if prefix == "" else prefix + "/" + name
                delete_contents(child, child_prefix)
            finally:
                os.close(child)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or identity(current) != expected_child["identity"]:
                refuse(66)
            os.rmdir(name, dir_fd=descriptor)
        else:
            if not stat.S_ISREG(item.st_mode):
                refuse(66)
            os.unlink(name, dir_fd=descriptor)


delete_contents(3, "")
named = os.stat(expected["basename"], dir_fd=4, follow_symlinks=False)
if not stat.S_ISDIR(named.st_mode) or not same_identity(named, expected["candidateIdentity"]):
    refuse(67)
os.rmdir(expected["basename"], dir_fd=4)
