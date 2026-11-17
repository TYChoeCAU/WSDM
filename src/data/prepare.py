from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from common.hashing import sha256


ARCHIVE_SHA256 = {
    "yelp2018.zip": "832aa897f5713d87befce01621db28389f8d0102bfa02f6b6f64f3ff28d70b74",
    "amazon-books.zip": "9428df10d252f5199dd32730cd0b835e9a3254afcd93071b6beb1e7137ece9e0",
    "ml-1m.zip": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
}


def verify_archive(path: Path, expected_name: str | None = None) -> None:
    expected = ARCHIVE_SHA256[expected_name or path.name]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"archive checksum mismatch for {path.name}: {actual}")


def _safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            destination = (target / Path(*relative.parts)).resolve()
            if target not in destination.parents and destination != target:
                raise RuntimeError(f"archive member escapes target: {member.filename}")
        handle.extractall(target)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing prepared file has wrong size: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _date(value: str) -> int:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return calendar.timegm(datetime.strptime(value.strip(), fmt).timetuple())
        except ValueError:
            pass
    raise ValueError(value)


def prepare_yelp(root: Path) -> None:
    source, target = root / "yelp2018", root / "yelp2018-paper"
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ("inter", "item"):
        _link_or_copy(source / f"yelp2018.{suffix}", target / f"yelp2018-paper.{suffix}")
    src, dst = source / "yelp2018.user", target / "yelp2018-paper.user"
    if dst.is_file() and dst.stat().st_size:
        with dst.open(encoding="utf-8") as handle:
            header = handle.readline().rstrip().split("\t")
            first = handle.readline().rstrip().split("\t")
        try:
            float(first[header.index("yelping_since:float")])
            return
        except (ValueError, IndexError):
            pass
    with src.open(encoding="utf-8") as inp:
        header_line = inp.readline()
        index = header_line.rstrip().split("\t").index("yelping_since:float")
        fd, temporary = tempfile.mkstemp(prefix=".yelp-user-", dir=target, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(header_line)
                for line in inp:
                    fields = line.rstrip("\n").split("\t")
                    fields[index] = str(_date(fields[index]))
                    out.write("\t".join(fields) + "\n")
            os.replace(temporary, dst)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _convert_ml1m(raw: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with (raw / "ratings.dat").open(encoding="latin-1") as inp, (target / "ml-1m.inter").open("w", encoding="utf-8") as out:
        out.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for line in inp:
            user, item, rating, timestamp = line.rstrip("\n").split("::")
            out.write(f"{user}\t{item}\t{rating}\t{timestamp}\n")
    with (raw / "users.dat").open(encoding="latin-1") as inp, (target / "ml-1m.user").open("w", encoding="utf-8") as out:
        out.write("user_id:token\tage:token\tgender:token\toccupation:token\tzip_code:token\n")
        for line in inp:
            user, gender, age, occupation, zipcode = line.rstrip("\n").split("::")
            out.write(f"{user}\t{age}\t{gender}\t{occupation}\t{zipcode}\n")
    with (raw / "movies.dat").open(encoding="latin-1") as inp, (target / "ml-1m.item").open("w", encoding="utf-8") as out:
        out.write("item_id:token\tmovie_title:token_seq\trelease_year:token\tgenre:token_seq\n")
        for line in inp:
            item, title, genres = line.rstrip("\n").split("::")
            matched = re.match(r"^.*\((\d{4})\)$", title)
            if matched:
                year, title = matched.group(1), title[:-7]
            else:
                year = ""
            out.write(f"{item}\t{title.replace(chr(9), ' ')}\t{year}\t{genres.replace('|', ' ')}\n")


def _archive_for(archive_root: Path, dataset: str) -> Path:
    if dataset != "ml-1m":
        return archive_root / f"{dataset}.zip"
    for name in ("ml-1m.zip", "ml-1m-grouplens-raw.zip"):
        path = archive_root / name
        if path.is_file():
            return path
    return archive_root / "ml-1m.zip"


def _dataset_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def prepare(data_root: Path, archive_root: Path, datasets) -> list[dict]:
    data_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in datasets:
        base = "yelp2018-paper" if name == "yelp2018" else name
        effective = data_root / base
        suffixes = ("inter", "item") if name == "amazon-books" else ("inter", "item", "user")
        already_ready = all((effective / f"{base}.{suffix}").is_file() for suffix in suffixes)
        if already_ready:
            rows.append({
                "dataset": name, "archive": None, "archive_sha256": None,
                "input_mode": "preprocessed_atomic_files",
                "prepared_directory": str(effective.resolve()),
                "prepared_digest": _dataset_digest(effective), "status": "READY",
            })
            continue
        if name == "yelp2018" and all((data_root / "yelp2018" / f"yelp2018.{suffix}").is_file() for suffix in ("inter", "item", "user")):
            prepare_yelp(data_root)
            rows.append({
                "dataset": name, "archive": None, "archive_sha256": None,
                "input_mode": "preprocessed_yelp_atomic_files_with_date_conversion",
                "prepared_directory": str(effective.resolve()),
                "prepared_digest": _dataset_digest(effective), "status": "READY",
            })
            continue
        archive = _archive_for(archive_root, name)
        if not archive.is_file():
            raise FileNotFoundError(f"obtain the archive described in docs/DATA.md: {archive}")
        verify_archive(archive, "ml-1m.zip" if name == "ml-1m" else None)
        if name == "yelp2018":
            if not (data_root / "yelp2018" / "yelp2018.inter").is_file():
                _safe_extract(archive, data_root)
            prepare_yelp(data_root)
            effective = data_root / "yelp2018-paper"
        elif name == "amazon-books":
            effective = data_root / "amazon-books"
            effective.mkdir(parents=True, exist_ok=True)
            if not (effective / "amazon-books.inter").is_file():
                with tempfile.TemporaryDirectory(prefix="amazon-books-") as temporary:
                    unpacked = Path(temporary)
                    _safe_extract(archive, unpacked)
                    _link_or_copy(unpacked / "Amazon_Books.inter", effective / "amazon-books.inter")
                    _link_or_copy(unpacked / "Amazon_Books.item", effective / "amazon-books.item")
        elif name == "ml-1m":
            effective = data_root / "ml-1m"
            if not (effective / "ml-1m.inter").is_file():
                with tempfile.TemporaryDirectory(prefix="ml-1m-") as temporary:
                    unpacked = Path(temporary)
                    _safe_extract(archive, unpacked)
                    _convert_ml1m(unpacked / "ml-1m", effective)
        else:
            raise KeyError(name)
        required = [effective / f"{base}.{suffix}" for suffix in ("inter", "item")]
        if name != "amazon-books":
            required.append(effective / f"{base}.user")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"dataset preparation is incomplete: {missing}")
        rows.append({
            "dataset": name,
            "archive": archive.name,
            "archive_sha256": sha256(archive),
            "input_mode": "verified_public_archive",
            "prepared_directory": str(effective.resolve()),
            "prepared_digest": _dataset_digest(effective),
            "status": "READY",
        })
    (data_root / "preparation_manifest.json").write_text(
        json.dumps({"datasets": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows
