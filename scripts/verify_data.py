"""Verify existing originals against published manifests / IRIDIA archive CRCs."""

import argparse, hashlib, json, zipfile, time
import zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def physionet(root):
    root = Path(root)
    manifest = root / "SHA256SUMS.txt"
    results = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("unsafe checksum path")
        observed = digest(path) if path.is_file() else None
        results.append(
            {
                "path": name,
                "expected": expected,
                "observed": observed,
                "ok": observed == expected,
            }
        )
    return {
        "manifest_sha256": digest(manifest),
        "files": results,
        "ok": all(r["ok"] for r in results),
    }


def iridia(root, archive):
    expected = "81d58d21c9835a601cd9eb6c68bf9feb"
    observed = digest(archive, "md5")
    result = {"archive_md5": observed, "expected_md5": expected, "files": []}
    if observed != expected:
        result["ok"] = False
        return result
    # Compare extracted files with CRC32 entries from the authenticated archive.
    root = Path(root)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            pos = next(
                (i for i, p in enumerate(parts) if p.startswith("record_")), None
            )
            if pos is None:
                continue
            path = root.joinpath(*parts[pos:])
            crc = 0
            size = 0
            if not path.is_file():
                result["files"].append({"path": str(path), "ok": False})
                continue
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
                    size += len(chunk)
            result["files"].append(
                {
                    "path": str(path),
                    "ok": size == member.file_size and crc == member.CRC,
                }
            )
    result["ok"] = bool(result["files"]) and all(f["ok"] for f in result["files"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())["datasets"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {
            "SHDB": pool.submit(physionet, config["SHDB"]["source"]),
            "CPSC": pool.submit(physionet, config["CPSC"]["source"]),
            "IRIDIA": pool.submit(iridia, config["IRIDIA"]["source"], args.archive),
        }
        result = {}
        for k, job in jobs.items():
            result[k] = job.result()
            print(
                k,
                "verified",
                result[k]["ok"],
                "files",
                len(result[k]["files"]),
                flush=True,
            )
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if not all(v["ok"] for v in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
