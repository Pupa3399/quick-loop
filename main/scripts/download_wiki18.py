from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_MIRROR = "https://hf-mirror.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_REPO = "PeterJinGo/wiki-18-e5-index"
INDEX_REVISION = "a4d31160a035f30764604f4827cd8f1d0315eb86"
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
EXPECTED_SHA256 = {
    "part_aa": "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023",
    "part_ab": "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11",
    "wiki-18.jsonl.gz": "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db",
}
CORPUS_MEMBER_SUFFIX = "/wiki_dump.jsonl"
CORPUS_UNCOMPRESSED_SIZE = 14_393_573_105


def require_hf_mirror() -> None:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")


def copy_stream(source: Path, destination: object) -> None:
    with source.open("rb") as handle:
        shutil.copyfileobj(handle, destination, length=16 * 1024 * 1024)


def verify_sha256(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = EXPECTED_SHA256[path.name]
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    print(f"verified_sha256={actual} file={path}")


def extract_corpus(archive_path: Path, corpus_path: Path) -> None:
    if corpus_path.is_file() and corpus_path.stat().st_size == CORPUS_UNCOMPRESSED_SIZE:
        return
    temporary = corpus_path.with_suffix(".jsonl.incomplete")
    found = False
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(CORPUS_MEMBER_SUFFIX):
                continue
            if found:
                raise RuntimeError(f"Archive contains multiple {CORPUS_MEMBER_SUFFIX} members")
            found = True
            if member.size != CORPUS_UNCOMPRESSED_SIZE:
                raise RuntimeError(
                    "Unexpected corpus size: "
                    f"expected {CORPUS_UNCOMPRESSED_SIZE}, got {member.size}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read corpus member {member.name}")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    if not found:
        raise RuntimeError(f"Archive does not contain {CORPUS_MEMBER_SUFFIX}")
    if temporary.stat().st_size != CORPUS_UNCOMPRESSED_SIZE:
        raise RuntimeError(f"Extracted corpus is truncated: {temporary}")
    temporary.replace(corpus_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Search-R1 Wiki18 E5 artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "main/datasets/wiki18",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "main/datasets/huggingface",
    )
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()
    require_hf_mirror()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        Path(
            hf_hub_download(
                INDEX_REPO,
                filename=name,
                repo_type="dataset",
                revision=INDEX_REVISION,
                local_dir=args.output_dir,
                cache_dir=args.cache_dir,
                endpoint=HF_MIRROR,
            )
        )
        for name in ("part_aa", "part_ab")
    ]
    for part in parts:
        verify_sha256(part)
    index_path = args.output_dir / "e5_Flat.index"
    expected_size = sum(part.stat().st_size for part in parts)
    if not index_path.exists() or index_path.stat().st_size != expected_size:
        temporary = index_path.with_suffix(".index.incomplete")
        with temporary.open("wb") as output:
            for part in parts:
                copy_stream(part, output)
        temporary.replace(index_path)

    corpus_download = Path(
        hf_hub_download(
            CORPUS_REPO,
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            revision=CORPUS_REVISION,
            local_dir=args.output_dir,
            cache_dir=args.cache_dir,
            endpoint=HF_MIRROR,
        )
    )
    archive_path = args.output_dir / "wiki-18.jsonl.gz"
    if corpus_download != archive_path:
        shutil.copy2(corpus_download, archive_path)
    verify_sha256(archive_path)
    corpus_path = args.output_dir / "wiki-18.jsonl"
    extract_corpus(archive_path, corpus_path)
    if not args.keep_parts:
        for part in parts:
            part.unlink(missing_ok=True)
    print(f"index={index_path} bytes={index_path.stat().st_size}")
    print(f"corpus={corpus_path} bytes={corpus_path.stat().st_size}")


if __name__ == "__main__":
    main()
