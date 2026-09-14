"""공통 기사 JSON 파일을 한 건씩 읽는다. JSON, JSONL, 디렉토리를 받는다."""

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson"}


@dataclass
class RawRecord:
    """파싱 직후의 레코드. record 가 None 이면 error 에 사유가 있다."""

    record: dict | None
    source_ref: str
    error: str | None = None


def expand_paths(paths: Iterable[Path | str]):
    """파일·디렉토리 목록을 지원 확장자 파일 목록으로 펼친다."""
    files = set()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.add(f.resolve())
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                log.warning(f"지원하지 않는 확장자, 건너뜀: {p}")
                continue
            files.add(p.resolve())
        else:
            log.warning(f"경로가 존재하지 않음, 건너뜀: {p}")
    return sorted(files)


def iter_records(paths: Iterable[Path | str]):
    for f in expand_paths(paths):
        if f.suffix.lower() == ".json":
            yield from _iter_json(f)
        else:
            yield from _iter_jsonl(f)


def _iter_jsonl(f):
    # utf-8-sig: Windows 메모장이나 엑셀이 저장한 파일은 BOM 이 붙는다. utf-8 로
    # 읽으면 첫 키가 '﻿schema_version' 이 되어 첫 줄이 통째로 리젝트된다.
    with f.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            ref = f"{f}:{lineno}"
            # 깨진 줄 하나 때문에 파일 전체를 버리지 않는다. DLQ 와 같은 사고방식.
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                yield RawRecord(None, ref, f"invalid JSON: {e.msg} (col {e.colno})")
                continue
            if not isinstance(parsed, dict):
                yield RawRecord(None, ref, "JSON line is not an object")
                continue
            yield RawRecord(parsed, ref)


def _iter_json(f):
    try:
        parsed = json.loads(f.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        yield RawRecord(None, str(f), f"invalid JSON: {e.msg} (line {e.lineno}, col {e.colno})")
        return
    if isinstance(parsed, dict):
        yield RawRecord(parsed, str(f))
    elif isinstance(parsed, list):
        for i, item in enumerate(parsed):
            ref = f"{f}[{i}]"
            if isinstance(item, dict):
                yield RawRecord(item, ref)
            else:
                yield RawRecord(None, ref, "array item is not an object")
    else:
        yield RawRecord(None, str(f), "top-level JSON is neither object nor array")
