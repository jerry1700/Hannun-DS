"""MinHash 서명 저장소 — 기사당 한 번만 계산해 실행 간 재사용한다 (S15P21E105-123).

15분 배정 실행이 창 8천 건의 서명(4-gram × 128 해시)을 매번 다시 만드는 데 ~50초를 쓰던
것을, 새 기사 수십 건만 계산하게 바꾸는 캐시. 정제본이 바뀌면(전처리 규칙 개정) 서명도
바뀌어야 하므로 정제본 해시를 함께 두고 다르면 다시 계산한다. 서명 설정(shingle_size·
num_perm)이 다른 행은 무시한다 — 설정을 바꾸면 자연히 전부 재계산된다.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

SIGNATURE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("published_date", pa.string()),
        # 서명을 만든 정제본의 해시 — 다르면 캐시 무효
        ("text_sha1", pa.string()),
        ("shingle_size", pa.int32()),
        ("num_perm", pa.int32()),
        # datasketch 의 해시 방식 — 라이브러리 판이 바뀌어 방식이 달라지면 캐시 무효
        ("scheme", pa.string()),
        # LSHEnsemble(containment) 질의에 필요한 4-gram 집합 크기
        ("shingle_count", pa.int32()),
        ("hashvalues", pa.list_(pa.uint64())),
        ("computed_at", pa.timestamp("us", tz="UTC")),
    ]
)


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SignatureStore:
    """<root>/dedup_sig/published_date=YYYY-MM-DD/sig.parquet — Gold 와 같은 발행일 파티션."""

    def __init__(self, root):
        self.root = Path(root)
        self.sig_dir = self.root / "dedup_sig"

    def read_cache(self, start_date: str, end_date: str, shingle_size: int, num_perm: int,
                   scheme: str):
        """범위의 서명을 article_id → (text_sha1, hashvalues, shingle_count) 로. 설정이 다른 행은 뺀다."""
        cache = {}
        for date_str in self.partition_dates():
            if date_str < start_date or date_str > end_date:
                continue
            table = pq.read_table(self.partition_path(date_str))
            for row in table.to_pylist():
                setting = (row["shingle_size"], row["num_perm"], row["scheme"])
                if setting != (shingle_size, num_perm, scheme):
                    continue
                cache[row["article_id"]] = (
                    row["text_sha1"],
                    np.array(row["hashvalues"], dtype=np.uint64),
                    row["shingle_count"],
                )
        return cache

    def merge_partition(self, published_date: str, fresh: dict, shingle_size: int, num_perm: int,
                        scheme: str):
        """새 서명을 그 날짜 파티션에 합친다 — 같은 article_id 는 새 것으로 바꾼다.

        fresh 는 article_id → (text_sha1, hashvalues, shingle_count). 파티션은 그 날짜 기사 수를
        넘어 자라지 않는다.
        """
        path = self.partition_path(published_date)
        rows = {}
        if path.exists():
            rows = {row["article_id"]: row for row in pq.read_table(path).to_pylist()}
        computed_at = datetime.now(timezone.utc)
        for article_id, (sha1, hashvalues, count) in fresh.items():
            rows[article_id] = {
                "article_id": article_id,
                "published_date": published_date,
                "text_sha1": sha1,
                "shingle_size": shingle_size,
                "num_perm": num_perm,
                "scheme": scheme,
                "shingle_count": count,
                "hashvalues": [int(v) for v in hashvalues],
                "computed_at": computed_at,
            }
        table = pa.Table.from_pylist(list(rows.values()), schema=SIGNATURE_SCHEMA)
        write_parquet_atomic(table.sort_by([("article_id", "ascending")]), path)
        return table.num_rows

    def partition_path(self, published_date: str):
        return self.sig_dir / f"published_date={published_date}" / "sig.parquet"

    def partition_dates(self):
        if not self.sig_dir.exists():
            return []
        return sorted(
            d.name.split("=", 1)[1] for d in self.sig_dir.iterdir()
            if d.is_dir() and d.name.startswith("published_date=") and (d / "sig.parquet").exists()
        )
