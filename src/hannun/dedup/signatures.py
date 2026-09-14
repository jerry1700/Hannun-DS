"""MinHash 서명 저장소 — 기사당 한 번만 계산해 실행 간 재사용한다 (티켓 123).

15분 배정 실행이 창 전체의 서명을 매번 다시 만들던 것을 새 기사만 계산하게 바꾸는 캐시.
결과는 캐시 없이 돌린 것과 같아야 하므로, 서명을 만든 정제본의 해시(text_sha1)가 다르거나
서명 설정(shingle_size·num_perm·seed·datasketch 해시 방식)이 다른 행은 없는 것으로 친다 —
정제 규칙이나 설정이 바뀌면 자연히 전부 다시 계산된다.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

from .candidates import CandidateConfig, minhash_scheme

SIGNATURE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("published_date", pa.string()),
        ("text_sha1", pa.string()),
        ("shingle_size", pa.int32()),
        ("num_perm", pa.int32()),
        ("seed", pa.int32()),
        ("scheme", pa.string()),
        ("shingle_count", pa.int32()),
        ("hashvalues", pa.list_(pa.uint64())),
        ("computed_at", pa.timestamp("us", tz="UTC")),
    ]
)


class SignatureStore:
    """<root>/dedup_sig/published_date=YYYY-MM-DD/sig.parquet — Gold 와 같은 발행일 파티션."""

    def __init__(self, root):
        self.root = Path(root)
        self.sig_dir = self.root / "dedup_sig"

    def read_cache(self, start_date: str, end_date: str, config: CandidateConfig):
        """범위의 서명을 article_id → (text_sha1, hashvalues, shingle_count) 로. 설정이 다른 행은 뺀다."""
        setting = _setting(config)
        cache = {}
        for date_str in self.partition_dates():
            if date_str < start_date or date_str > end_date:
                continue
            for row in self._rows(self.partition_path(date_str)):
                if (row["shingle_size"], row["num_perm"], row["seed"], row["scheme"]) != setting:
                    continue
                cache[row["article_id"]] = (
                    row["text_sha1"],
                    np.array(row["hashvalues"], dtype=np.uint64),
                    row["shingle_count"],
                )
        return cache

    def merge_partition(self, published_date: str, fresh: dict, config: CandidateConfig):
        """새 서명을 그 날짜 파티션에 합친다. 같은 article_id 는 새 것으로 바꾼다.

        fresh 는 article_id → (text_sha1, hashvalues, shingle_count). 파티션은 그 날짜 기사 수를
        넘어 자라지 않는다.
        """
        path = self.partition_path(published_date)
        rows = {}
        if path.exists():
            rows = {row["article_id"]: row for row in self._rows(path)}
        shingle_size, num_perm, seed, scheme = _setting(config)
        computed_at = datetime.now(timezone.utc)
        for article_id, (sha1, hashvalues, count) in fresh.items():
            rows[article_id] = {
                "article_id": article_id,
                "published_date": published_date,
                "text_sha1": sha1,
                "shingle_size": shingle_size,
                "num_perm": num_perm,
                "seed": seed,
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

    def _rows(self, path):
        table = pq.read_table(path)
        # 컬럼 구성이 달라진 옛 파티션은 버린다. 캐시라서 잃어도 다시 계산하면 그만이고,
        # 다음 merge 가 새 스키마로 다시 쓴다.
        if table.schema.names != SIGNATURE_SCHEMA.names:
            return []
        return table.to_pylist()


def _setting(config):
    return (config.shingle_size, config.num_perm, config.seed, minhash_scheme())
