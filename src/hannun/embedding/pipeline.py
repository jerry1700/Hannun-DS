"""Gold·clean·dedup 을 읽어 대표 기사만 임베딩하고 embedding 테이블을 갱신한다."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from hannun.dedup.store import DedupStore
from hannun.ingest.gold import GoldStore
from hannun.preprocess.store import CleanStore

from .encoder import EncoderConfig, build_input, encode_texts, input_sha1, load_model
from .store import EmbeddingStore

log = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    model: str
    partitions: list[str] = field(default_factory=list)
    rows: int = 0
    folded: int = 0
    already: int = 0
    encoded: int = 0
    reencoded: int = 0
    removed: int = 0
    docs_per_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def embed(gold: GoldStore, clean: CleanStore, dedup: DedupStore, store: EmbeddingStore,
          config: EncoderConfig | None = None, start_date: str | None = None,
          end_date: str | None = None, encode_fn=None):
    """날짜 범위(UTC, 양끝 포함)의 대표 기사를 임베딩한다. 재실행해도 바뀐 기사만 인코딩한다.

    접힌 기사(duplicate_of 있음)는 건너뛰고, 이전 실행에서 인코딩됐더라도 벡터를 지운다 —
    같은 글이 벡터 공간에 여러 번 찍히면 STEP 3 의 밀도 기반 군집화가 그 자리를 과대평가한다.
    저장된 벡터는 인코더 입력 해시(input_sha1)가 같을 때만 재사용한다 — 재크롤링(replace)이나
    정제 규칙 개정으로 본문이 바뀌면 다시 인코딩한다(티켓 131). 모델이 다른 행도 버린다 —
    모델이 섞인 테이블은 군집화에서 쓸 수 없다. dedup 파티션이 아직 없으면 전부 대표로
    간주하고 경고만 남긴다(운영 순서: 정제 → 중복 → 임베딩). encode_fn 은 테스트 주입용.
    """
    config = config or EncoderConfig()
    stats = EmbedStats(model=config.model_name)
    encoded_at = datetime.now(timezone.utc)
    total_secs = 0.0

    # 모델 로드는 무겁다(수 초 + 수백 MB) — 인코딩할 기사가 실제로 나올 때까지 미룬다
    model_box = {}

    def default_encode(texts):
        if "model" not in model_box:
            model_box["model"] = load_model(config)
        return encode_texts(model_box["model"], config, texts)

    encode = encode_fn or default_encode

    dates = [
        d for d in gold.partition_dates()
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
    ]
    dedup_dates = set(dedup.partition_dates())

    for date_str in dates:
        table = gold.read_table(date_str, date_str,
                                columns=["article_id", "publisher_id", "title", "content"])
        clean_table = clean.read_table(date_str, date_str, columns=["article_id", "content_clean"])
        clean_of = dict(zip(clean_table.column("article_id").to_pylist(),
                            clean_table.column("content_clean").to_pylist()))

        folded = set()
        if date_str in dedup_dates:
            dedup_table = dedup.read_table(date_str, date_str, columns=["article_id", "duplicate_of"])
            folded = {a for a, rep in zip(dedup_table.column("article_id").to_pylist(),
                                          dedup_table.column("duplicate_of").to_pylist()) if rep is not None}
        else:
            log.warning(f"dedup 파티션 없음: {date_str} — 전부 대표로 간주하고 임베딩합니다")

        existing_rows = store.read_table(date_str, date_str).to_pylist()
        # 옛 파티션에는 input_sha1 컬럼이 없어 해시가 None 으로 읽힌다 — 그 행은 전부 다시 인코딩된다(첫 실행 한 번)
        current = {row["article_id"]: row for row in existing_rows if row["model"] == config.model_name}

        kept, todo = [], []
        for article_id, publisher_id, title, content in zip(
                *(table.column(c).to_pylist()
                  for c in ("article_id", "publisher_id", "title", "content"))):
            stats.rows += 1
            if article_id in folded:
                stats.folded += 1
                continue
            text = clean_of.get(article_id) or content
            digest = input_sha1(config, title, text)
            row = current.get(article_id)
            if row is not None and row.get("input_sha1") == digest:
                stats.already += 1
                kept.append(row)
                continue
            if row is not None:
                stats.reencoded += 1
            todo.append((article_id, publisher_id, build_input(title, text, config.body_chars), digest))

        # kept 에도 todo 에도 없는 기존 행 — 접힌 기사, 다른 모델, Gold 에서 사라진 기사 — 는 여기서 빠진다
        replaced = {article_id for article_id, _, _, _ in todo if article_id in current}
        stats.removed += len(existing_rows) - len(kept) - len(replaced)

        rows = kept
        if todo:
            vectors, docs_per_s = encode([text for _, _, text, _ in todo])
            total_secs += len(todo) / max(docs_per_s, 1e-9)
            stats.encoded += len(todo)
            rows = kept + [{
                "article_id": article_id,
                "publisher_id": publisher_id,
                "published_date": date_str,
                "vector": vector,
                "dim": len(vector),
                "model": config.model_name,
                "input_sha1": digest,
                "encoded_at": encoded_at,
            } for (article_id, publisher_id, _, digest), vector in zip(todo, vectors)]
        if todo or len(kept) != len(existing_rows):
            store.write_partition(date_str, rows)
        stats.partitions.append(date_str)

    if total_secs:
        stats.docs_per_s = round(stats.encoded / total_secs, 1)
    log.info(
        f"embed done: partitions={len(stats.partitions)} rows={stats.rows} folded={stats.folded} "
        f"already={stats.already} encoded={stats.encoded} reencoded={stats.reencoded} "
        f"removed={stats.removed} ({stats.docs_per_s}/s)"
    )
    return stats
