"""임베딩 인코더 — 제목과 본문 앞부분을 문장 벡터로 바꾼다. STEP 2 의 심장."""

import time
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    # 후보 비교 실험(티켓 11)으로 선정. 품질 지표가 세 후보 동점이라 운영 비용(속도·차원)이 갈랐다
    model_name: str = "dragonkue/multilingual-e5-small-ko-v2"
    # e5 계열은 학습 때 붙인 접두어를 인코딩 때도 붙여야 성능이 나온다
    passage_prefix: str = "passage: "
    # 입력은 "제목 + 본문 앞부분". 제목이 핵심 사건을, 본문 앞이 육하원칙을 담는다(역피라미드)
    body_chars: int = 500
    batch_size: int = 32
    # EC2(t3.xlarge)의 vCPU 수에 맞춘 기본값. 0 이면 torch 기본값을 그대로 둔다
    threads: int = 4


def build_input(title: str, body: str, body_chars: int):
    return f"{title}\n{body[:body_chars]}"


def load_model(config: EncoderConfig):
    """모델을 CPU 로 올린다.

    torch 임포트를 함수 안에 두는 이유: embedding 의존성 그룹을 설치하지 않은
    환경(DS2 등)에서도 패키지의 다른 모듈은 그대로 임포트되어야 한다.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if config.threads:
        torch.set_num_threads(config.threads)
    return SentenceTransformer(config.model_name, device="cpu")


def encode_texts(model, config: EncoderConfig, texts: list[str]):
    """인코딩하고 (벡터 리스트, 문서/초) 를 돌려준다. 정규화해 두면 코사인이 곧 내적이다."""
    payload = [config.passage_prefix + t for t in texts]
    t0 = time.monotonic()
    vectors = model.encode(payload, batch_size=config.batch_size, normalize_embeddings=True,
                           show_progress_bar=False)
    elapsed = max(time.monotonic() - t0, 1e-9)
    return [[float(x) for x in v] for v in vectors], round(len(texts) / elapsed, 1)
