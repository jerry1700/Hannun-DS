"""품질 규칙 — 정형(템플릿) 이슈 판별과 노이즈 구제 문턱. STEP 4 의 판단 기준."""

from dataclasses import dataclass


@dataclass
class QualityConfig:
    # 정형 이슈 판별 — 적은 언론사가 많은 기사를 내는 시황·인사·포토 류 템플릿을 잡는다.
    # 더 느슨하면 지역·단독 보도가 걸린다. 값의 근거는 티켓 93
    structured_max_publishers: int = 3
    structured_min_size: int = 10
    # 노이즈 구제 — 이 아래 구간에서 정탐률이 급락한다(티켓 93 검수). 벡터가 정규화돼 있어
    # 내적이 곧 코사인
    rescue_min_sim: float = 0.85
    # 증분 배정(티켓 102) — 낮추면 오배정이 압도한다(시뮬레이션). 구제와 같은 값이라 규칙이 하나다
    assign_min_sim: float = 0.85


def structured_issue_ids(meta: dict, config: QualityConfig):
    """이슈별 (규모, 언론사 수) 로 정형 이슈 번호 집합을 고른다. meta 는 issue_local → (size, publishers)."""
    return {
        issue_id for issue_id, (size, publishers) in meta.items()
        if publishers <= config.structured_max_publishers and size >= config.structured_min_size
    }
