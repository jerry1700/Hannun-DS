"""이슈 카테고리 — 언론사별 원문 라벨을 팀 어휘로 맞추고, 구성 기사의 다수결로 이슈 값을 정한다."""

import collections

# 팀 확정 어휘(2026-09-15, 티켓 34): BE·FE 탭과 같다. 기사 단위 원문(gold.category)은
# 그대로 두고 이슈 단위에서만 맞춘다 — 원본 보존, 어휘가 또 바뀌면 여기만 고친다
CATEGORIES = ("정치", "경제", "사회", "국제", "연예", "스포츠", "문화", "IT/과학")
OTHER = "OTHER"
# DE 수집 라벨 중 어휘와 다른 것. 산업은 경제로 합치기로 했다
ALIASES = {"산업": "경제", "IT과학": "IT/과학"}


def normalize_category(value: str | None):
    """원문 라벨을 팀 어휘로. 빈값은 OTHER.

    어휘에 없는 값은 그대로 둔다 — OTHER 로 삼키면 새 라벨이 들어온 것을 못 본다.
    """
    if not value:
        return OTHER
    return ALIASES.get(value, value)


def majority_category(categories, fallback: str | None):
    """이슈 카테고리 — 구성 기사 중 OTHER·빈값을 뺀 다수결. 별칭은 합쳐서 센다.

    동률에 대표 기사 값이 끼어 있으면 그것을, 라벨 기사가 하나도 없으면 대표 기사 값을
    쓴다(그것도 없으면 OTHER). 기사 단위 라벨은 수집 피드에 섹션 정보가 없는 언론사 탓에
    드물지만, 이슈에 라벨 기사가 하나라도 있으면 채워지므로 이슈 단위 커버리지는 훨씬
    높다(티켓 34).
    """
    fallback = normalize_category(fallback)
    votes = collections.Counter(normalize_category(c) for c in categories)
    votes.pop(OTHER, None)
    if not votes:
        return fallback
    ranked = votes.most_common()
    tied = [c for c, n in ranked if n == ranked[0][1]]
    return fallback if fallback in tied else tied[0]
