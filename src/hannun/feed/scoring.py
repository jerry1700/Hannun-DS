"""화제성 점수 — 이슈 피드 정렬용. (규모 + 언론사 다양성) × 신선도 감쇠.

설계 근거(티켓 91):
- 언론사 다양성을 규모보다 무겁게 — 실측에서 정형 기사는 1~3곳, 진짜 화제는
  13~17곳이 다뤘다(티켓 93). 다양성이 "사회적 화제"를 가장 잘 가른다
- 로그 스케일 — 483건짜리 정형 덩어리가 선형으로 점수를 압도하지 않게
- 신선도는 마지막 기사 이후 경과 시간의 지수 감쇠 — 어제 끝난 이슈는 오늘 밀려난다
"""

import math
from dataclasses import dataclass


@dataclass
class FeedConfig:
    # log2(1+언론사 수) 에 곱하는 가중 — 2.0 이면 언론사 4곳 증가가 규모 4배와 같은 무게
    publisher_weight: float = 2.0
    # 신선도 반감 상수(시간) — 마지막 기사로부터 τ시간이 지나면 점수가 1/e 로
    recency_tau_hours: float = 24.0


def hot_score(size: int, publishers: int, hours_since_last: float, config: FeedConfig):
    base = math.log2(1 + size) + config.publisher_weight * math.log2(1 + publishers)
    return base * math.exp(-max(hours_since_last, 0.0) / config.recency_tau_hours)
