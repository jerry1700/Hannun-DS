"""기사 본문 한 건을 정제한다. 원문은 그대로 두고 새 문자열과 적용한 규칙 이름을 돌려준다."""

import html
import re
from dataclasses import dataclass, field

from .rules import (
    BROKEN_ENTITY, BROKEN_ENTITY_MAP, COPYRIGHT_START, INLINE_RULES, LINE_RULES, MISSING_SENTENCE_SPACE,
    TAIL_INLINE_RULES, TAIL_RULES,
)

STATUS_OK = "ok"
STATUS_SHORT = "short"
STATUS_EMPTY = "empty"

_ODD_SPACES = re.compile(r"[\t\u00a0\u3000\u200b]")
_MANY_SPACES = re.compile(r" {2,}")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_DANGLING_OPEN_QUOTE = re.compile(r"\s*[“‘]\s*$")


@dataclass
class PreprocessConfig:
    """min_clean_len 미만이면 short. 100 은 DE 크롤러가 발행 전에 버리는 기준과 같다."""

    min_clean_len: int = 100
    rules_version: str = "2026-08-28"


@dataclass
class CleanResult:
    content_clean: str
    status: str
    rules_applied: list[str] = field(default_factory=list)
    removed_chars: int = 0


def clean_content(content: str, publisher_id: str | None, config: PreprocessConfig):
    """규칙을 엔티티 → 인라인 → 저작권 꼬리 → 줄 → 꼬리 줄 → 마지막 줄 안 → 공백 순으로 적용한다.

    엔티티를 먼저 고쳐야 'amp;' 가 섞인 저작권 문구가 걸린다. 저작권 꼬리를 자른 뒤에야 그 앞에
    붙어 있던 서명이 마지막 줄 끝에 오고, 공백 정리는 마지막이어야 지운 자리의 빈 줄이 정리된다.
    """
    applied = []
    text = fix_broken_entities(content)
    if text != content:
        applied.append("broken_entity")
    # zero-width space 는 str.strip() 도 \s 도 못 잡는다. 마침표와 서명 사이에 끼어 있으면 뒤의 모든
    # 규칙이 빗나가므로 먼저 보통 공백으로 바꾼다. 줄 구조는 그대로다.
    text = _ODD_SPACES.sub(" ", text)

    for rule in _for(INLINE_RULES, publisher_id):
        text, n = rule.pattern.subn("", text)
        if n:
            applied.append(rule.name)

    lines, n = _cut_copyright(text.split("\n"))
    if n:
        applied.append("copyright")

    for rule in _for(LINE_RULES, publisher_id):
        kept = [line for line in lines if not rule.pattern.search(line.strip())]
        if len(kept) != len(lines):
            applied.append(rule.name)
        lines = kept

    lines, hit = _strip_tail(lines, _for(TAIL_RULES, publisher_id))
    applied.extend(hit)

    lines, hit = _clean_last_line(lines, _for(TAIL_INLINE_RULES, publisher_id))
    applied.extend(hit)

    text = normalize_whitespace("\n".join(lines))
    # 서명 앞에 홀로 있던 여는 따옴표('… “ 이민종 기자')가 끝에 남는다. 글이 여는 따옴표로 끝날 수는 없다.
    if _DANGLING_OPEN_QUOTE.search(text):
        text = _DANGLING_OPEN_QUOTE.sub("", text)
        applied.append("dangling_quote")
    if MISSING_SENTENCE_SPACE.search(text):
        text = MISSING_SENTENCE_SPACE.sub(r"\1 ", text)
        applied.append("sentence_space")

    if not text:
        status = STATUS_EMPTY
    elif len(text) < config.min_clean_len:
        status = STATUS_SHORT
    else:
        status = STATUS_OK
    return CleanResult(text, status, applied, len(content) - len(text))


def fix_broken_entities(text: str):
    """'apos;' 처럼 '&' 가 떨어진 엔티티를 복원한 뒤 정상 엔티티(&amp; 등)도 푼다."""
    fixed = BROKEN_ENTITY.sub(lambda m: BROKEN_ENTITY_MAP[m.group(1)], text)
    return html.unescape(fixed)


def normalize_whitespace(text: str):
    """탭·NBSP 를 공백으로, 줄 끝 공백과 연속 공백을 하나로, 빈 줄은 최대 하나로."""
    text = _ODD_SPACES.sub(" ", text)
    lines = [_MANY_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _for(rules, publisher_id):
    return [r for r in rules if r.publishers is None or publisher_id in r.publishers]


def _cut_copyright(lines):
    cut, n = [], 0
    for line in lines:
        m = COPYRIGHT_START.search(line)
        if m:
            line = line[:m.start()]
            n += 1
        cut.append(line)
    return cut, n


def _strip_tail(lines, rules):
    # 빈 줄은 건너뛰고 본문 줄이 나올 때까지만 본다. 본문 중간의 이메일 주소는 남는다.
    hit = []
    end = len(lines)
    while end > 0:
        stripped = lines[end - 1].strip()
        if not stripped:
            end -= 1
            continue
        rule = next((r for r in rules if r.pattern.search(stripped)), None)
        if rule is None:
            break
        hit.append(rule.name)
        end -= 1
    return lines[:end], hit


def _clean_last_line(lines, rules):
    # 사진 출처를 지우면 그 앞의 서명이 줄 끝에 드러나므로, 더 지워지는 게 없을 때까지 돈다.
    hit = []
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue
        changed = True
        while changed:
            changed = False
            for rule in rules:
                lines[i], n = rule.pattern.subn("", lines[i])
                if n:
                    hit.append(rule.name)
                    changed = True
        break
    return lines, hit
