import pytest

from hannun.preprocess import PreprocessConfig, clean_content

CFG = PreprocessConfig(min_clean_len=10)
ZWSP = chr(0x200B)  # 원문에 섞여 있는 zero-width space
BODY = "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다."
# 기자 이름·이메일은 전부 가공값 — 규칙이 보는 형태(한글 2~4자 + 직함, 로컬파트@매체 도메인)만 실제와 같다


def clean(text, publisher_id=None, config=CFG):
    return clean_content(text, publisher_id, config)


def test_broken_entities_restored():
    r = clean("법원, 김철수 amp;#39;면직 집행정지amp;#39; 기각 apos;냉장고apos; 사건 A&amp;B quot;인용quot;")
    assert r.content_clean == "법원, 김철수 '면직 집행정지' 기각 '냉장고' 사건 A&B \"인용\""
    assert "broken_entity" in r.rules_applied


def test_copyright_lines_removed_for_any_publisher():
    tails = [
        "[ⓒ 세계일보 & Segye.com, 무단전재 및 재배포 금지]",
        "GoodNews paper ⓒ , 무단전재 및 수집, 재배포금지",
        "Copyright © 뉴스1. All rights reserved. 무단 전재 및 재배포 금지.",
        "<ⓒ투자가를 위한 경제콘텐츠 플랫폼, 아시아경제(www.asiae.co.kr) 무단전재 배포금지>",
    ]
    for tail in tails:
        r = clean(f"{BODY}\n\t\t\t\n\n{tail}")
        assert r.content_clean == BODY, tail
        assert "copyright" in r.rules_applied


def test_copyright_glued_to_one_line_article_keeps_the_body():
    one_line = f"{BODY}{BODY}Copyright © 뉴스1. All rights reserved. 무단 전재 및 재배포 금지."
    r = clean(one_line, "fnnews")
    assert r.content_clean == f"{BODY}{BODY}".replace("한다.정부는", "한다. 정부는")
    assert "copyright" in r.rules_applied


@pytest.mark.parametrize("tail", [
    "☞공감언론 뉴시스 plain@newsis.com <저작권자ⓒ 공감언론 뉴시스통신사. 무단전재-재배포 금지.>",
    "“저작권ⓒ `건강을 위한 정직한 지식` 코메디닷컴(https://kormedi.com) / 무단전재-재배포 금지“",
    "<저작권자 ⓒ MBN(www.mbn.co.kr) 무단전재 및 재배포 금지>",
    "<저작권자(c) 연합뉴스, 무단 전재-재배포 금지>",
])
def test_copyright_tail_removed_from_its_opening_mark(tail):
    assert clean(f"{BODY}{tail}").content_clean == BODY
    assert clean(f"{BODY}\n\n{tail}").content_clean == BODY


def test_copyright_words_inside_the_body_are_kept():
    text = f"{BODY} 해킹으로 추정되는 무단 접속 공격을 받았다. 무단 전재를 금지한 약관도 논란이다. {BODY}"
    assert clean(text).content_clean == text


def test_dateline_header_and_suicide_notice_removed():
    notice = "※ 우울감 등 말하기 어려운 고민이 있거나 주변에 이런 어려움을 겪는 가족·지인이 있을 경우 자살 예방 핫라인 1393"
    r = clean(f"광주=이영희 기자\n{BODY}\n{notice}", "munhwa")
    assert r.content_clean == BODY
    assert {"dateline_header", "suicide_notice"} <= set(r.rules_applied)


def test_bracket_wire_header_removed():
    r = clean(f"[서울=뉴시스] 박민수 기자 = {BODY}", "fnnews")
    assert r.content_clean == BODY


def test_agency_source_lines_removed():
    r = clean(f"{BODY}\n[연합뉴스]\n<뉴시스>")
    assert r.content_clean == BODY


@pytest.mark.parametrize("signature", [
    "도쿄/정수진 특파원",
    "제주 최지훈 기자",
    "한소라 온라인 뉴스 기자 news1701@segye.com",
    "[장미래 기자]",
    "[조은별 디지털뉴스 기자 digitalnews@mbn.co.kr]",
    "(SBS 디지털뉴스편집부)",
])
def test_signature_variants_removed_at_tail(signature):
    assert clean(f"{BODY}\n\n{signature}").content_clean == BODY


@pytest.mark.parametrize("signature", [
    "동아닷컴 IT전문 윤서준 기자(tech@itdonga.com)",
    " 강민아 엔터뉴스팀 기자 kang.mina@jtbc.co.kr (콘텐트비즈니스본부)",
    " 1press@tf.co.kr[연예부 | ent@tf.co.kr]",
    " 신동해 기자",
    " 임하늘 전문기자",
    " [이 기사는 증시분석 전문기자 서경뉴스봇(newsbot@sedaily.com)이 실시간으로 작성했습니다.]",
])
def test_signature_variants_glued_to_last_sentence_removed(signature):
    assert clean(f"{BODY}{signature}", "sedaily").content_clean == BODY


def test_name_without_punctuation_before_it_is_kept():
    text = "이번 행사를 기획한 사람은 신동해 기자"
    assert clean(text).content_clean == text


@pytest.mark.parametrize("signature", [
    "[ 문지혜 기자 Moon8166@mbn.co.kr ]",
    "[ 황보람 기자 / hbr712@mbn.co.kr ]",
])
def test_bracket_signature_with_spaces_removed(signature):
    assert clean(f"{BODY}\n\n{signature}", "mbn").content_clean == BODY


def test_series_intro_line_removed():
    intro = "[생활의 발견]은 우리의 삶과 밀접한 관계가 있는 소재들을 다룹니다. 먹고 입고 쓰는 것들의 이야기입니다."
    assert clean(f"{intro}\n{BODY}").content_clean == BODY


def test_stacked_tail_lines_all_removed():
    r = clean(f"{BODY}\ndesk79@yna.co.kr\n\n제보는 카카오톡 okjebo\n\n2023/06/23 10:12 송고", "yonhap")
    assert r.content_clean == BODY
    assert {"reporter_email_line", "tip_line", "date_stamp_line"} <= set(r.rules_applied)


def test_signature_before_copyright_tail_removed():
    r = clean(f"{BODY}곽민정 기자 newsDesk@kmib.co.kr\n\t\t\t\n\n\t\t\t"
              "GoodNews paper ⓒ , 무단전재 및 수집, 재배포금지", "kmib")
    assert r.content_clean == BODY
    assert {"copyright", "reporter_signature_inline"} <= set(r.rules_applied)


def test_factcheck_notice_removed_for_jtbc_only():
    notice = "JTBC 팩트체크는 국제팩트체킹네트워크(IFCN) 인증사입니다.※JTBC는 시청자 여러분의 ‘팩트체크‘ 소재를 기다립니다. (factcheck@jtbc.co.kr)"
    assert clean(f"{BODY}{notice}", "jtbc").content_clean == BODY
    assert "IFCN" in clean(f"{BODY}{notice}", "sbs").content_clean


def test_reporter_email_on_last_line_removed():
    r = clean(f"{BODY}\t\t\t\t\n\nreporter@news1.kr", "news1")
    assert r.content_clean == BODY


def test_reporter_signature_glued_to_last_sentence_removed():
    r = clean(f"{BODY}유진아기자 reporter@donga.com", "donga")
    assert r.content_clean == BODY
    assert "reporter_signature_inline" in r.rules_applied


def test_reporter_name_line_removed_only_at_tail():
    r = clean(f"{BODY}\n홍길동 기자")
    assert r.content_clean == BODY
    r2 = clean(f"홍길동 기자\n{BODY}")
    assert r2.content_clean.startswith("홍길동 기자")


def test_wire_header_removed():
    r = clean(f"(안성=뉴스1) 배지수 기자 = {BODY}", "fnnews")
    assert r.content_clean == BODY
    assert "wire_header" in r.rules_applied


def test_broadcast_script_markers_and_signoff_removed():
    text = ("[앵커] 정부가 청년 정책을 발표했습니다. 박민수 기자가 보도합니다. [리포트] 청년 주거 안정이 핵심입니다. "
            "KBS 뉴스 박민수입니다. 촬영기자:김철수/영상편집:이영희/그래픽:정수진")
    r = clean(text, "kbs")
    assert r.content_clean == "정부가 청년 정책을 발표했습니다. 박민수 기자가 보도합니다. 청년 주거 안정이 핵심입니다."
    assert {"broadcast_marker", "broadcast_signoff", "credits_inline"} <= set(r.rules_applied)


def test_mbn_broadcast_tail_block_removed():
    tail = "\n\nMBN뉴스 설민아입니다.\n[ news21@mbn.co.kr ]\n\n영상취재 : 최지훈 기자\n영상편집 : 장미래"
    assert clean(f"{BODY}{tail}", "mbn").content_clean == BODY


def test_yonhap_auto_article_notices_removed():
    tail = ("\nabc@yna.co.kr\n※ 이 기사는 엔씨소프트의 인공지능 기술인 자연어처리기술을 이용해 자동 작성됐습니다."
            "\n기사의 원 데이터인 기상청 기상예보는 웹사이트에서도 확인할 수 있습니다.\n광고\n기사 문의나 제보는 카카오톡 okjebo")
    assert clean(f"{BODY}{tail}", "yonhap").content_clean == BODY


@pytest.mark.parametrize("signature", [
    "photo01@tf.co.kr사진영상기획부",
    " 세종 한소라 기자",
    " 윤서준 인턴기자·강민아 기자",
    " 전주 신동해·봉화 임하늘 기자",
    " 대만 가오슝시=문지혜기자",
    " 워싱턴 황보람 특파원·서울 곽민정 기자",
    " [ flash@mbn.co.kr ]",
    ZWSP + "photo01@tf.co.kr사진영상기획부",
    " 유진아 엔터뉴스팀 기자 yoo.jina@jtbc.co.kr (콘텐트비즈니스본부) 사진=연합뉴스",
    " 배지수 엔터뉴스팀 기자 bae.jisu@jtbc.co.kr(콘텐트비즈니스본부) SM엔터테인먼트 제공",
    "\n\n설민아 기자 seol@munhwa.com, 사진=마노엔터테인먼트 제공",
    " <사진=국민권익위원회 제공>photo@tf.co.kr사진영상기획부",
    "\n\n김철수기자 news13@dt.co.kr\n\n▶관련기사 17면",
    "“ 이영희 기자",
    "\n박민수 기자 5news@hani.co.kr, 정수진 기자 desk@hani.co.kr",
    "\n\nnewsdesk@fnnews.com 최지훈 기자",
    " 배지수 엔터뉴스팀 기자 bae.jisu@jtbc.co.kr(콘텐트비즈니스본부) 사진=장미래 기자",
    "\n\n[조은별 디지털뉴스부 인턴기자 intern2001@naver.com]]",
    "\n\n[ 윤서준 기자 reporter@mbn.co.kr ]·",
    "\n콘텐츠 사용과 관련해 궁금한 점이 있으면 전화(☎:02-398-3655) 또는 이메일(desk@yna.co.kr)로 문의하기 바랍니다.",
])
def test_more_signature_variants_removed(signature):
    assert clean(f"{BODY}{signature}", "yonhap").content_clean == BODY


def test_mbn_anchor_marker_and_desk_signature_removed():
    r = clean(f"【 앵커멘트 】\n{BODY}\n\n[강민아 디지털뉴스부 인턴기자 intern98@gmail.com]", "mbn")
    assert r.content_clean == BODY


def test_photo_credit_removed_inline_and_as_line():
    r = clean(f"{BODY} (사진=연합뉴스 TV 제공, 연합뉴스, 게티이미지코리아)\n  \n (사진=연합뉴스)\n<연합>")
    assert r.content_clean == BODY


def test_image_caption_marker_line_removed_in_the_middle():
    r = clean(f"{BODY}\n이미지 확대\n{BODY}")
    assert r.content_clean == f"{BODY}\n{BODY}"


def test_publisher_specific_rule_does_not_leak():
    ad = "주식투자도 이제 글로벌 시대! 와우스탁론에서 새롭게 출시한 해외주식 스탁론."
    assert ad not in clean(f"{BODY}\n{ad}", "wowtv").content_clean
    assert ad in clean(f"{BODY}\n{ad}", "kbs").content_clean


def test_missing_space_after_sentence_end_inserted():
    r = clean("보조금을 대폭 삭감했습니다.대구시는 24.7% 줄였습니다.3.5%는 유지했다.“인용”도 그렇다.")
    assert r.content_clean == "보조금을 대폭 삭감했습니다. 대구시는 24.7% 줄였습니다. 3.5%는 유지했다. “인용”도 그렇다."
    assert "sentence_space" in r.rules_applied


def test_whitespace_normalized_but_paragraphs_kept():
    r = clean("첫 문단이다.  둘째  문장.\n\n\n\n둘째 문단이다.\t끝.")
    assert r.content_clean == "첫 문단이다. 둘째 문장.\n\n둘째 문단이다. 끝."


def test_plain_body_is_unchanged_and_ok():
    r = clean(BODY)
    assert r.content_clean == BODY
    assert r.status == "ok"
    assert r.rules_applied == []
    assert r.removed_chars == 0


@pytest.mark.parametrize("text,status", [
    ("[ⓒ 세계일보 & Segye.com, 무단전재 및 재배포 금지]", "empty"),
    ("짧은 본문.", "short"),
])
def test_status_reflects_remaining_length(text, status):
    assert clean(text).status == status


def test_removed_chars_counts_original_minus_clean():
    r = clean(f"{BODY}\n\n[ⓒ 세계일보 & Segye.com, 무단전재 및 재배포 금지]")
    assert r.removed_chars == len(f"{BODY}\n\n[ⓒ 세계일보 & Segye.com, 무단전재 및 재배포 금지]") - len(BODY)
