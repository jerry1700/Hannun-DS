"""본문에서 지울 것들 — 언론사 × 패턴 표. 근거는 docs/ds1/DATASET_PROFILE.md 의 보일러플레이트 절."""

import re
from dataclasses import dataclass

# 원문 CSV 에는 '&apos;' 의 '&' 가 떨어진 'apos;' 꼴이, 세계일보에는 '&amp;#39;' 가 한 번 더 깨진
# 'amp;#39;' 꼴이 남아 있다. 긴 것부터 바꿔야 'amp;' 가 먼저 먹어 '&#39;' 로 남는 일이 없다.
BROKEN_ENTITIES = (
    ("amp;#39;", "'"),
    ("amp;quot;", '"'),
    ("amp;amp;", "&"),
    ("apos;", "'"),
    ("quot;", '"'),
    ("nbsp;", " "),
    ("amp;", "&"),
    ("lt;", "<"),
    ("gt;", ">"),
)
# 앞이 '&' 면 정상 엔티티라 html.unescape 에 맡긴다. 영문자 뒤도 제외해야 'result;' 의 'lt;' 가 안 걸린다.
# 한글 뒤에는 \b 가 성립하지 않으므로('집행정지amp;#39;') 단어 경계 대신 이 조건을 쓴다.
BROKEN_ENTITY = re.compile(r"(?<![&A-Za-z])(" + "|".join(re.escape(k) for k, _ in BROKEN_ENTITIES) + ")")
BROKEN_ENTITY_MAP = dict(BROKEN_ENTITIES)

# \w 는 한글도 매치해서 '한다.cjg05023@tf.co.kr' 의 '한다.' 까지 아이디로 먹는다. ASCII 로 묶고
# 아이디는 영숫자로 시작하게 해서 앞 문장의 마침표를 삼키지 않는다.
EMAIL = r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"


@dataclass(frozen=True)
class Rule:
    """지울 패턴 하나.

    scope 는 넷 중 하나다. inline 은 본문 어디서든 패턴만 지운다. line 은 패턴이 걸리는 줄을
    통째로 지운다. tail 은 뒤에서부터 패턴이 걸리는 줄만 연속으로 지우고 본문 중간은 건드리지
    않는다. tail_inline 은 마지막 줄 안에서 패턴만 지운다 — 서명이 마지막 문장에 이어 붙은 경우다.
    publishers 가 None 이면 모든 언론사에 건다.
    """

    name: str
    scope: str
    pattern: re.Pattern
    publishers: tuple[str, ...] | None = None


def _rule(name, scope, pattern, publishers=None):
    return Rule(name, scope, re.compile(pattern), publishers)


# 순서가 결과를 바꾼다. 통신사 헤더를 먼저 떼야 그 뒤 매체명 접두가 줄 머리에 온다.
INLINE_RULES = (
    # (안성=뉴스1) 배수아 기자 = ... / [서울=뉴시스] 최지윤 기자 = ... — 통신사 전재 기사의 머리
    _rule("wire_header", "inline",
          r"^\s*[\[(][^\[\]()=]{1,15}=\s*[^\[\]()]{1,12}[\])]\s*[가-힣]{2,5}\s*(?:선임|수석)?기자\s*=\s*"),
    # 광주=김대우 기자 — 문화일보는 서명이 첫 줄에 온다
    _rule("dateline_header", "inline", r"^\s*[가-힣]{1,6}\s*=\s*[가-힣]{2,4}\s*(?:선임|수석)?기자\s*\n"),
    _rule("outlet_prefix", "inline", r"^\s*\[[가-힣A-Za-z]{2,10}(?:뉴스|신문|일보|경제|타임스)\]\s*"),
    # <앵커> [리포트] 【 앵커멘트 】 — 방송 원고 마커. 문장이 아니라 지운다
    _rule("broadcast_marker", "inline",
          r"<\s*(?:앵커|기자|출연자|진행|리포트|앵커멘트|인터뷰)\s*>"
          r"|\[\s*(?:앵커|리포트|기자|인터뷰|녹취|싱크|VCR|CG)\s*\]"
          r"|【\s*(?:앵커멘트|앵커|기자|인터뷰|녹취)\s*】"),
    # (사진=연합뉴스) / 사진=김현우 기자 — 괄호 없는 꼴은 JTBC 서명 뒤에 붙는다
    _rule("photo_credit", "inline", r"\(\s*사진\s*[=:：][^()]{1,80}\)|\s*사진\s*=\s*[가-힣]{2,4}\s*기자"),
    # KBS 뉴스 김석입니다. / MBN뉴스 배준우입니다. — 방송 원고의 마무리 인사
    _rule("broadcast_signoff", "inline",
          r"\s*(?:KBS|MBN|SBS|JTBC|YTN|MBC|TV조선|채널A)\s*뉴스\s*[가-힣]{2,4}입니다\.?"),
    # 촬영기자:김종우/영상편집:여동용/그래픽:김지혜 — KBS 는 마지막 문장 뒤에 줄바꿈 없이 붙는다
    _rule("credits_inline", "inline",
          r"\s*(?:촬영기자|영상취재|영상편집|그래픽|촬영|편집|CG|화면제공|자료조사)\s*[:：][^\n]*$"),
)

# 저작권 꼬리가 시작되는 표지. 여기부터 그 줄 끝까지 지운다. 파이낸셜뉴스·머니S·뉴스1 은 기사 전체가
# 한 줄이라 줄 단위로 지우면 기사가 통째로 사라지므로, 줄 안에서 표지 이후만 자른다. 줄 전체가 꼬리면
# 빈 줄이 되어 공백 정리에서 사라진다. '무단 전재' 는 뒤에 배포·수집이 따라올 때만 — 본문의
# '무단 전재를 금지한 약관' 같은 문장은 남긴다. 여는 따옴표·괄호를 앞에 붙여 '“저작권ⓒ' 의 따옴표가 안 남게 한다.
COPYRIGHT_START = re.compile(
    r"[“\"\[<(]?\s*(?:GoodNews paper|Copyright\s*[©ⓒⒸ]|저작권자?\s*(?:[ⓒ©Ⓒ]|\(c\))|[ⓒ©Ⓒ]\s*\S|☞"
    r"|무단\s*전재\s*(?:및|[-·,/와과])?\s*(?:수집|재배포|배포|복사|복제|캡처))"
)

LINE_RULES = (
    _rule("news1_terms", "line", r"뉴스1이 제공하는 기사"),
    # 자살 관련 기사에 의무적으로 붙는 예방 안내문
    _rule("suicide_notice", "line", r"^※\s*우울감"),
    # (사진=연합뉴스) / <연합> / [뉴시스] — 출처만 적힌 줄
    _rule("photo_credit_line", "line",
          r"^(?:\(\s*사진\s*[=:：][^()]*\)|[\[<(]\s*(?:연합뉴스|뉴시스|뉴스1|연합|AP|AFP|로이터)\s*[\])>])$"),
    _rule("image_caption_marker", "line", r"^(?:이미지\s*확대|사진\s*확대|확대보기)$"),
    _rule("newsletter_promo", "line", r"^☞|뉴스레터를 구독"),
    _rule("tip_line", "line", r"^(?:기사 문의나\s*)?제보는\s*카카오톡"),
    # ※ 이 기사는 엔씨소프트의 인공지능 기술인 자연어처리기술…로 자동 작성 — 연합뉴스 자동 기사 안내
    _rule("ai_notice", "line", r"^※\s*이 기사는.*(?:인공지능|자연어|자동)"),
    _rule("weather_source_notice", "line", r"^기사의 원 데이터인", ("yonhap",)),
    # 콘텐츠 사용과 관련해 궁금한 점이 있으면 전화(☎…) 또는 이메일(…)로 문의하기 바랍니다 — 연합뉴스 이용 안내
    _rule("license_notice", "line", r"콘텐츠 사용과 관련해 궁금한 점|저작권법에 따라 보호되는 콘텐츠", ("yonhap",)),
    _rule("video_notice", "line", r"^\*?\s*해당 내용은 관련 동영상|^※\s*자세한 내용은 동영상|^\(남은 이야기는 스프에서\)$"),
    _rule("wowtv_ad", "line", r"와우스탁론", ("wowtv",)),
    # [생활의 발견]은 우리의 삶과 … 다룹니다. / [‘건강’한 ‘먹’거리 정보’방’, 건강먹방은 … 코너입니다.
    _rule("series_intro", "line", r"^\[[^\]]{2,40}\]?\s*(?:은|는|,)?\s.*(?:다룹니다|코너입니다|소개합니다)"),
    # ---- 실수집(실시간 크롤) 잔재 — 티켓 119. 사이트 UI·요약봇 문구가 본문에 섞여 들어온다 ----
    # 연합 실시간분은 "요약문 + 안내 + 송고 시각" 형태로 온다
    _rule("rt_summary_notice", "line",
          r"^전체 내용을 이해하기 위해서는 기사 본문과 함께|^인공지능이 자동으로 줄인", ("yonhap",)),
    _rule("rt_dispatch_stamp", "line", r"^송고\s*\d{4}년", ("yonhap",)),
    # 구글 검색 우선 노출 안내 — 연합·한경·매경이 문구만 다르게 단다
    _rule("rt_google_promo", "line",
          r"^(?:구글|Google)\s*검색(?:에서)?\s.*(?:우선적으로|더 자주|선호)"),
    _rule("rt_hankyung_reco", "line", r"^AI 추천 뉴스$", ("hankyung",)),
    _rule("rt_hankyung_ui", "line", r"^(?:기사 스크랩|댓글|공유|프린트|글자크기|-)$", ("hankyung",)),
    _rule("rt_mk_ui", "line",
          r"^(?:공유|글자 크기|가|번역|구글 검색 선호 추가|M매경미디어 소개|뉴스 바로가기"
          r"|햄버거|AI검색|로그인|마이페이지|로그아웃|매일경제\s*\d*)$", ("mk",)),
    _rule("rt_donga_ui", "line",
          r"^(?:-|-\s*\d+개|-\s*(?:좋아요|슬퍼요|화나요|후속기사 원해요|추천해요)"
          r"|글자크기 설정|공유하기(?:\s.*)?)$", ("donga",)),
    _rule("rt_seoul_summary_label", "line", r"^세줄 요약$", ("seoul",)),
    _rule("rt_ohmynews_motto", "line", r"^오마이뉴스의 모토는", ("ohmynews",)),
    # 홀로 있는 '광고'·'▲' 줄은 어느 매체든 본문이 아니다
    _rule("rt_ad_line", "line", r"^광고$"),
    _rule("rt_image_marker", "line", r"^▲$"),
    # (서울=연합뉴스) / [서울=뉴시스] — 통신사 크레딧만 홀로 있는 줄
    _rule("rt_wire_credit_line", "line", r"^[\[(][가-힣]{2,8}\s*=\s*(?:연합뉴스|뉴시스|뉴스1)[\])]$"),
    _rule("rt_google_register", "line", r"^구글 선호 매체 등록$"),
    _rule("rt_donga_wire_list", "line", r"^-\s*(?:뉴시스|뉴스1|연합뉴스)(?:\s*\([가-힣]{1,6}\))?$", ("donga",)),
    _rule("rt_etnews_ui", "line", r"^(?:-|-\s*ET\s*Events)$", ("etnews",)),
    # 홍민성 한경닷컴 기자 mshong@… — 위젯 사이에 끼어 tail 규칙이 못 닿는 서명
    _rule("rt_hankyung_signature", "line",
          r"^[가-힣]{2,4}\s*한경닷컴\s*기자\s+" + EMAIL + r"$", ("hankyung",)),
)

# 표지 줄부터 본문 끝까지 통째로 지운다 — 기사 뒤에 붙는 위젯 블록(퀴즈·추천 기사).
# 표지는 위젯의 고정 머리말이라 본문 문장과 겹칠 수 없는 것만 고른다.
CUT_RULES = (
    _rule("rt_seoul_quiz", "cut",
          r"^기사를 끝까지 읽으셨나요|^\[\s*내안의 AI 본성 분석|^Q\.$", ("seoul",)),
    _rule("rt_kmib_widget", "cut",
          r"^클릭! 기사는 어떠셨나요|^많이 본 기사$|^오늘의 추천기사$", ("kmib",)),
    _rule("rt_hankyung_premium", "cut", r"^한경 프리미엄9의 모든 콘텐츠는", ("hankyung",)),
    # 매경 기사 뒤 위젯들(주요뉴스 추천·종목 위젯·퀴즈·구독 유도)
    _rule("rt_mk_widget", "cut",
          r"^(?:매경에서 선정한 주요뉴스|기사 속 종목 이야기|추천질문|뉴스퀴즈"
          r"|아직 가입을 안 하셨다면,?)$", ("mk",)),
)

# 마지막 줄부터 거슬러 올라가며 이 중 하나라도 걸리면 지우고, 아무것도 안 걸릴 때까지 반복한다.
# 연합뉴스는 이메일·제보 안내·날짜가 석 줄로 겹쳐 있어 규칙 하나씩 한 번만 보면 맨 끝 한 줄만 지워진다.
TAIL_RULES = (
    _rule("reporter_email_line", "tail", r"^[\[(]?\s*" + EMAIL + r"\s*[\])]?$"),
    # 김유민 기자 / 광주=김대우 기자 / 도쿄/김소연 특파원 / 최윤정 온라인 뉴스 기자 mary1701@segye.com /
    # 윤예림 인턴기자·신진호 기자. 기자·특파원 앞에 단어 셋까지만 — '이번 행사를 기획한 사람은 김민지 기자'
    # 같은 문장은 단어가 더 많아서 남는다
    _rule("reporter_signature_line", "tail",
          r"^(?:[가-힣/=·]{1,10}\s+){0,3}[가-힣/=·]{2,12}\s*(?:기자|특파원)(?:\s+" + EMAIL + r")?$"),
    # helpfire@fnnews.com 임우섭 기자 — 파이낸셜뉴스는 이메일이 이름 앞에 온다
    _rule("email_then_name_line", "tail", r"^" + EMAIL + r"\s+[가-힣]{2,4}\s*(?:선임|수석|전문|인턴|객원)?\s*기자$"),
    # 오세진 기자 5sjin@hani.co.kr, 김OO 기자 xx@hani.co.kr — 한 줄에 서명 여럿
    _rule("multi_signature_line", "tail",
          r"^(?:[가-힣]{2,4}\s*(?:선임|수석|전문|인턴|객원)?\s*기자\s+" + EMAIL + r"[,\s]*)+$"),
    # 영상취재 : 배완호 기자 / 영상편집 : 이재형 — MBN 은 제작진이 줄마다 온다
    _rule("credits_line", "tail", r"^(?:촬영기자|영상취재|영상편집|그래픽|취재|편집|촬영|CG|화면제공|자료조사)\s*[:：]"),
    _rule("ad_marker_line", "tail", r"^광고$"),
    _rule("related_line", "tail", r"^▶\s*관련\s*기사"),
    # [디지털뉴스부] / [박통일 기자] / [ 이재호 기자 Jay8166@mbn.co.kr ] / [ 박규원 기자 / pkw712@mbn.co.kr ]
    _rule("desk_signature_line", "tail",
          r"^[\[(]\s*(?:[A-Za-z가-힣]+\s*){1,5}(?:기자|디지털뉴스부|디지털뉴스편집부|온라인뉴스부|온라인뉴스팀)"
          r"(?:\s*/?\s*" + EMAIL + r")?\s*[\])]+[·,\s]*$"),
    # 2023/01/02 11:35 송고 — 연합뉴스 송고 시각
    _rule("date_stamp_line", "tail", r"^\d{4}[/.-]\d{2}[/.-]\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?(?:\s*송고)?$"),
    _rule("affiliate_signoff", "tail", r"^[A-Z]{2,5}\s+[가-힣]{2,4}$", ("sbs",)),
    _rule("kormedi_credit", "tail", r"^◆?\s*기사\s*도움\s*[:：]", ("kormedi",)),
)

# 꼬리 줄을 정리한 뒤의 마지막 줄에만 건다. 저작권 꼬리를 먼저 잘라야 그 앞의 서명이 줄 끝에 온다.
# 더 지워지는 게 없을 때까지 반복 적용되므로, 사진 출처 뒤에 숨은 서명도 차례로 지워진다.
TAIL_INLINE_RULES = (
    # …기자 email (콘텐트비즈니스본부) 사진=연합뉴스 / …, 사진=마노엔터테인먼트 제공 / <사진=국민권익위원회 제공>
    _rule("photo_credit_tail", "tail_inline", r"[,\s]*(?:사진\s*=\s*[^\n,<>]{1,40}|<[^<>\n]{1,40}>)\s*$"),
    # SM엔터테인먼트 제공 — 문장은 마침표로 끝나므로 부호 없이 '제공' 으로 끝나면 출처 표기다
    _rule("provided_credit_tail", "tail_inline",
          r"[,\s]*[가-힣A-Za-z0-9·&]{2,25}(?:\s+[가-힣A-Za-z0-9·&]{1,15}){0,2}\s*제공\s*$"),
    # 황재성기자 jsonhng@donga.com / 동아닷컴 IT전문 정연호 기자(hoho@itdonga.com) /
    # 조연경 엔터뉴스팀 기자 cho.yeongyeong@jtbc.co.kr (콘텐트비즈니스본부) / 2kuns@tf.co.kr[연예부 | ssent@tf.co.kr]
    # 이름·소속 단어는 넷까지. 앞 문장은 마침표로 끝나서 단어 사슬이 거기서 끊기므로 본문을 먹지 않는다
    _rule("reporter_signature_inline", "tail_inline",
          r"\s*(?:(?:[가-힣A-Za-z]{2,6}\s*){1,4}기자\s*)?[\[(]?\s*" + EMAIL + r"\s*[\])]?"
          r"(?:\s*\[[^\]]{0,60}\])?(?:\s*\([^()]{1,30}\))?(?:\s*[가-힣]{2,12}(?:부|팀|본부|국))?[,\s]*$"),
    # …조사하고 있다. 김민지 기자 / …늘었다. 세종 강주리 기자 / …알려졌다. 윤예림 인턴기자·신진호 기자
    # 이메일 없이 이름만 붙은 꼴. 문장 부호 뒤일 때만 — 부호 없이 이어지면 문장의 일부다
    # 지역 표기는 '세종 ', '대만 가오슝시=' 처럼 한두 단어 뒤 공백이나 '=' 로 이름과 갈린다
    _rule("reporter_name_glued", "tail_inline",
          r"(?<=[.!?“”‘’\"'\)])\s*"
          r"(?:(?:[가-힣]{1,8}(?:\s+[가-힣]{1,8})?\s*[=\s]\s*)?[가-힣]{2,4}\s*"
          r"(?:(?:선임|수석|전문|인턴|객원)?\s*(?:기자|특파원))?\s*[·,]\s*)*"
          r"(?:[가-힣]{1,8}(?:\s+[가-힣]{1,8})?\s*[=\s]\s*)?[가-힣]{2,4}\s*(?:선임|수석|전문|인턴|객원)?\s*(?:기자|특파원)\s*$"),
    # [이 기사는 증시분석 전문기자 서경뉴스봇(newsbot@sedaily.com)이 실시간으로 작성했습니다.]
    _rule("newsbot_notice", "tail_inline", r"\s*\[이 기사는[^\]]*작성했습니다\.?\]\s*$", ("sedaily",)),
    # JTBC 팩트체크는 국제팩트체킹네트워크(IFCN) 인증사입니다.※JTBC는 시청자 여러분의 ‘팩트체크‘ 소재를 기다립니다.
    _rule("factcheck_notice", "tail_inline",
          r"\s*(?:JTBC\s*팩트체크는[^\n※]*)?※\s*JTBC는\s*시청자[^\n]*$", ("jtbc",)),
)

# 문장 끝 부호 뒤에 공백 없이 다음 문장이 붙은 자리('삭감했습니다.대구시는'). 앞이 한글일 때만 봐서
# 소수점(24.7%)과 영문 약어(U.S.)는 건드리지 않는다. KBS·뉴스1 원문에 이 꼴이 많아 DS2 문장 분리가 깨진다.
MISSING_SENTENCE_SPACE = re.compile(r"(?<=[가-힣])([.!?])(?=[가-힣0-9“‘\"'\[(])")
