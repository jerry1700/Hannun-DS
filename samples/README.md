# samples — 커밋되는 입력 예시

둘 다 **합성 데이터**다. 실제 기사 원문은 이 저장소에 없다(`local/`·`gold/` 는 .gitignore).

| 파일 | 무엇 | 만드는 법 |
|---|---|---|
| `articles_sample.jsonl` | DE 공통 기사 JSON 형식의 기사 7건. 적재 검증(리젝트 사유 포함)과 STEP 0~2 스모크용 | `python scripts/make_sample_data.py` |
| `ds1_issue_output_v1.example.json` | DS1 → DS2 전달 형식의 이슈 3개·기사 66건. 구조(이슈·기사 메타·점수)는 2026-08-24 실제 산출에서 왔고, 제목과 본문은 가상의 사건("가온시 반도체 산업단지" 등)으로 다시 쓴 문장이다. DS2 관점 분석 테스트가 읽는다 | 고정 파일 |
