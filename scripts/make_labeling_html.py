"""쌍 판정 라벨링 HTML 생성 — 팀원이 브라우저에서 O/X 버튼으로 라벨링한다.

쌍 CSV(issue_pairs.csv / dedup_pairs.csv)를 읽어 자체 완결 HTML 하나를 만든다.
데이터는 파일 안에 심는다(file:// 에서 CSV fetch 가 CORS 로 막히므로).
라벨러는 자기 번호를 고르면 자동 배정 구간만 보이고, 진행 상황은 localStorage 에
자동 저장, 완료 후 "결과 내보내기"로 채점기 형식 CSV 를 다운로드한다.

기사 제목이 들어가므로 산출물은 local/ 이나 배포 채널로만 — 커밋 금지.

    python scripts/make_labeling_html.py --pairs <pairs.csv> --type issue --labelers 5
"""

import argparse
import csv
import json
from pathlib import Path

QUESTIONS = {
    "issue": {
        "title": "이슈 판정",
        "question": "두 기사가 <b>같은 이슈</b>(같은 사건·국면)를 다루나요?",
        "hint_o": "같은 사건의 다른 기사 (속보 vs 종합, 다른 언론사) / 같은 국면이면 통합",
        "hint_x": "같은 소재지만 다른 사건 (예: 북한 무인기 vs 중국 무인기)",
        "verdict_col": "판정(O=같은이슈,X=다른이슈)",
    },
    "dedup": {
        "title": "중복 판정",
        "question": "두 기사가 <b>사실상 같은 글</b>(전재·재게시)인가요?",
        "hint_o": "문장까지 거의 그대로인 글 (제목만 바뀐 전재 포함)",
        "hint_x": "같은 사건이라도 따로 쓴 기사면 X — 이슈 판정과 기준이 다릅니다!",
        "verdict_col": "판정(O=같은기사,X=다른기사)",
    },
}

TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>한눈 라벨링 — __TITLE__</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: system-ui, "Malgun Gothic", sans-serif; background: #f6f5f2;
         color: #1a1a1a; max-width: 980px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 19px; margin-bottom: 6px; }
  .question { font-size: 16px; margin: 8px 0; }
  .hints { font-size: 13px; color: #52514e; background: #fff; border: 1px solid #e1e0d9;
           border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; line-height: 1.6; }
  .hints b.o { color: #1a7f4e; } .hints b.x { color: #b02b20; }
  .bar { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
  select, button { font-size: 15px; padding: 8px 14px; border-radius: 8px;
                   border: 1px solid #c9c6bd; background: #fff; cursor: pointer; }
  .progress { font-variant-numeric: tabular-nums; color: #52514e; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
  .card { background: #fff; border: 1px solid #e1e0d9; border-radius: 10px; padding: 16px;
          min-height: 190px; }
  .card .pub { font-size: 12px; color: #898781; margin-bottom: 6px; }
  .card .title { font-size: 16px; font-weight: 700; margin-bottom: 10px; line-height: 1.4; }
  .card .body { font-size: 14px; color: #3a3935; line-height: 1.6; }
  .actions { display: flex; gap: 10px; }
  .actions button { flex: 1; font-size: 18px; padding: 16px; font-weight: 700; }
  .btn-o { background: #e8f6ee; border-color: #1a7f4e; color: #1a7f4e; }
  .btn-x { background: #fdecea; border-color: #b02b20; color: #b02b20; }
  .btn-back { flex: 0 0 110px; font-weight: 400; }
  .picked { outline: 3px solid #1a1a1a; }
  .done { background: #fff; border: 1px solid #e1e0d9; border-radius: 10px;
          padding: 24px; text-align: center; }
  .done button { font-size: 17px; padding: 14px 26px; background: #1a1a1a; color: #fff; }
  .keyhint { font-size: 12px; color: #898781; margin-top: 10px; }
  [hidden] { display: none !important; }
</style></head><body>
<h1>한눈 라벨링 — __TITLE__</h1>
<div class="question">__QUESTION__</div>
<div class="hints"><b class="o">O</b> = __HINT_O__<br><b class="x">X</b> = __HINT_X__</div>
<div class="bar">
  <label>내 번호: <select id="labeler"><option value="">선택</option>__LABELER_OPTIONS__</select></label>
  <span class="progress" id="progress"></span>
  <button id="export" hidden>결과 내보내기 (CSV)</button>
</div>
<div id="work" hidden>
  <div class="cards">
    <div class="card"><div class="pub" id="pubA"></div>
      <div class="title" id="titleA"></div><div class="body" id="bodyA"></div></div>
    <div class="card"><div class="pub" id="pubB"></div>
      <div class="title" id="titleB"></div><div class="body" id="bodyB"></div></div>
  </div>
  <div class="actions">
    <button class="btn-back" id="back">← 이전</button>
    <button class="btn-o" id="btnO">O — 같다</button>
    <button class="btn-x" id="btnX">X — 다르다</button>
  </div>
  <div class="keyhint">키보드: O / X 키로 판정, ← 로 이전. 진행 상황은 자동 저장됩니다.</div>
</div>
<div class="done" id="doneBox" hidden>
  <p style="margin-bottom:12px">🎉 배정 구간 완료! 아래 버튼으로 CSV 를 받아 제출해 주세요.</p>
  <button id="export2">결과 내보내기 (CSV)</button>
  <p class="keyhint">수정하려면 ← 이전 키로 돌아갈 수 있어요.</p>
</div>
<script>
const ROWS = __DATA__;
const VERDICT_COL = __VERDICT_COL__;
const SHEET = __SHEET__;
const N_LABELERS = __N_LABELERS__;

let labeler = 0, idx = 0, answers = {};

function rangeOf(k) {
  const per = Math.floor(ROWS.length / N_LABELERS);
  const start = (k - 1) * per;
  const end = (k === N_LABELERS) ? ROWS.length : start + per;
  return [start, end];
}
function storeKey() { return `hannun-label-${SHEET}-${labeler}`; }
function save() {
  try { localStorage.setItem(storeKey(), JSON.stringify({idx, answers})); } catch (e) {}
}
function load() {
  try {
    const raw = localStorage.getItem(storeKey());
    if (raw) { const s = JSON.parse(raw); idx = s.idx; answers = s.answers; return; }
  } catch (e) {}
  idx = rangeOf(labeler)[0]; answers = {};
}
function render() {
  const [start, end] = rangeOf(labeler);
  const doneCount = Object.keys(answers).length;
  document.getElementById("progress").textContent =
    labeler ? `${doneCount}/${end - start} 완료` : "";
  document.getElementById("export").hidden = !(labeler && doneCount > 0);
  if (!labeler) { work.hidden = true; doneBox.hidden = true; return; }
  if (idx >= end) { work.hidden = true; doneBox.hidden = false; return; }
  work.hidden = false; doneBox.hidden = true;
  const r = ROWS[idx];
  pubA.textContent = r.pa; titleA.textContent = r.ta; bodyA.textContent = r.ba || "(본문 없음)";
  pubB.textContent = r.pb; titleB.textContent = r.tb; bodyB.textContent = r.bb || "(본문 없음)";
  btnO.classList.toggle("picked", answers[idx] === "O");
  btnX.classList.toggle("picked", answers[idx] === "X");
}
function answer(v) {
  if (!labeler || idx >= rangeOf(labeler)[1]) return;
  answers[idx] = v; idx += 1; save(); render();
}
function goBack() {
  if (!labeler) return;
  const [start] = rangeOf(labeler);
  if (idx > start) { idx -= 1; save(); render(); }
}
function exportCsv() {
  const [start, end] = rangeOf(labeler);
  const lines = ["article_id_a,article_id_b," + JSON.stringify(VERDICT_COL)];
  for (let i = start; i < end; i++) {
    lines.push(`${ROWS[i].ia},${ROWS[i].ib},${answers[i] || ""}`);
  }
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n")], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${SHEET}_labeler${labeler}.csv`;
  a.click();
}
const work = document.getElementById("work"), doneBox = document.getElementById("doneBox");
const pubA = document.getElementById("pubA"), titleA = document.getElementById("titleA"),
      bodyA = document.getElementById("bodyA"), pubB = document.getElementById("pubB"),
      titleB = document.getElementById("titleB"), bodyB = document.getElementById("bodyB");
const btnO = document.getElementById("btnO"), btnX = document.getElementById("btnX");
document.getElementById("labeler").addEventListener("change", e => {
  labeler = parseInt(e.target.value || "0", 10); if (labeler) load(); render();
});
btnO.addEventListener("click", () => answer("O"));
btnX.addEventListener("click", () => answer("X"));
document.getElementById("back").addEventListener("click", goBack);
document.getElementById("export").addEventListener("click", exportCsv);
document.getElementById("export2").addEventListener("click", exportCsv);
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "o") answer("O");
  else if (k === "x") answer("X");
  else if (k === "arrowleft" || k === "backspace") { e.preventDefault(); goBack(); }
});
render();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description="쌍 판정 라벨링 HTML 생성 (티켓 104)")
    p.add_argument("--pairs", required=True, help="쌍 CSV (issue_pairs.csv 또는 dedup_pairs.csv)")
    p.add_argument("--type", required=True, choices=["issue", "dedup"], help="판정 종류")
    p.add_argument("--labelers", type=int, default=5)
    p.add_argument("--out", default=None, help="출력 HTML (기본: 입력과 같은 자리 .html)")
    args = p.parse_args()

    meta = QUESTIONS[args.type]
    with open(args.pairs, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    data = [{
        "pa": r["publisher_a"], "ta": r["title_a"], "ba": r["body_a"],
        "pb": r["publisher_b"], "tb": r["title_b"], "bb": r["body_b"],
        "ia": r["article_id_a"], "ib": r["article_id_b"],
    } for r in rows]

    sheet = Path(args.pairs).stem
    options = "".join(f'<option value="{i}">{i}번</option>' for i in range(1, args.labelers + 1))
    html = (TEMPLATE
            .replace("__TITLE__", meta["title"])
            .replace("__QUESTION__", meta["question"])
            .replace("__HINT_O__", meta["hint_o"])
            .replace("__HINT_X__", meta["hint_x"])
            .replace("__LABELER_OPTIONS__", options)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__VERDICT_COL__", json.dumps(meta["verdict_col"], ensure_ascii=False))
            .replace("__SHEET__", json.dumps(sheet, ensure_ascii=False))
            .replace("__N_LABELERS__", str(args.labelers)))

    out = Path(args.out) if args.out else Path(args.pairs).with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"저장: {out} (쌍 {len(data)}개, 라벨러 {args.labelers}명 분할)")


if __name__ == "__main__":
    main()
