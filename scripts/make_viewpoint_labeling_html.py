"""세부 견해 골든셋 라벨링 HTML 생성.

전체 40개 pair를 문항당 3명에게 순환 배정한다.
라벨러가 5명이면 각자 정확히 24개를 판정한다.
공식 기준:
- 최종 주장·결론과 핵심 근거가 모두 실질적으로 같으면 O.
- 같은 stance라도 최종 주장·결론 또는 핵심 근거가 다르면 X.
- 판단하기 어려우면 ?.

입력 CSV 필수 컬럼:
pair_id,issue_id,event_name,stance,
article_id_a,title_a,evidence_a,
article_id_b,title_b,evidence_b

예:
python scripts/make_viewpoint_labeling_html.py \
  --pairs local/viewpoint_pairs.csv \
  --labelers 5 \
  --pair-count 40 \
  --out local/viewpoint_labeling.html
"""

import argparse
import csv
import json
from pathlib import Path


TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>한눈 라벨링 — 세부 견해 판정</title>
<style>
* { box-sizing: border-box; }
body {
  font-family: system-ui, "Malgun Gothic", sans-serif;
  background: #f6f5f2;
  color: #1a1a1a;
  max-width: 1100px;
  margin: 0 auto;
  padding: 20px;
}
h1 { font-size: 20px; margin-bottom: 6px; }
.question { font-size: 17px; margin: 10px 0; }
.hints {
  font-size: 13px;
  line-height: 1.7;
  background: #fff;
  border: 1px solid #ddd9d0;
  border-radius: 9px;
  padding: 12px 15px;
  margin-bottom: 14px;
}
.o { color: #18794e; font-weight: 700; }
.x { color: #b42318; font-weight: 700; }
.q { color: #7a5b00; font-weight: 700; }

.bar {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
select, button {
  font-size: 15px;
  padding: 8px 13px;
  border-radius: 8px;
  border: 1px solid #c9c6bd;
  background: #fff;
}
.context {
  background: #fff;
  border: 1px solid #ddd9d0;
  border-radius: 9px;
  padding: 12px 15px;
  margin-bottom: 12px;
  line-height: 1.6;
}
.context b { margin-right: 5px; }

.cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 14px;
}
.card {
  background: #fff;
  border: 1px solid #ddd9d0;
  border-radius: 10px;
  padding: 16px;
  min-height: 230px;
}
.tag {
  font-size: 12px;
  color: #777;
  margin-bottom: 7px;
}
.title {
  font-size: 16px;
  font-weight: 700;
  line-height: 1.45;
  margin-bottom: 12px;
}
.label {
  font-size: 12px;
  font-weight: 700;
  color: #666;
  margin-bottom: 5px;
}
.evidence {
  font-size: 14px;
  line-height: 1.65;
  background: #f7f7f5;
  padding: 10px;
  border-radius: 7px;
}
.actions {
  display: flex;
  gap: 9px;
}
.actions button {
  flex: 1;
  padding: 15px 8px;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
}
.back { flex: 0 0 100px !important; font-weight: 400 !important; }
.btn-o { background: #e8f6ee; border-color: #18794e; color: #18794e; }
.btn-x { background: #fdecea; border-color: #b42318; color: #b42318; }
.btn-q { background: #fff5cc; border-color: #9a7600; color: #765a00; }

.progress { color: #555; }
.keyhint { color: #888; font-size: 12px; margin-top: 10px; }
.done {
  background: #fff;
  border: 1px solid #ddd9d0;
  border-radius: 10px;
  padding: 25px;
  text-align: center;
}
[hidden] { display: none !important; }
</style>
</head>

<body>

<h1>한눈 라벨링 — 세부 견해 판정</h1>

<div class="question">
두 기사가 해당 이슈에 대해 <b>같은 최종 주장과 핵심 근거</b>를 말하고 있나요?
</div>

<div class="hints">
<span class="o">O — 같은 세부 견해</span>:
표현이 달라도 <b>최종 주장과 핵심 근거가 모두 실질적으로 같으면 O</b><br>

<span class="x">X — 다른 세부 견해</span>:
같은 stance라도 <b>최종 주장 또는 핵심 근거가 다르면 X</b><br>
(예: “녹지 보존을 위해 청년주택에 반대” vs “법·행정 절차가 부족해 청년주택에 반대”)<br>

<span class="q">? — 판단 어려움</span>:
기사에서 주장이나 핵심 근거가 불명확해 판단하기 어려움
</div>

<div class="bar">
<label>
내 번호:
<select id="labeler">
<option value="">선택</option>
__LABELER_OPTIONS__
</select>
</label>

<span class="progress" id="progress"></span>
<button id="export" hidden>결과 내보내기 (CSV)</button>
</div>

<div id="work" hidden>

<div class="context">
<div><b>이슈:</b> <span id="eventName"></span></div>
<div><b>stance:</b> <span id="stance"></span></div>
<div><b>문항:</b> <span id="pairNo"></span></div>
</div>

<div class="cards">

<div class="card">
<div class="tag">기사 A</div>
<div class="title" id="titleA"></div>
<div class="label">판단 근거 문장</div>
<div class="evidence" id="evidenceA"></div>
</div>

<div class="card">
<div class="tag">기사 B</div>
<div class="title" id="titleB"></div>
<div class="label">판단 근거 문장</div>
<div class="evidence" id="evidenceB"></div>
</div>

</div>

<div class="actions">
<button class="back" id="back">← 이전</button>
<button class="btn-o" id="btnO">O — 같다</button>
<button class="btn-x" id="btnX">X — 다르다</button>
<button class="btn-q" id="btnQ">? — 애매</button>
</div>

<div class="keyhint">
키보드: O / X / ? · ← 이전 · 진행상황은 브라우저에 자동 저장됩니다.
</div>

</div>

<div class="done" id="doneBox" hidden>
<p>🎉 전체 문항 판정 완료!</p>
<p style="margin:12px 0">CSV를 내려받아 제출해 주세요.</p>
<button id="export2">결과 내보내기 (CSV)</button>
</div>

<script>
const ROWS = __DATA__;
const SHEET = __SHEET__;

let labeler = 0;
let idx = 0;
let answers = {};
let assignedRows = [];

function storeKey() {
  return `hannun-viewpoint-v2-${SHEET}-${labeler}`;
}

function save() {
  try {
    localStorage.setItem(
      storeKey(),
      JSON.stringify({idx, answers})
    );
  } catch (e) {}
}

function load() {
  try {
    const raw = localStorage.getItem(storeKey());
    if (raw) {
      const saved = JSON.parse(raw);
      idx = saved.idx || 0;
      answers = saved.answers || {};
      return;
    }
  } catch (e) {}

  idx = 0;
  answers = {};
}

function render() {
  const doneCount = Object.keys(answers).length;

  progress.textContent =
    labeler ? `${doneCount}/${assignedRows.length} 완료` : "";

  exportBtn.hidden = !(labeler && doneCount > 0);

  if (!labeler) {
    work.hidden = true;
    doneBox.hidden = true;
    return;
  }

  if (idx >= assignedRows.length) {
    work.hidden = true;
    doneBox.hidden = false;
    return;
  }

  work.hidden = false;
  doneBox.hidden = true;

  const r = assignedRows[idx];

  eventName.textContent = r.event_name || "(이슈명 없음)";
  stance.textContent = r.stance || "(stance 없음)";
  pairNo.textContent = `${idx + 1} / ${assignedRows.length}`;

  titleA.textContent = r.title_a || "(제목 없음)";
  titleB.textContent = r.title_b || "(제목 없음)";

  evidenceA.textContent = r.evidence_a || "(근거 문장 없음)";
  evidenceB.textContent = r.evidence_b || "(근거 문장 없음)";
}

function answer(value) {
  if (!labeler || idx >= assignedRows.length) return;

  answers[assignedRows[idx].pair_id] = value;
  idx += 1;
  save();
  render();
}

function goBack() {
  if (!labeler || idx <= 0) return;

  idx -= 1;
  save();
  render();
}

function csvEscape(value) {
  const s = String(value ?? "");
  return '"' + s.replaceAll('"', '""') + '"';
}

function exportCsv() {
  const header = [
    "pair_id",
    "issue_id",
    "article_id_a",
    "article_id_b",
    "labeler",
    "verdict"
  ];

  const lines = [header.join(",")];

  assignedRows.forEach(r => {
    lines.push([
      csvEscape(r.pair_id),
      csvEscape(r.issue_id),
      csvEscape(r.article_id_a),
      csvEscape(r.article_id_b),
      csvEscape(labeler),
      csvEscape(answers[r.pair_id] || "")
    ].join(","));
  });

  const blob = new Blob(
    ["\ufeff" + lines.join("\r\n")],
    {type: "text/csv;charset=utf-8"}
  );

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${SHEET}_labeler${labeler}.csv`;
  a.click();
}

const work = document.getElementById("work");
const doneBox = document.getElementById("doneBox");
const progress = document.getElementById("progress");
const exportBtn = document.getElementById("export");

const eventName = document.getElementById("eventName");
const stance = document.getElementById("stance");
const pairNo = document.getElementById("pairNo");

const titleA = document.getElementById("titleA");
const titleB = document.getElementById("titleB");
const evidenceA = document.getElementById("evidenceA");
const evidenceB = document.getElementById("evidenceB");

document.getElementById("labeler").addEventListener("change", e => {
  labeler = parseInt(e.target.value || "0", 10);
  assignedRows = labeler
    ? ROWS.filter(r => r.assigned_labelers.includes(labeler))
    : [];
  if (labeler) load();
  render();
});

document.getElementById("btnO").addEventListener("click", () => answer("O"));
document.getElementById("btnX").addEventListener("click", () => answer("X"));
document.getElementById("btnQ").addEventListener("click", () => answer("?"));

document.getElementById("back").addEventListener("click", goBack);
exportBtn.addEventListener("click", exportCsv);
document.getElementById("export2").addEventListener("click", exportCsv);

document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();

  if (k === "o") answer("O");
  else if (k === "x") answer("X");
  else if (k === "?" || k === "/") answer("?");
  else if (k === "arrowleft" || k === "backspace") {
    e.preventDefault();
    goBack();
  }
});

render();
</script>

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="세부 견해 골든셋 라벨링 HTML 생성"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--labelers", type=int, default=5)
    parser.add_argument("--pair-count", type=int, default=40)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "pair_id",
        "issue_id",
        "event_name",
        "stance",
        "article_id_a",
        "title_a",
        "evidence_a",
        "article_id_b",
        "title_b",
        "evidence_b",
    }

    missing = required - set(rows[0].keys()) if rows else required

    if missing:
        raise SystemExit(
            "필수 컬럼 없음: " + ", ".join(sorted(missing))
        )

    if args.labelers < 3:
        raise SystemExit("문항당 3명 배정을 위해 라벨러는 3명 이상이어야 합니다.")

    if args.pair_count <= 0:
        raise SystemExit("--pair-count는 1 이상이어야 합니다.")

    if len(rows) < args.pair_count:
        raise SystemExit(
            f"요청한 문항은 {args.pair_count}개지만 "
            f"입력 CSV에는 {len(rows)}개만 있습니다."
        )

    rows = rows[:args.pair_count]

    for index, row in enumerate(rows):
        row["assigned_labelers"] = [
            ((index + offset) % args.labelers) + 1
            for offset in range(3)
        ]

    labeler_names = {
        1: "L3",
        2: "L5",
        3: "L4",
        4: "L2",
        5: "L6",
    }

    options = "".join(
        f'<option value="{i}">{i}번 {labeler_names.get(i, "")}</option>'
        for i in range(1, args.labelers + 1)
    )

    sheet = Path(args.pairs).stem

    html = (
        TEMPLATE
        .replace("__LABELER_OPTIONS__", options)
        .replace("__DATA__", json.dumps(rows, ensure_ascii=False))
        .replace("__SHEET__", json.dumps(sheet, ensure_ascii=False))
    )

    out = (
        Path(args.out)
        if args.out
        else Path(args.pairs).with_suffix(".html")
    )

    out.write_text(html, encoding="utf-8")

    print(
        f"저장: {out} "
        f"(전체 {len(rows)}문항 × 문항당 3명, "
        f"총 {len(rows) * 3}건 판정)"
    )

    counts = {
        labeler: sum(
            labeler in row["assigned_labelers"]
            for row in rows
        )
        for labeler in range(1, args.labelers + 1)
    }
    print("라벨러별 문항 수:", counts)


if __name__ == "__main__":
    main()
