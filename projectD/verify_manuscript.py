# 프로젝트D 원고의 모든 '출력 블록'이 실제 실행 로그에 존재하는지 대조 검증
import re, glob, sys

BOOK = "/Users/choejunhyeon/교재/프로젝트D_설비예지보전.md"
LOGS = sorted(glob.glob("/Users/choejunhyeon/교재/projectD/outputs/*_output.txt"))

log_text = ""
for p in LOGS:
    log_text += open(p, encoding="utf-8").read() + "\n"
log_lines = set(l.rstrip() for l in log_text.splitlines())

md = open(BOOK, encoding="utf-8").read()
lines = md.splitlines()

blocks, i = [], 0
while i < len(lines):
    if lines[i].startswith("```"):
        lang = lines[i][3:].strip()
        j, buf = i + 1, []
        while j < len(lines) and not lines[j].startswith("```"):
            buf.append(lines[j]); j += 1
        blocks.append((lang or "OUT", i + 1, buf))
        i = j + 1
    else:
        i += 1

py  = [b for b in blocks if b[0] == "python"]
sh  = [b for b in blocks if b[0] in ("bash", "yaml", "sql", "markdown")]
out = [b for b in blocks if b[0] == "OUT"]
print(f"로그 파일         : {len(LOGS)}개")
print(f"코드 블록(python) : {len(py)}개")
print(f"코드 블록(기타)   : {len(sh)}개")
print(f"출력 블록         : {len(out)}개")

# 출력이 아닌 블록(도식·의사코드·구조도·체크리스트)은 제외
SKIP_MARKERS = (
    "├──", "└──", "│", "→", "↓", "predictive-maintenance-portfolio/",
    "[Part 1]", "절삭 부하가 커진다", "설비 단계", "train:", "train  (모델 학습)",
    "[ ]", "AI4I 고장률 3.39% →", "| 항목 |", "git ", "pip install", "mkdir",
    "for i in $(seq", "streamlit run", "python ", "cd ", "name: ", "on:",
    "  \"SECOM 데이터로", "예지보전", "\"CNC 설비의",
)

# 에러 사전용 '예시 에러 메시지'는 우리 로그가 아니라 재현 예시입니다.
ERROR_EXAMPLES = (
    "FutureWarning", "SettingWithCopyWarning", "sqlite3.", "ValueError",
    "UndefinedMetricWarning", "due to no predicted samples", "ModuleNotFoundError",
    "remote: Permission", "fatal: unable to access", "Resource limits exceeded",
    "재색인 후 200만 행", "please use 'min' instead",
    "Your app is having trouble loading",
)


def is_diagram(buf):
    j = "\n".join(buf)
    return any(mk in j for mk in SKIP_MARKERS) or any(mk in j for mk in ERROR_EXAMPLES)


missing, checked, skipped = [], 0, 0
for lang, ln, buf in out:
    if is_diagram(buf):
        skipped += 1
        continue
    for line in buf:
        s = line.rstrip()
        if not s or s.strip() in ("...", "```"):
            continue
        checked += 1
        if s not in log_lines:
            missing.append((ln, s))

print(f"도식/명령/에러예시 : {skipped}개 제외")
print(f"대조한 출력 라인  : {checked}줄")
print(f"로그에 없는 라인  : {len(missing)}줄")
for ln, s in missing[:80]:
    print(f"  L{ln}: {s!r}")
sys.exit(0)
