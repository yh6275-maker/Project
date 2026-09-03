"""원고의 python 코드 블록 문법 검사 + 그림 참조 존재 검사 + 분량 추정"""
import ast, re, pathlib

BOOK = pathlib.Path("/Users/choejunhyeon/교재/프로젝트D_설비예지보전.md")
IMG  = pathlib.Path("/Users/choejunhyeon/교재/images")
md = BOOK.read_text(encoding="utf-8")
lines = md.splitlines()

blocks, i = [], 0
while i < len(lines):
    if lines[i].startswith("```"):
        lang = lines[i][3:].strip()
        j, buf = i + 1, []
        while j < len(lines) and not lines[j].startswith("```"):
            buf.append(lines[j]); j += 1
        blocks.append((lang, i + 1, buf)); i = j + 1
    else:
        i += 1

py = [b for b in blocks if b[0] == "python"]
bad = []
for lang, ln, buf in py:
    src = "\n".join(buf)
    if any(l.lstrip().startswith("!") for l in buf):   # Colab 매직(!apt-get 등)
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        # 함수 일부만 발췌한 블록은 들여쓰기 때문에 실패할 수 있음 → 감싸서 재시도
        try:
            ast.parse("def _wrap():\n" + "\n".join("    " + l for l in buf))
        except SyntaxError:
            bad.append((ln, f"{e.msg} (line {e.lineno})"))

print(f"python 코드 블록 : {len(py)}개")
print(f"문법 오류        : {len(bad)}개")
for ln, msg in bad:
    print(f"  L{ln}: {msg}")

# 그림 참조
refs = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", md)
missing_img = [r for r in refs if not (IMG.parent / r).exists()]
print(f"\n그림 참조        : {len(refs)}개")
print(f"없는 그림        : {len(missing_img)}개 {missing_img}")

# placeholder 검사
ph = []
for pat in [r"<본인계정>", r"<본인 이름>", r"TODO", r"FIXME", r"XXX"]:
    for m in re.finditer(pat, md):
        ph.append((md[:m.start()].count("\n") + 1, pat))
print(f"\n원고 내 placeholder: {len(ph)}개 (템플릿 안내용은 정상)")

# 분량 추정 (A4, 본문 기준 약 2,600자/쪽)
chars = len(md)
print(f"\n총 문자수        : {chars:,}")
print(f"총 줄수          : {len(lines):,}")
print(f"추정 분량        : 약 {chars/2600:.0f}쪽 (2,600자/쪽 기준)")
