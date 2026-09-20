"""report.html 의 5·6번 섹션을 텍스트로 출력 (러너 로그에서 렌더링 확인용)."""
import re, sys
h = open("report.html", encoding="utf-8").read()
def text(x): return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x)).strip()
for title in ("5. 시가총액", "6. 외국인"):
    i = h.find(title)
    if i < 0:
        print(f"[{title}] 섹션 없음"); continue
    j = h.find("<h2", i + 1); block = h[i:j if j > 0 else None]
    print(f"\n##### {text(block[:120])}")
    for hdr in re.findall(r"<h3[^>]*>(.*?)</h3>", block):
        print("  ---", text(hdr))
    for tr in re.findall(r"<tr>(.*?)</tr>", block, re.S)[:60]:
        print("  " + " | ".join(text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)))
    for p in re.findall(r"<p style='color:#888'>(.*?)</p>", block):
        print("  (p)", text(p))
print("\nreport bytes:", len(h.encode("utf-8")))
