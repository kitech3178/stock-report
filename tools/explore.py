"""네이버 금융 엔드포인트 탐색용 (2차). 워크플로 '탐색 스크립트 실행'에서만 쓴다."""
import json, requests

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
     "Referer": "https://m.stock.naver.com/"}
S = requests.Session(); S.headers.update(H)

def get(url):
    r = S.get(url, timeout=15)
    print(f"\n===== {url}\nstatus {r.status_code} ctype {r.headers.get('content-type')}")
    if "json" in (r.headers.get("content-type") or ""):
        return r.json()
    print(r.text[:200]); return None

# 1) 업종 그룹 전체
d = get("https://m.stock.naver.com/api/stocks/industry?page=1&pageSize=100")
if d:
    print("groups:", [(g["no"], g["name"], g["totalCount"]) for g in d["groups"]])
# 2) 업종별 종목 목록 후보
for u in ["https://m.stock.naver.com/api/stocks/industry/294?page=1&pageSize=3",
          "https://m.stock.naver.com/api/stocks/industry/294/stocks?page=1&pageSize=3",
          "https://m.stock.naver.com/api/stocks/industryGroup/294?page=1&pageSize=3",
          "https://m.stock.naver.com/api/stock/industry/294?page=1&pageSize=3",
          "https://m.stock.naver.com/api/stocks/theme?page=1&pageSize=3"]:
    try:
        d = get(u)
        if isinstance(d, dict):
            print("keys:", list(d.keys()))
            for k in ("stocks", "groups", "items"):
                if isinstance(d.get(k), list) and d[k]:
                    print(f"{k}[0]:", json.dumps(d[k][0], ensure_ascii=False)[:700]); break
            else:
                print(json.dumps(d, ensure_ascii=False)[:700])
        elif isinstance(d, list):
            print("list", len(d), json.dumps(d[0], ensure_ascii=False)[:700] if d else "")
    except Exception as e:
        print("ERR", u, e)
# 3) integration 의 industryCode / industryCompareInfo
d = get("https://m.stock.naver.com/api/stock/005930/integration")
if d:
    print("industryCode:", d.get("industryCode"))
    print("industryCompareInfo:", json.dumps(d.get("industryCompareInfo"), ensure_ascii=False)[:900])
d = get("https://m.stock.naver.com/api/stock/035720/integration")
if d:
    print("035720 industryCode:", d.get("industryCode"), "| compare:", json.dumps(d.get("industryCompareInfo"), ensure_ascii=False)[:300])
# 4) trend 8일치: 순서/일수 확인
d = get("https://m.stock.naver.com/api/stock/005930/trend?pageSize=8")
if d:
    print("trend dates:", [x["bizdate"] for x in d])
    print("trend[0]:", json.dumps(d[0], ensure_ascii=False))
d = get("https://m.stock.naver.com/api/stock/005930/trend?pageSize=8&page=2")
if d:
    print("trend page2 dates:", [x["bizdate"] for x in d])
