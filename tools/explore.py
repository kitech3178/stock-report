"""네이버 금융 엔드포인트 탐색용. 워크플로 '탐색 스크립트 실행'에서만 쓴다."""
import json, re, requests

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
     "Referer": "https://m.stock.naver.com/"}
S = requests.Session(); S.headers.update(H)

def probe(url, n=1800, enc=None):
    print(f"\n===== {url}")
    try:
        r = S.get(url, timeout=15)
        body = r.content.decode(enc or r.encoding or "utf-8", errors="replace")
        print("status", r.status_code, "len", len(body), "ctype", r.headers.get("content-type"))
        if "json" in (r.headers.get("content-type") or ""):
            try:
                d = r.json()
                if isinstance(d, dict):
                    print("top keys:", list(d.keys()))
                    for k in ("stocks", "items", "list", "data", "result"):
                        if isinstance(d.get(k), list) and d[k]:
                            print(f"first {k}[0]:", json.dumps(d[k][0], ensure_ascii=False)[:n]); break
                    else:
                        print(json.dumps(d, ensure_ascii=False)[:n])
                elif isinstance(d, list) and d:
                    print("list len", len(d), "first:", json.dumps(d[0], ensure_ascii=False)[:n])
                return d
            except Exception as e:
                print("json err", e)
        print(re.sub(r"\s+", " ", body)[:n])
        return body
    except Exception as e:
        print("ERR", e)

# 1) 시가총액 순 목록 필드
probe("https://m.stock.naver.com/api/stocks/marketValue/KOSPI?page=1&pageSize=2")
# 2) 종목 기본정보 (업종 필드 있는지)
probe("https://m.stock.naver.com/api/stock/005930/basic")
probe("https://m.stock.naver.com/api/stock/005930/integration", n=2500)
# 3) 투자자별 매매동향 후보
probe("https://m.stock.naver.com/api/stock/005930/trend?pageSize=3")
probe("https://api.stock.naver.com/stock/005930/trend?pageSize=3")
probe("https://m.stock.naver.com/api/stock/005930/investor?pageSize=3")
probe("https://m.stock.naver.com/api/stock/005930/trade?pageSize=3")
# 4) 구 PC 페이지 (외국인·기관 순매매, 업종 목록)
b = probe("https://finance.naver.com/item/frgn.naver?code=005930", n=600, enc="euc-kr")
if isinstance(b, str):
    i = b.find("기관"); print("frgn snippet:", re.sub(r"\s+", " ", b[i-200:i+1500]) if i > 0 else "no 기관")
b = probe("https://finance.naver.com/sise/sise_group.naver?type=upjong", n=300, enc="euc-kr")
if isinstance(b, str):
    links = re.findall(r'href="(/sise/sise_group_detail\.naver\?type=upjong&no=\d+)"[^>]*>([^<]+)<', b)
    print("upjong count:", len(links), "sample:", links[:8])
    if links:
        d = probe("https://finance.naver.com" + links[0][0].replace("&amp;", "&"), n=300, enc="euc-kr")
        if isinstance(d, str):
            codes = re.findall(r'/item/main\.naver\?code=(\d{6})', d)
            print("detail codes:", len(codes), codes[:10])
# 5) 업종 API 후보
probe("https://m.stock.naver.com/api/stocks/industry?page=1&pageSize=3")
probe("https://m.stock.naver.com/api/industry/list")
probe("https://m.stock.naver.com/api/stock/005930/industry")
