"""
한국 주식 데일리 리포트 에이전트
- 전일 등락률 상위/하위 종목
- 3거래일 연속 상승 종목
- 최근 6개월 고가 대비 -30% 이상 하락 후 횡보 중인 종목
결과를 HTML 메일로 발송합니다.

환경변수
  GMAIL_USER          보내는 Gmail 주소
  GMAIL_APP_PASSWORD  Gmail 앱 비밀번호(16자리)
  MAIL_TO             받는 주소 (쉼표로 여러 명 가능)
  DRY_RUN=1           메일 대신 report.html 파일로만 저장
"""
import os
import re
import sys
import time
import smtplib
import datetime as dt
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import requests

# ───────── 설정값 (여기만 바꾸면 기준이 바뀝니다) ─────────
MIN_TRADING_VALUE = 1_000_000_000   # 전일 거래대금 하한: 10억 원
TOP_N_MOVERS = 20                   # 등락률 상·하위 각 N개
TOP_N_STREAK = 30                   # 3일 연속 상승 최대 표시 개수
TOP_N_SIDEWAYS = 30                 # 횡보 종목 최대 표시 개수
STREAK_DAYS = 3                     # 연속 상승 일수
HIGH_LOOKBACK = 120                 # 약 6개월(거래일)
DRAWDOWN_MIN = 0.30                 # 고가 대비 30% 이상 하락
SIDEWAYS_WINDOW = 20                # 횡보 판단 기간(거래일)
SIDEWAYS_RANGE_MAX = 0.10           # 기간 내 (최고-최저)/최저 ≤ 10%
HISTORY_COUNT = 140                 # 받아올 일봉 개수
WORKERS = 8
# ─────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Referer": "https://m.stock.naver.com/"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _get(url, **kw):
    for i in range(3):
        try:
            r = SESSION.get(url, timeout=10, **kw)
            r.raise_for_status()
            return r
        except Exception:
            if i == 2:
                raise
            time.sleep(1.5 * (i + 1))


# ───────── 1. 종목 목록 ─────────
def list_market_naver(market):
    rows, page = [], 1
    while True:
        url = f"https://m.stock.naver.com/api/stocks/marketValue/{market}?page={page}&pageSize=100"
        data = _get(url).json()
        stocks = data.get("stocks", [])
        if not stocks:
            break
        for s in stocks:
            rows.append({"code": s.get("itemCode"), "name": s.get("stockName"),
                         "market": market, "type": s.get("stockEndType", "stock")})
        total = data.get("totalCount") or 0
        if page * 100 >= total:
            break
        page += 1
    return rows


def list_universe():
    try:
        rows = list_market_naver("KOSPI") + list_market_naver("KOSDAQ")
        df = pd.DataFrame(rows)
        if len(df) < 1000:
            raise RuntimeError(f"종목 수가 너무 적음: {len(df)}")
    except Exception as e:
        print(f"[warn] 네이버 종목목록 실패({e}) → FinanceDataReader로 대체", file=sys.stderr)
        import FinanceDataReader as fdr
        k = fdr.StockListing("KRX")
        k = k[k["Market"].isin(["KOSPI", "KOSDAQ"])]
        df = pd.DataFrame({"code": k["Code"], "name": k["Name"],
                           "market": k["Market"], "type": "stock"})
    return filter_universe(df)


def filter_universe(df):
    df = df.dropna(subset=["code", "name"]).drop_duplicates("code")
    df = df[df["type"].fillna("stock").str.lower() == "stock"]          # ETF/ETN 제외
    df = df[df["code"].str.fullmatch(r"\d{5}0")]                         # 우선주(끝자리≠0) 제외
    df = df[~df["name"].str.contains(r"스팩|\d+호$|우$|우B$|우C$", regex=True)]  # 스팩·우선주 제외
    return df.reset_index(drop=True)


# ───────── 2. 일봉 데이터 ─────────
def fetch_daily(code, count=HISTORY_COUNT):
    url = ("https://fchart.stock.naver.com/sise.nhn"
           f"?symbol={code}&timeframe=day&count={count}&requestType=0")
    r = _get(url)
    root = ET.fromstring(r.content.decode("euc-kr", errors="ignore"))
    recs = []
    for item in root.iter("item"):
        d, o, h, l, c, v = item.get("data").split("|")
        recs.append((d, float(o), float(h), float(l), float(c), float(v)))
    df = pd.DataFrame(recs, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df[df["volume"] > 0].reset_index(drop=True)  # 거래정지일 제거


# ───────── 3. 조건 분석 ─────────
def analyze(code, hist):
    if len(hist) < STREAK_DAYS + 2:
        return None
    c, h, l = hist["close"].values, hist["high"].values, hist["low"].values
    last = hist.iloc[-1]
    out = {
        "code": code,
        "date": last["date"],
        "close": c[-1],
        "chg": c[-1] / c[-2] - 1,
        "value": c[-1] * last["volume"],  # 거래대금 근사치(종가×거래량)
    }
    # 연속 상승
    streak = 0
    for i in range(len(c) - 1, 0, -1):
        if c[i] > c[i - 1]:
            streak += 1
        else:
            break
    out["streak"] = streak
    out["ret_n"] = c[-1] / c[-1 - STREAK_DAYS] - 1

    # 6개월 고가 대비 하락 + 횡보
    out["sideways"] = False
    if len(hist) >= HIGH_LOOKBACK:
        hi = h[-HIGH_LOOKBACK:].max()
        dd = 1 - c[-1] / hi
        win_hi, win_lo = h[-SIDEWAYS_WINDOW:].max(), l[-SIDEWAYS_WINDOW:].min()
        rng = win_hi / win_lo - 1
        out.update(high6m=hi, drawdown=dd, range20=rng)
        # 고점이 횡보 구간 안에 있으면 '하락 후 횡보'가 아니므로 제외
        hi_idx = int(h[-HIGH_LOOKBACK:].argmax())
        out["sideways"] = (dd >= DRAWDOWN_MIN and rng <= SIDEWAYS_RANGE_MAX
                           and hi_idx < HIGH_LOOKBACK - SIDEWAYS_WINDOW)
    return out


def build_frame(universe):
    results = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_daily, code): code for code in universe["code"]}
        for n, f in enumerate(as_completed(futs), 1):
            code = futs[f]
            try:
                r = analyze(code, f.result())
                if r:
                    results.append(r)
            except Exception as e:
                print(f"[skip] {code}: {e}", file=sys.stderr)
            if n % 500 == 0:
                print(f"  {n}/{len(futs)} 처리", file=sys.stderr)
    df = pd.DataFrame(results).merge(universe[["code", "name", "market"]], on="code")
    # 가장 최근 거래일 기준 데이터만 사용
    latest = df["date"].max()
    df = df[df["date"] == latest]
    df = df[df["value"] >= MIN_TRADING_VALUE]
    return df, latest


# ───────── 4. 리포트 ─────────
def fmt_table(df, cols):
    if df.empty:
        return "<p style='color:#888'>해당 종목 없음</p>"
    th = "".join(f"<th style='padding:6px 10px;border-bottom:2px solid #333;text-align:right'>{t}</th>"
                 for t, _ in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for _, fn in cols:
            v = fn(r)
            color = ""
            if isinstance(v, tuple):
                v, color = v
            tds.append(f"<td style='padding:5px 10px;border-bottom:1px solid #eee;"
                       f"text-align:right;{color}'>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<table style='border-collapse:collapse;font-size:13px;margin-bottom:8px'>"
            f"<tr>{th}</tr>{''.join(rows)}</table>")


def pct(x, signed=True):
    s = f"{x*100:+.2f}%" if signed else f"{x*100:.1f}%"
    color = "color:#d62728" if x > 0 else ("color:#1f5fbf" if x < 0 else "")
    return (s, color) if signed else s


def naver_link(r):
    return (f"<a href='https://finance.naver.com/item/main.naver?code={r['code']}' "
            f"style='color:#222;text-decoration:none'>{r['name']}</a>")


BASE_COLS = [
    ("종목", naver_link),
    ("시장", lambda r: r["market"]),
    ("종가", lambda r: f"{r['close']:,.0f}"),
]
VALUE_COL = ("거래대금(억)", lambda r: f"{r['value']/1e8:,.0f}")


def make_report(df, latest):
    gainers = df.sort_values("chg", ascending=False).head(TOP_N_MOVERS)
    losers = df.sort_values("chg").head(TOP_N_MOVERS)
    streak = (df[df["streak"] >= STREAK_DAYS]
              .sort_values("ret_n", ascending=False).head(TOP_N_STREAK))
    side = (df[df["sideways"]]
            .sort_values("drawdown", ascending=False).head(TOP_N_SIDEWAYS))

    movers_cols = BASE_COLS + [("등락률", lambda r: pct(r["chg"])), VALUE_COL]
    streak_cols = BASE_COLS + [("연속상승(일)", lambda r: r["streak"]),
                               (f"{STREAK_DAYS}일 수익률", lambda r: pct(r["ret_n"])),
                               ("전일", lambda r: pct(r["chg"])), VALUE_COL]
    side_cols = BASE_COLS + [("6개월 고가", lambda r: f"{r['high6m']:,.0f}"),
                             ("고가대비", lambda r: pct(-r["drawdown"])),
                             ("20일 변동폭", lambda r: pct(r["range20"], signed=False)),
                             VALUE_COL]

    d = latest.strftime("%Y-%m-%d (%a)")
    sec = lambda t, s, body: (f"<h2 style='font-size:16px;margin:28px 0 4px'>{t}</h2>"
                              f"<p style='color:#666;font-size:12px;margin:0 0 8px'>{s}</p>{body}")
    html = f"""
<div style="font-family:'Malgun Gothic',Apple SD Gothic Neo,sans-serif;color:#222;max-width:760px">
<h1 style="font-size:20px;margin-bottom:4px">📈 한국 주식 데일리 리포트</h1>
<p style="color:#666;margin-top:0">기준 거래일: <b>{d}</b> · 분석 종목 {len(df):,}개
(코스피·코스닥, 거래대금 {MIN_TRADING_VALUE/1e8:,.0f}억 이상, ETF·우선주·스팩 제외)</p>
{sec(f"1. 등락률 상위 {TOP_N_MOVERS}", "전일 종가 대비", fmt_table(gainers, movers_cols))}
{sec(f"2. 등락률 하위 {TOP_N_MOVERS}", "전일 종가 대비", fmt_table(losers, movers_cols))}
{sec(f"3. {STREAK_DAYS}거래일 이상 연속 상승 ({len(df[df['streak'] >= STREAK_DAYS])}개 중 상위 {len(streak)})",
     f"매일 종가가 전일보다 높은 종목, {STREAK_DAYS}일 누적 수익률 순", fmt_table(streak, streak_cols))}
{sec(f"4. 6개월 고가 대비 -{DRAWDOWN_MIN*100:.0f}% 이상 하락 후 횡보 ({len(df[df['sideways']])}개)",
     f"최근 {HIGH_LOOKBACK}거래일 최고가 대비 {DRAWDOWN_MIN*100:.0f}% 이상 하락 + 최근 {SIDEWAYS_WINDOW}거래일 "
     f"(최고-최저)/최저 ≤ {SIDEWAYS_RANGE_MAX*100:.0f}%, 하락률 순", fmt_table(side, side_cols))}
<p style="color:#999;font-size:11px;margin-top:32px">데이터: 네이버 금융 · 거래대금은 종가×거래량 근사치 ·
투자 권유가 아닌 참고용 자동 리포트입니다.</p>
</div>"""
    subject = (f"[주식리포트] {latest:%m/%d} 상승1위 {gainers.iloc[0]['name']} "
               f"{gainers.iloc[0]['chg']*100:+.1f}% · 연속상승 {len(df[df['streak'] >= STREAK_DAYS])} · "
               f"횡보 {len(df[df['sideways']])}") if not gainers.empty else f"[주식리포트] {latest:%m/%d}"
    return subject, html


def send_mail(subject, html):
    user, pw = os.environ["GMAIL_USER"], os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
    to = [a.strip() for a in os.environ.get("MAIL_TO", user).split(",") if a.strip()]
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(to)
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, to, msg.as_string())
    print(f"메일 발송 완료 → {to}")


def main():
    t0 = time.time()
    uni = list_universe()
    print(f"대상 종목 {len(uni)}개, 일봉 수집 시작", file=sys.stderr)
    df, latest = build_frame(uni)
    subject, html = make_report(df, latest)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{subject}  ({time.time()-t0:.0f}s)")
    if os.environ.get("DRY_RUN") != "1":
        send_mail(subject, html)


if __name__ == "__main__":
    main()
