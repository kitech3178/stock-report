"""
한국 주식 데일리 리포트 에이전트
- 전일 등락률 상위/하위 종목
- 3거래일 연속 상승 종목
- 최근 6개월 고가 대비 -30% 이상 하락 후 횡보 중인 종목
결과를 HTML 메일로 발송합니다.

환경변수
  GMAIL_USER          보내는 Gmail 주소
  GMAIL_APP_PASSWORD  Gmail 앱 비밀번호(16자리)
  MAIL_TO             받는 사람 주소 (쉼표로 여러 명 가능, 메일 헤더에 보임)
  MAIL_BCC            숨은 참조 주소 (쉼표로 여러 명 가능, 서로에게 보이지 않음)
  DRY_RUN=1           메일 대신 report.html 파일로만 저장
  ANTHROPIC_API_KEY        있으면 Claude API(SDK)로 섹션별 분석 코멘트를 씁니다
  CLAUDE_CODE_OAUTH_TOKEN  API 키가 없을 때 Claude 구독(claude setup-token)으로 코멘트를 씁니다
                           (둘 다 없으면 코멘트 없이 발송)
  LLM_COMMENT=0            AI 코멘트 끄기
  LLM_MODEL                코멘트에 쓸 모델 (API 기본 claude-opus-5, 구독은 Claude Code 기본값)
"""
import os
import re
import sys
import json
import shutil
import subprocess
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
LLM_MODEL = os.environ.get("LLM_MODEL")   # 비우면 API는 claude-opus-5, 구독은 Claude Code 기본 모델
LLM_API_DEFAULT_MODEL = "claude-opus-5"
LLM_MAX_TOKENS = 16000
LLM_CLI_TIMEOUT = 600                     # 구독(claude -p) 호출 제한 시간(초)
LLM_WATCHLIST_N = 5                 # AI가 고르는 관심 종목 수
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


# ───────── 4. AI 분석 코멘트 (Claude) ─────────
COMMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "market_summary": {"type": "string",
                           "description": "오늘 표 전체를 훑은 3~5문장 시장 총평"},
        "gainers": {"type": "string", "description": "등락률 상위 표 코멘트 2~4문장"},
        "losers": {"type": "string", "description": "등락률 하위 표 코멘트 2~4문장"},
        "streak": {"type": "string", "description": "연속 상승 표 코멘트 2~4문장"},
        "sideways": {"type": "string", "description": "고점 대비 하락 후 횡보 표 코멘트 2~4문장"},
        "watchlist": {
            "type": "array",
            "description": f"표 안에서 추가로 살펴볼 만한 종목 최대 {LLM_WATCHLIST_N}개",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "표에 있는 종목명 그대로"},
                    "reason": {"type": "string", "description": "왜 볼 만한지 1~2문장"},
                },
                "required": ["name", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["market_summary", "gainers", "losers", "streak", "sideways", "watchlist"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """당신은 한국 주식 데일리 리포트에 짧은 해설을 붙이는 애널리스트입니다.

- 근거는 오직 사용자가 준 표의 숫자입니다. 표에 없는 뉴스·실적·공시를 아는 것처럼 쓰지 마세요.
  이유를 모르면 "이유는 표만으로는 알 수 없다"고 쓰면 됩니다.
- 같은 업종·테마로 묶이는 종목이 여럿 보이면 그 묶음을 짚어 주세요.
- 상위/하위 표에서는 거래대금이 큰 종목과 작은 종목을 구분해서 보세요. 거래대금이 작은 급등은 그렇게 표시하세요.
- 연속 상승 표에서는 상승 폭이 과열인지, 거래대금이 받쳐 주는지를 보세요.
- 횡보 표에서는 하락 폭과 20일 변동폭을 함께 보고, 바닥 다지기인지 아직 판단하기 이른지를 씁니다.
- 매수·매도 권유는 하지 않습니다. 관찰, 주의점, 확인할 것 위주로 씁니다.
- 한국어 존댓말, 각 항목은 지정된 문장 수를 지킵니다. 종목명은 표에 있는 그대로 씁니다.
"""


def _rows_text(df, fields):
    """LLM에 넘길 표를 한 줄에 한 종목씩 짧은 텍스트로 만든다."""
    lines = []
    for _, r in df.iterrows():
        parts = [f"{r['name']}({r['market']})", f"종가 {r['close']:,.0f}"]
        parts += [fn(r) for fn in fields]
        parts.append(f"거래대금 {r['value']/1e8:,.0f}억")
        lines.append(" | ".join(parts))
    return "\n".join(lines) if lines else "(없음)"


def build_llm_prompt(df, latest, sec):
    gainers, losers, streak, side = sec["gainers"], sec["losers"], sec["streak"], sec["side"]
    chg = lambda r: f"등락 {r['chg']*100:+.2f}%"
    return f"""기준 거래일: {latest:%Y-%m-%d}
분석 종목 {len(df):,}개 (코스피·코스닥, 거래대금 {MIN_TRADING_VALUE/1e8:,.0f}억 이상, ETF·우선주·스팩 제외)
전체 종목 등락률 중앙값 {df['chg'].median()*100:+.2f}%, 상승 {int((df['chg'] > 0).sum())}개 / 하락 {int((df['chg'] < 0).sum())}개

[1. 등락률 상위 {len(gainers)}]
{_rows_text(gainers, [chg])}

[2. 등락률 하위 {len(losers)}]
{_rows_text(losers, [chg])}

[3. {STREAK_DAYS}거래일 이상 연속 상승 (전체 {int((df['streak'] >= STREAK_DAYS).sum())}개 중 상위 {len(streak)})]
{_rows_text(streak, [lambda r: f"연속 {r['streak']}일", lambda r: f"{STREAK_DAYS}일 수익률 {r['ret_n']*100:+.2f}%", chg])}

[4. {HIGH_LOOKBACK}거래일 고가 대비 -{DRAWDOWN_MIN*100:.0f}% 이상 하락 후 횡보 (전체 {int(df['sideways'].sum())}개 중 상위 {len(side)})]
{_rows_text(side, [lambda r: f"6개월 고가 {r['high6m']:,.0f}", lambda r: f"고가대비 {-r['drawdown']*100:.1f}%", lambda r: f"20일 변동폭 {r['range20']*100:.1f}%"])}

위 네 표를 보고 시장 총평, 표별 코멘트, 관심 종목 최대 {LLM_WATCHLIST_N}개를 JSON으로 써 주세요."""


def llm_comments(df, latest, sec):
    """Claude에게 표를 보여주고 섹션별 코멘트를 받는다. 실패하면 None (리포트는 코멘트 없이 나감).

    ANTHROPIC_API_KEY 가 있으면 API(SDK), 없고 CLAUDE_CODE_OAUTH_TOKEN 이 있으면
    Claude Code CLI(구독)로 호출한다.
    """
    if os.environ.get("LLM_COMMENT") == "0":
        print("[info] LLM_COMMENT=0 → AI 코멘트 생략", file=sys.stderr)
        return None
    prompt = build_llm_prompt(df, latest, sec)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _comments_via_api(prompt)
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return _comments_via_claude_code(prompt)
    print("[info] ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN 없음 → AI 코멘트 생략", file=sys.stderr)
    return None


def _comments_via_api(prompt):
    """Anthropic SDK 경로 (API 키, 크레딧 과금)."""
    import anthropic

    client = anthropic.Anthropic()
    model = LLM_MODEL or LLM_API_DEFAULT_MODEL
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": COMMENT_SCHEMA}},
            # 안전 정책으로 거절되면 서버가 다른 모델로 같은 요청을 다시 실행
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        print("[warn] ANTHROPIC_API_KEY가 잘못됨 → AI 코멘트 생략", file=sys.stderr)
        return None
    except anthropic.RateLimitError:
        print("[warn] Anthropic 사용량 한도 초과 → AI 코멘트 생략", file=sys.stderr)
        return None
    except anthropic.APIStatusError as e:
        print(f"[warn] Anthropic API 오류 {e.status_code}: {e.message} → AI 코멘트 생략", file=sys.stderr)
        return None
    except anthropic.APIConnectionError as e:
        print(f"[warn] Anthropic 접속 실패: {e} → AI 코멘트 생략", file=sys.stderr)
        return None

    if resp.stop_reason == "refusal":
        print("[warn] 모델이 응답을 거절함 → AI 코멘트 생략", file=sys.stderr)
        return None
    if resp.stop_reason == "max_tokens":
        print("[warn] AI 코멘트가 max_tokens에서 잘림 → 생략", file=sys.stderr)
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        print("[warn] AI 응답에 텍스트 없음 → 생략", file=sys.stderr)
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[warn] AI 응답 JSON 파싱 실패: {e} → 생략", file=sys.stderr)
        return None
    u = resp.usage
    print(f"AI 코멘트 완료 (API {resp.model}, 입력 {u.input_tokens:,} / 출력 {u.output_tokens:,} 토큰)",
          file=sys.stderr)
    data["model"] = resp.model
    return data


def _comments_via_claude_code(prompt):
    """Claude Code CLI 경로 (구독 토큰 CLAUDE_CODE_OAUTH_TOKEN, 구독 한도 차감).

    --bare 는 OAuth 토큰을 읽지 않으므로 쓰지 않는다. --tools "" 로 도구를 모두 끄고
    --json-schema 로 구조화 출력을 받는다(결과의 structured_output 필드).
    """
    exe = shutil.which("claude")
    if not exe:
        print("[warn] claude 명령을 찾을 수 없음 (Claude Code CLI 미설치) → AI 코멘트 생략", file=sys.stderr)
        return None
    cmd = [exe, "-p", prompt,
           "--system-prompt", SYSTEM_PROMPT,
           "--tools", "",
           "--permission-mode", "dontAsk",
           "--no-session-persistence",
           "--output-format", "json",
           "--json-schema", json.dumps(COMMENT_SCHEMA, ensure_ascii=False)]
    if LLM_MODEL:
        cmd += ["--model", LLM_MODEL]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=LLM_CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[warn] claude -p 가 {LLM_CLI_TIMEOUT}초 안에 끝나지 않음 → AI 코멘트 생략", file=sys.stderr)
        return None
    if proc.returncode != 0 and not proc.stdout.strip():
        print(f"[warn] claude -p 실패 (exit {proc.returncode}): {proc.stderr.strip()[:500]} → AI 코멘트 생략",
              file=sys.stderr)
        return None
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"[warn] claude -p 출력 JSON 파싱 실패: {e}: {proc.stdout[:300]!r} → 생략", file=sys.stderr)
        return None
    if out.get("is_error"):
        print(f"[warn] claude -p 오류: {str(out.get('result'))[:500]} → AI 코멘트 생략", file=sys.stderr)
        return None
    data = out.get("structured_output")
    if not isinstance(data, dict):
        print(f"[warn] claude -p 결과에 structured_output 없음: {str(out.get('result'))[:300]} → 생략",
              file=sys.stderr)
        return None
    # modelUsage 에는 제목 생성 등 부수 작업에 쓰인 작은 모델도 섞여 있으므로
    # 출력 토큰이 가장 많은 모델을 코멘트를 쓴 모델로 본다.
    mu = out.get("modelUsage") or {}
    def _out_tokens(m):
        v = mu.get(m) or {}
        return v.get("outputTokens") or v.get("output_tokens") or 0
    model = max(mu, key=_out_tokens) if mu else (LLM_MODEL or "claude-code")
    used = ", ".join(f"{m}: 출력 {_out_tokens(m):,}" for m in sorted(mu, key=_out_tokens, reverse=True))
    u = out.get("usage") or {}
    inp = sum(u.get(k, 0) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    print(f"AI 코멘트 완료 (구독 {model}, 입력 {inp:,} / 출력 {u.get('output_tokens', 0):,} 토큰; "
          f"사용 모델 {used or '알 수 없음'})", file=sys.stderr)
    data["model"] = model
    return data


# ───────── 5. 리포트 ─────────
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


def select_sections(df):
    """리포트 네 표에 들어갈 종목을 고른다. HTML과 AI 코멘트가 같은 표를 보도록 한 곳에서 계산."""
    return {
        "gainers": df.sort_values("chg", ascending=False).head(TOP_N_MOVERS),
        "losers": df.sort_values("chg").head(TOP_N_MOVERS),
        "streak": (df[df["streak"] >= STREAK_DAYS]
                   .sort_values("ret_n", ascending=False).head(TOP_N_STREAK)),
        "side": (df[df["sideways"]]
                 .sort_values("drawdown", ascending=False).head(TOP_N_SIDEWAYS)),
    }


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ai_box(comments):
    """리포트 맨 위 AI 총평 + 관심 종목 상자."""
    if not comments:
        return ""
    items = "".join(f"<li style='margin:2px 0'><b>{esc(w['name'])}</b> — {esc(w['reason'])}</li>"
                    for w in comments.get("watchlist", []))
    watch = (f"<p style='margin:10px 0 2px;font-weight:bold'>관심 종목</p>"
             f"<ul style='margin:0;padding-left:18px'>{items}</ul>") if items else ""
    return (f"<div style='background:#f4f6fa;border-left:4px solid #4a6fd6;padding:12px 14px;"
            f"margin:16px 0;font-size:13px;line-height:1.6'>"
            f"<p style='margin:0 0 4px;font-weight:bold'>🤖 AI 총평</p>"
            f"<p style='margin:0'>{esc(comments['market_summary'])}</p>{watch}</div>")


def ai_note(comments, key):
    """섹션 표 위에 붙는 한 문단 코멘트."""
    if not comments or not comments.get(key):
        return ""
    return (f"<p style='font-size:13px;line-height:1.6;background:#fafafa;border:1px solid #eee;"
            f"padding:8px 10px;margin:0 0 8px'>{esc(comments[key])}</p>")


def make_report(df, latest, sec, comments=None):
    gainers, losers, streak, side = sec["gainers"], sec["losers"], sec["streak"], sec["side"]

    movers_cols = BASE_COLS + [("등락률", lambda r: pct(r["chg"])), VALUE_COL]
    streak_cols = BASE_COLS + [("연속상승(일)", lambda r: r["streak"]),
                               (f"{STREAK_DAYS}일 수익률", lambda r: pct(r["ret_n"])),
                               ("전일", lambda r: pct(r["chg"])), VALUE_COL]
    side_cols = BASE_COLS + [("6개월 고가", lambda r: f"{r['high6m']:,.0f}"),
                             ("고가대비", lambda r: pct(-r["drawdown"])),
                             ("20일 변동폭", lambda r: pct(r["range20"], signed=False)),
                             VALUE_COL]

    d = latest.strftime("%Y-%m-%d (%a)")
    sec_html = lambda t, s, key, body: (
        f"<h2 style='font-size:16px;margin:28px 0 4px'>{t}</h2>"
        f"<p style='color:#666;font-size:12px;margin:0 0 8px'>{s}</p>{ai_note(comments, key)}{body}")
    html = f"""
<div style="font-family:'Malgun Gothic',Apple SD Gothic Neo,sans-serif;color:#222;max-width:760px">
<h1 style="font-size:20px;margin-bottom:4px">📈 한국 주식 데일리 리포트</h1>
<p style="color:#666;margin-top:0">기준 거래일: <b>{d}</b> · 분석 종목 {len(df):,}개
(코스피·코스닥, 거래대금 {MIN_TRADING_VALUE/1e8:,.0f}억 이상, ETF·우선주·스팩 제외)</p>
{ai_box(comments)}
{sec_html(f"1. 등락률 상위 {TOP_N_MOVERS}", "전일 종가 대비", "gainers", fmt_table(gainers, movers_cols))}
{sec_html(f"2. 등락률 하위 {TOP_N_MOVERS}", "전일 종가 대비", "losers", fmt_table(losers, movers_cols))}
{sec_html(f"3. {STREAK_DAYS}거래일 이상 연속 상승 ({len(df[df['streak'] >= STREAK_DAYS])}개 중 상위 {len(streak)})",
     f"매일 종가가 전일보다 높은 종목, {STREAK_DAYS}일 누적 수익률 순", "streak", fmt_table(streak, streak_cols))}
{sec_html(f"4. 6개월 고가 대비 -{DRAWDOWN_MIN*100:.0f}% 이상 하락 후 횡보 ({len(df[df['sideways']])}개)",
     f"최근 {HIGH_LOOKBACK}거래일 최고가 대비 {DRAWDOWN_MIN*100:.0f}% 이상 하락 + 최근 {SIDEWAYS_WINDOW}거래일 "
     f"(최고-최저)/최저 ≤ {SIDEWAYS_RANGE_MAX*100:.0f}%, 하락률 순", "sideways", fmt_table(side, side_cols))}
<p style="color:#999;font-size:11px;margin-top:32px">데이터: 네이버 금융 · 거래대금은 종가×거래량 근사치 ·
{"AI 코멘트: " + esc(comments["model"]) + " · " if comments else ""}투자 권유가 아닌 참고용 자동 리포트입니다.</p>
</div>"""
    subject = f"[ {latest:%y}년 {latest:%m}월 {latest:%d}일 - 주식 레포트 ]"
    return subject, html


def send_mail(subject, html):
    user = os.environ.get("GMAIL_USER", "").strip()
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not user or not pw:
        sys.exit("[error] GMAIL_USER / GMAIL_APP_PASSWORD 가 비어 있습니다. "
                 "저장소 Settings → Secrets and variables → Actions 에 등록하세요. "
                 "(report.html 은 Artifacts 에 저장되어 있습니다)")
    split = lambda v: [a.strip() for a in v.split(",") if a.strip()]
    to = split(os.environ.get("MAIL_TO", user))
    bcc = [a for a in split(os.environ.get("MAIL_BCC", "")) if a not in to]
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(to)
    # 숨은 참조는 헤더에 넣지 않고 봉투(sendmail 수신자)에만 넣는다
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, to + bcc, msg.as_string())
    print(f"메일 발송 완료 → 받는 사람 {to}" + (f", 숨은 참조 {bcc}" if bcc else ""))


def main():
    t0 = time.time()
    uni = list_universe()
    print(f"대상 종목 {len(uni)}개, 일봉 수집 시작", file=sys.stderr)
    df, latest = build_frame(uni)
    sec = select_sections(df)
    comments = llm_comments(df, latest, sec)
    subject, html = make_report(df, latest, sec, comments)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{subject}  ({time.time()-t0:.0f}s)")
    if os.environ.get("DRY_RUN") != "1":
        send_mail(subject, html)


if __name__ == "__main__":
    main()
