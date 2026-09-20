# 한국 주식 데일리 리포트 에이전트

매주 월~금 아침 8시 45분(한국시간)에 전 거래일 기준 리포트를 boribab73@gmail.com, yozosukny0@naver.com으로 보냅니다.
받는 주소는 `.github/workflows/daily-report.yml`에서 바꿀 수 있습니다. `MAIL_TO`는 받는 사람(헤더에 보임), `MAIL_BCC`는 숨은 참조(서로에게 보이지 않음)이고 둘 다 쉼표로 여러 명을 적을 수 있습니다.

| 섹션 | 기준 |
|---|---|
| 등락률 상위/하위 20 | 전일 종가 대비 |
| 3일 연속 상승 | 3거래일 이상 매일 종가 상승, 3일 누적 수익률 순 |
| 고점 대비 하락 후 횡보 | 최근 6개월(120거래일) 최고가 대비 -30% 이하, 최근 20거래일 (최고-최저)/최저 ≤ 10%, 고점이 최근 20일 이전 |
| 시총 상위 10종목 주간 비교 | 최근 5거래일 vs 그 이전 5거래일의 등락률·고가·저가·거래량·거래대금 변화, 외국인·기관 7거래일 순매수 |
| 외국인·기관 섹터별 순매매 | 전 종목의 최근 7거래일 외국인·기관 순매수(수량×종가 근사)를 네이버 업종 분류로 합산. 일별 시장 전체, 주체별 순매수·순매도 상위 8섹터(주요 종목 포함), 상위 10종목 |

대상: 코스피와 코스닥 종목 중 거래대금 10억 원 이상. ETF, 우선주, 스팩은 뺍니다.
기준값은 `stock_report.py` 맨 위의 설정값 블록에서 바꿀 수 있습니다.

### AI 분석 코멘트 (선택)
아래 둘 중 하나를 시크릿으로 등록하면 Claude가 네 표를 읽고 리포트에 코멘트를 붙입니다.

| 시크릿 | 방식 | 비용 |
|---|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Pro/Max 구독. 내 컴퓨터에서 `claude setup-token` 실행 후 나온 토큰 | 구독 한도에서 차감, 추가 결제 없음 |
| `ANTHROPIC_API_KEY` | 개발자 플랫폼(https://platform.claude.com) API 키 | 크레딧 충전 후 토큰당 과금 |

둘 다 있으면 API 키를 먼저 씁니다. 구독 토큰은 1년 뒤 만료되니 그때 다시 만들어 갱신하세요.

- 맨 위에 **AI 총평**과 **관심 종목** 최대 5개
- 각 표 위에 2~4문장 코멘트 (업종·테마 묶음, 거래대금이 작은 급등, 과열 여부, 바닥 다지기 판단 등)

코멘트는 표에 있는 숫자만 근거로 쓰고, 매수·매도 권유는 하지 않도록 지시되어 있습니다.
키가 없거나 API 호출이 실패하면 코멘트 없이 기존 리포트가 그대로 발송됩니다.

## 설정 (한 번만, 약 10분)

### 1. Gmail 앱 비밀번호 발급
1. 메일을 **보낼** Gmail 계정에 로그인합니다(받는 주소와 같아도 됩니다).
2. https://myaccount.google.com/security 에서 **2단계 인증**을 켭니다.
3. https://myaccount.google.com/apppasswords 에서 이름을 `stock-report`로 정하고 만들기를 누릅니다.
4. 화면에 나온 16자리 비밀번호를 복사해 둡니다.

### 2. GitHub 저장소 만들기
1. https://github.com/new 에서 **Private** 저장소를 만듭니다(예: `stock-report`).
2. 이 폴더의 파일을 모두 올립니다. `.github/workflows/daily-report.yml`도 **꼭** 같은 경로로 올려야 합니다.
   - 웹에서 올릴 때는 "Add file → Create new file"을 누르고 파일 이름에 `.github/workflows/daily-report.yml`을 입력한 뒤 내용을 붙여넣으면 됩니다.

### 3. Secrets 등록
저장소의 **Settings → Secrets and variables → Actions → New repository secret**에서 아래 두 값을 등록합니다.

| Name | 값 |
|---|---|
| `GMAIL_USER` | 보내는 Gmail 주소 |
| `GMAIL_APP_PASSWORD` | 1단계에서 받은 16자리 비밀번호 |
| `CLAUDE_CODE_OAUTH_TOKEN` | (선택) AI 코멘트용. 구독 계정으로 `claude setup-token` 실행해 발급 |
| `ANTHROPIC_API_KEY` | (선택) AI 코멘트용. https://platform.claude.com 에서 발급 |

### 4. 테스트
**Actions** 탭에서 "한국주식 데일리 리포트"를 고르고 **Run workflow**를 누릅니다.
2~5분쯤 지나면 메일이 옵니다. 실행 기록의 Artifacts에서 `report.html`도 받을 수 있습니다.

## 메일 없이 시험 실행하기
Actions 탭의 **드라이런 실행 (수동)** 워크플로를 실행하면 메일을 보내지 않고 리포트만 만듭니다.
로그에 5·6번 섹션이 텍스트로 찍히고, Artifacts에서 `report.html`을 받을 수 있습니다.

## 로컬에서 실행하기 (선택)
```bash
pip install -r requirements.txt
DRY_RUN=1 python stock_report.py        # 메일은 보내지 않고 report.html만 만듭니다
```
Windows PowerShell에서는 `$env:DRY_RUN=1; python stock_report.py`로 실행합니다.

AI 코멘트 관련 환경변수:

| 변수 | 설명 |
|---|---|
| `ANTHROPIC_API_KEY` | 있으면 API로 AI 코멘트 생성 |
| `CLAUDE_CODE_OAUTH_TOKEN` | API 키가 없을 때 구독(Claude Code CLI)으로 AI 코멘트 생성. 로컬에서는 `claude` 명령이 설치돼 있어야 함 |
| `LLM_COMMENT=0` | 키가 있어도 AI 코멘트 끄기 |
| `LLM_MODEL` | 사용할 모델. 비우면 API는 `claude-opus-5`, 구독은 Claude Code 기본 모델 |

## 참고
- GitHub의 예약 실행은 부하에 따라 10~30분 늦게 시작될 수 있습니다.
- 공휴일에는 마지막 거래일 기준으로 리포트가 다시 옵니다. 리포트의 기준 거래일 표시를 확인하세요.
- 저장소에 60일 동안 활동이 없으면 GitHub가 예약 실행을 멈춥니다. 이때 알림 메일이 오는데, 그 메일에서 다시 켜면 됩니다.
- 데이터는 네이버 금융에서 가져옵니다. 거래대금은 종가×거래량으로 계산한 근사치입니다.
- AI 코멘트는 실행 1회당 입력 약 4천 토큰, 출력 약 1~2천 토큰을 씁니다. 요청이 안전 정책으로 거절되면 서버가 다른 Claude 모델로 자동 재시도합니다(`fallbacks="default"`).
