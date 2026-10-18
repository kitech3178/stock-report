# UR5 + OnRobot RG2 + 손목 Depth 카메라 — 강화학습 파지(Grasp) 시뮬레이터

**Q-learning(표 기반)과 DQN(신경망)을 같은 로봇 파지 문제에서 비교**하기 위한 실습/리포트용 프로젝트입니다.

* 로봇: **UR5** (표준 DH 파라미터, 수치 역기구학)
* 그리퍼: **OnRobot RG2** (스트로크 0–110 mm, 파지력 3–40 N, 마찰·미끄러짐 물리 모델)
* 센서: 그리퍼에 장착한 **depth 카메라** (Intel RealSense D405 급, 노이즈 모델 포함)
* 시뮬레이터 두 가지 (같은 MDP, 같은 에이전트 코드)
  * `simple` — 이 저장소의 **물리 모델**(numpy). 설치 즉시 동작, 한 스텝 ≈ 2 ms
  * `isaac` — **NVIDIA Isaac Sim**(PhysX + RTX depth 카메라). URDF는 설정 파일에서 자동 생성
* **State / Action / Reward 와 모든 하이퍼파라미터는 `configs/default.yaml` 하나**에서 바꾸고,
  명령줄 `--set key=value` 로 즉석에서 덮어쓸 수 있습니다.

---

## 1. 빠른 시작

```bash
cd rl_grasp_sim
pip install -r requirements.txt          # numpy, pyyaml, gymnasium, matplotlib, torch

python tests/test_core.py                # 물리/기구학/URDF/환경 검사 (몇 초)
python scripts/evaluate.py --policy scripted --episodes 50   # 손으로 짠 정답 제어기: 성공률 ~100%
python scripts/play.py                   # 직접 행동을 입력하며 State/Reward 확인

python scripts/train.py --algo qlearning                     # ~3분 (CPU)
python scripts/train.py --algo dqn --set dqn.observation=features
python scripts/compare.py --algos qlearning dqn_features dqn --seeds 0 1 2   # 리포트용 비교
python scripts/visualize.py --algo qlearning --model results/compare/qlearning_seed0/q_table.npz
```

결과는 `results/<실험이름>/` 에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `episodes.csv` | 에피소드별 return, 성공 여부, 길이, 결과(성공/미스/정렬불량/미끄러짐…), ε, TD오차/손실 |
| `eval.csv` | 주기적 greedy 평가(고정 시드 30개) 성공률 |
| `summary.json` | 최종 성능, 학습 시간, 파라미터 수(Q-table 크기 vs 신경망 가중치 수) |
| `comparison.png`, `summary.md` | `compare.py`/`sweep.py` 가 만드는 비교 그래프와 표 |
| `config.yaml` | 그 실행에 사용된 전체 설정 (재현용) |

---

## 2. 폴더 구조

```
rl_grasp_sim/
├── configs/
│   ├── default.yaml          ★ 모든 파라미터 (로봇·그리퍼·카메라·물체·State·Action·Reward·알고리즘)
│   └── isaac.yaml            Isaac Sim 실행용 덮어쓰기
├── grasp_rl/
│   ├── physics/
│   │   ├── ur5_kinematics.py UR5 DH 순기구학, 야코비안, 감쇠 최소제곱 역기구학
│   │   ├── rg2_gripper.py    RG2 충돌 형상, 닫힘, 마찰/비틀림/미끄러짐 + 들어올리기 동역학
│   │   ├── depth_camera.py   핀홀 depth 카메라 (레이캐스팅, 스테레오 노이즈 모델)
│   │   └── objects.py        상자/원통 물체, 레이-물체 교차, 2D 충돌(SAT)
│   ├── perception.py         depth 이미지 → 물체 분할 → (dx, dy, dz, yaw) 특징 추정
│   ├── envs/
│   │   ├── actions.py        ★ ACTION 정의
│   │   ├── state.py          ★ STATE 정의 (discrete / features / depth)
│   │   ├── reward.py         ★ REWARD 정의 (희소 보상 + 페널티 + 퍼텐셜 셰이핑)
│   │   └── grasp_env.py      Gymnasium 환경 (위 셋을 묶음)
│   ├── backends/             simple(내장 물리) / isaac(Isaac Sim) 시뮬레이터 교체 계층
│   ├── agents/
│   │   ├── q_learning.py     표 기반 Q-learning
│   │   ├── dqn.py            DQN (+Double DQN, n-step, 타깃망, 리플레이 버퍼, CNN/MLP)
│   │   └── scripted.py       정답 제어기 / 무작위 정책 (기준선)
│   ├── train_loop.py         학습·평가·로그 공통 루프
│   ├── analysis.py           비교 그래프와 표 생성
│   ├── render.py             3D 로봇 / 상면도 / depth 이미지 렌더링
│   └── urdf.py               설정 → UR5+RG2+카메라 URDF 생성 (Isaac Sim 용 물리 모델)
├── scripts/  train.py · compare.py · sweep.py · evaluate.py · visualize.py · play.py · make_urdf.py
├── assets/urdf/ur5_rg2.urdf  생성된 URDF (질량·관성·관절 한계·충돌 형상 포함)
└── tests/test_core.py
```

---

## 3. 문제 정의 (MDP)

에피소드 하나: 테이블 위 무작위 위치·방향의 물체(기본: 4×7×5 cm 상자, 0.3 kg) →
그리퍼가 물체 위 22 cm 에서 벌린 채 시작 → 이산 행동으로 손을 움직이고 → `grasp`
(닫기 → 10 cm 들어올리기 → 물리 판정) 하면 종료. 최대 60 스텝.

### 3.1 Action (`configs/default.yaml` → `actions`, 코드 `envs/actions.py`)

| 행동 | 의미 | 기본 크기 |
|---|---|---|
| `x+`, `x-` | 손가락 **닫힘축** 방향 이동 | 1.0 cm (`step_xy`) |
| `y+`, `y-` | 닫힘축에 수직 방향 이동 | 1.0 cm |
| `z-`, `z+` | 하강 / 상승 | 3.0 cm (`step_z`) |
| `yaw+`, `yaw-` | 손목 회전 | 15° (`step_yaw_deg`), ±90° 제한 |
| `grasp` | 닫기 → 들기 → 성공/실패 판정, **에피소드 종료** | — |

* `actions.frame: tool` 이면 이동이 그리퍼 기준(손이 돌면 x, y 방향도 같이 돈다), `world` 면 월드 기준.
* 이동 명령은 UR5 **역기구학**으로 관절각으로 바뀌며, 해가 없으면 `ik_fail`.
* 실행 오차 `robot.execution_noise_std` (기본 2 mm) 가 더해져 전이가 확률적입니다.
* `actions.set` 에서 행동을 빼거나 순서를 바꿀 수 있습니다 (예: yaw 행동을 빼고 `object.shape: cylinder` 로 단순화).

### 3.2 State (`state`, 코드 `perception.py`, `envs/state.py`)

손목 depth 카메라 이미지 → (로봇 순기구학 + 핸드-아이 보정으로) 3D 점군 → 테이블보다 6 mm 이상 높은
점을 물체로 분할 → 윗면 점들의 중심·주성분 방향으로 **그리퍼 기준 특징**을 추정합니다.

| 특징 | 의미 |
|---|---|
| `dx` | 물체 중심의 닫힘축 방향 오프셋 [m] |
| `dy` | 물체 중심의 수직 방향 오프셋 [m] |
| `dz` | 손가락 끝 높이 − 물체 윗면 높이 [m] (음수 = 물체 옆으로 내려가 있음) |
| `yaw` | 닫힘축과 물체의 **좁은 면** 방향의 각도 차 (−90°, 90°] |

관측 형식 (`qlearning.observation`, `dqn.observation`):

| 모드 | 사용 | 내용 |
|---|---|---|
| `discrete` | Q-learning | 위 4개 특징을 `state.bins` 의 경계로 잘라 만든 정수. 기본 7×7×6×7 + "안 보임" 1 = **2,059 상태** |
| `features` | DQN (MLP) | 같은 4개 특징의 연속값 (+ sin/cos 2·yaw, 가시성, 높이) |
| `depth` | DQN (CNN) | **원시 depth 이미지 64×48** + 손 높이·yaw. 특징 추출까지 신경망이 학습 |

`state.source: ground_truth` 로 바꾸면 카메라 대신 시뮬레이터 참값을 씁니다 (인식 오차의 영향 비교용).

### 3.3 Reward (`reward`, 코드 `envs/reward.py`)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `step` | −0.05 | 매 스텝 시간 비용 |
| `success` | +10 | 물체를 들어올리는 데 성공 |
| `grasp_fail` | −3 | 잡기 실패 (높이 부족, 빗나감, 정렬 불량, 미끄러짐, 떨어뜨림) |
| `collision` | −2 | 손가락/몸체가 물체나 테이블에 부딪힘 (이동은 막힘) |
| `ik_fail`, `out_of_workspace` | −1 | 도달 불가 / 작업 영역 밖 |
| `shaping` | γΦ(s′) − Φ(s) | 퍼텐셜 기반 셰이핑, Φ = −(20·수평거리 + 10·|높이오차| + 2·|yaw오차|) |

`terminate_on_grasp_fail`, `terminate_on_collision` 으로 실패 시 종료 여부를,
`shaping.enabled/gamma/terminal_potential/w_*` 로 셰이핑을 조절합니다. 아래 6장에 셰이핑 관련
중요한 관찰이 있습니다.

---

## 4. 물리 모델 (simple 백엔드)

### 4.1 UR5
표준 DH ( d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823], a = [0, −0.425, −0.39225, 0, 0, 0],
α = [π/2, 0, 0, π/2, −π/2, 0] ), `T_i = Rz(q_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`.
TCP = 플랜지 + 20 cm (RG2 손끝). 역기구학은 기하 야코비안을 이용한 감쇠 최소제곱
`Δq = Jᵀ(JJᵀ + λ²I)⁻¹ e` 를 현재 관절각에서 시작해 반복합니다.

### 4.2 OnRobot RG2
* 힘 제어 평행 그리퍼: 법선력 `F_n` = `gripper.force` (3–40 N), 닫힘 속도는 힘에 따라 38–127 mm/s.
* **충돌**: 벌린 손가락(두께 8 mm, 폭 20 mm)과 몸체의 바닥 사각형을 물체 바닥 형상과 SAT로 검사.
* **접촉 높이**: 패드(25 mm)와 물체 측면의 겹침 길이 `L` 이 6 mm 미만이면 `no_contact_height`.
* **자기 정렬**: 평행 집게는 물체를 닫힘축 방향으로 가운데 정렬시키고, yaw 오차가 20° 이하면 물체를 돌려 맞춥니다.
  그 이상이면 모서리 접촉으로 `misaligned`.
* **마찰 (Coulomb)**: 두 패드의 정지 마찰 한계
  `F_lin = 2·μs·F_n`,  비틀림 한계 `T_rot = 2·(2/3)·μs·F_n·ρ` (접촉 패치 반경 ρ).
* **들어올리기 동역학 (1 ms 적분)**: 사다리꼴 속도 프로파일(가속도 `task.lift_accel`)로 들어올릴 때
  * `m(g + a) > F_lin` → 미끄러짐 시작, 운동 마찰 `μk` 로 물체가 따로 움직임. 누적 미끄럼 > `L` 이면 `dropped`
  * `m(g + a)·r > T_rot` (r = 질량중심과 파지선 사이 거리) → 물체가 돌아 빠짐 `rotational_slip`
  → 그래서 **중심을 잘 맞추는 것, 충분한 깊이로 잡는 것, 파지력·마찰·질량**이 모두 결과에 반영됩니다.

### 4.3 Depth 카메라 (손목 장착)
D405 급: 수평 FOV 87°, 64×48 해상도, 측정 범위 7–70 cm, TCP 기준 (0, 7, −12) cm 위치에서 손가락 쪽으로 15° 기울임.
레이캐스팅으로 테이블·물체까지의 z-depth를 구하고, 스테레오 오차 모델 `σ = σ₀ + k·z²`,
무효 픽셀(dropout), 1 mm 양자화를 적용합니다. Isaac Sim 백엔드도 같은 내부 파라미터·장착 위치를 씁니다.

### 4.4 단순화 가정 (리포트에 명시 권장)
* 물체는 테이블 위에 놓인 강체 상자/원통, 파지 중에만 1차원(수직) 동역학을 적분
* 이동 중 경로(스윕) 충돌은 검사하지 않고 목표 자세에서만 검사 (스텝이 1–3 cm 로 작음)
* RG2의 실제 링크 구조는 평행 이동 손가락으로 근사, 로봇 자기충돌·관절 동역학은 생략 (simple)
* 이런 부분이 Isaac Sim(PhysX) 결과와 다를 수 있으며, 그 차이(sim-to-sim gap)도 좋은 리포트 주제입니다.

---

## 5. 알고리즘과 파라미터

### Q-learning (`agents/q_learning.py`, 설정 `qlearning`)
```
Q(s,a) ← Q(s,a) + α [ r + γ·max_a′ Q(s′,a′)·(1 − terminated) − Q(s,a) ]
```
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `episodes` | 4000 | 학습 에피소드 |
| `alpha` | 0.2 | 학습률. `alpha_decay: visit` 이면 α = 1/N(s,a)^ω (하한 `alpha_min`) |
| `gamma` | 0.99 | 할인율 |
| `epsilon_start/end/decay/decay_episodes` | 1.0 / 0.05 / exp / 2500 | ε-greedy 탐험 스케줄 (exp 또는 linear) |
| `q_init` | 0.0 | 초기 Q값 (양수 = 낙관적 초기화) |

### DQN (`agents/dqn.py`, 설정 `dqn`)
```
G = r_t + γ r_{t+1} + … + γ^{n−1} r_{t+n−1}
y = G + γⁿ (1 − terminated) Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))     (Double DQN)
loss = Huber(Q_online(s_t, a_t) − y)
```
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `observation` | depth | `depth`(CNN, 원시 이미지) / `features`(MLP) |
| `episodes` | 1500 | |
| `lr`, `batch_size` | 5e-4, 64 | Adam |
| `buffer_size`, `learning_starts` | 30000, 1000 | 경험 리플레이 (depth는 uint8로 저장) |
| `train_freq`, `gradient_steps` | 2, 1 | 몇 스텝마다 몇 번 업데이트 |
| `target_update`, `tau`, `target_update_interval` | soft, 0.01, 1000 | 타깃 네트워크 갱신 |
| `double_dqn`, `n_step`, `huber_loss`, `grad_clip` | true, 1, true, 10 | n_step>1 이면 n-step 리턴 |
| `epsilon_*` | 1.0 → 0.05, 800 에피소드 | |
| `hidden`, `conv_channels` | [256,128], [16,32,32] | 네트워크 구조 |

* 시간 제한으로 끝난 경우(truncated)는 종료 상태가 아니므로 두 알고리즘 모두 **부트스트랩을 유지**합니다.
* GPU가 있으면 `dqn.device: auto` 가 자동으로 사용합니다. CPU에서 depth-CNN은 느리므로 `train_freq` 를 2–4로.

---

## 6. 리포트를 위한 실험 가이드

### 6.1 기본 비교 (Q-learning vs DQN)
```bash
python scripts/compare.py --algos qlearning dqn_features dqn --seeds 0 1 2 --name main
```
`results/main/comparison.png` 에 (1) 에피소드별 성공률, (2) **환경 스텝 대비 성공률(표본 효율)**,
(3) greedy 평가 성공률, (4) 셰이핑 제외 과제 return, (5) 에피소드 길이, (6) 실패 원인 분포가 그려지고,
`summary.md` 에 최종 성능 ± 표준편차, 80% 도달 에피소드, 학습 시간, **파라미터 수(Q-table vs 신경망)** 가 정리됩니다.

비교 관점 예시
* **상태 표현**: Q-learning은 사람이 설계한 구간(bin) 상태만 볼 수 있다 → 구간이 거칠면 정밀 파지가 불가능,
  촘촘하면 상태 수가 폭발 (`state.bins` 를 바꿔 실험). DQN은 연속 특징이나 **원시 depth 이미지**를 바로 받는다.
* **일반화**: 표는 방문한 칸만 배운다(`summary.json` 의 `states_visited`). 신경망은 비슷한 상태끼리 일반화한다.
* **탐험**: 아래 6.3의 관찰처럼 표 기반 방법은 "안 가본 (s,a)는 0" 이라는 암묵적 낙관성으로 체계적으로 탐험하지만,
  신경망은 "잡기는 늘 −3" 을 모든 상태로 일반화해 버려 성공을 늦게 발견할 수 있다.
* **비용**: 스텝당 계산량, 메모리(표 18,531칸 vs 수만~수십만 가중치), 하이퍼파라미터 민감도.

### 6.2 파라미터 스윕 예시
```bash
python scripts/sweep.py --algo qlearning --param qlearning.alpha --values 0.05 0.2 0.5 --seeds 0 1
python scripts/sweep.py --algo qlearning --param qlearning.gamma --values 0.9 0.99
python scripts/sweep.py --algo qlearning --param qlearning.q_init --values 0 5
python scripts/sweep.py --algo qlearning --param reward.shaping.enabled --values true false
python scripts/sweep.py --algo dqn --param dqn.double_dqn --values true false --set dqn.observation=features
python scripts/sweep.py --algo dqn --param dqn.n_step --values 1 3 5 --set dqn.observation=features
python scripts/sweep.py --algo qlearning --param state.source --values perception ground_truth
python scripts/sweep.py --algo qlearning --param gripper.force --values 5 20 40 --set object.mass=1.0
python scripts/sweep.py --algo qlearning --param actions.step_xy --values 0.005 0.01 0.02
```
상태 구간을 바꿀 때: `--set "state.bins.dx=[-0.03,-0.01,0.01,0.03]"` (따옴표 필수).

### 6.3 꼭 짚어볼 관찰: 셰이핑의 γ 와 "멀리서 버티기" 함정
퍼텐셜 셰이핑 `F = γΦ(s′) − Φ(s)` 에서 Φ ≤ 0 이고 γ < 1 이면, **제자리에 머물기만 해도**
`F = −(1−γ)Φ > 0` 의 보상이 생깁니다. 물체에서 멀수록(Φ가 작을수록) 이 공짜 보상이 커서,
아직 성공을 한 번도 못 본 에이전트에게는 "물체에서 멀어져 버티다 시간 종료" 가 최선으로 보입니다.
`reward.shaping.gamma` 를 0.99 와 1.0 으로 바꿔 DQN을 학습해 보세요 (기본값은 1.0).
또 `terminal_potential: zero` (Ng et al. 1999의 이론적 형태) 는 정책 불변성을 보장하지만 종료 직전 상태의 −Φ 만큼
보너스가 생겨 "아무데서나 바로 잡기" 가 초반에 매력적으로 보입니다 — 이론과 실제 학습 속도의 차이를 보여주는 좋은 예입니다.

### 6.4 학습된 정책 확인
```bash
python scripts/visualize.py --algo qlearning --model results/main/qlearning_seed0/q_table.npz --episodes 3
python scripts/evaluate.py  --algo dqn --model results/main/dqn_seed0/dqn.pt --episodes 200
```
GIF에는 3D 로봇 자세, 상면도(물체·손가락·작업영역), 손목 depth 이미지와 추정 특징이 함께 나옵니다.

---

## 7. Isaac Sim 에서 실행하기

### 7.1 구성
* `grasp_rl/urdf.py` 가 **현재 설정으로 UR5 + RG2 + 카메라 URDF** 를 만듭니다
  (DH 표에서 바로 관절 원점을 만들어 내장 기구학과 정확히 일치 — `tests/test_core.py::test_urdf_matches_dh` 로 검증).
  링크 질량은 ur_description 값, 형상은 단순화된 원통/상자입니다.
* `backends/isaac_backend.py` 가 이 URDF를 PhysX 관절체로 불러오고, 물체(`DynamicCuboid`/`DynamicCylinder`)와
  마찰 재질, TCP 링크에 붙인 depth `Camera`(ROS 광학축, 같은 FOV) 를 만듭니다.
* 행동 → TCP 목표 → **같은 UR5 IK** → 관절 위치 목표 → PhysX 스텝. 손가락은 최대 힘을 RG2 파지력으로 제한한
  프리즘 관절 드라이브라서 **접촉·마찰·미끄러짐은 PhysX가 계산**합니다.
* MDP(State/Action/Reward), 에이전트, 로그 형식은 simple 백엔드와 완전히 동일합니다.

### 7.2 실행
```bash
# Isaac Sim 설치 폴더의 python 으로 의존성 설치 (한 번)
~/isaacsim/python.sh -m pip install gymnasium pyyaml matplotlib pillow

cd rl_grasp_sim
# (1) 내장 물리로 빠르게 학습한 정책을 Isaac Sim 에서 평가 (sim-to-sim 전이)
~/isaacsim/python.sh scripts/evaluate.py --config configs/isaac.yaml \
    --algo qlearning --model results/main/qlearning_seed0/q_table.npz --episodes 20
# (2) Isaac Sim 에서 직접 학습 (느림: 스텝마다 물리 수백 스텝 + 렌더링)
~/isaacsim/python.sh scripts/train.py --config configs/isaac.yaml --algo dqn \
    --set isaac.headless=true --set dqn.episodes=300
```
Windows 는 `python.bat`. URDF만 확인하려면 `python scripts/make_urdf.py` 후 Isaac Sim GUI의
*File → Import* 로 `assets/urdf/ur5_rg2.urdf` 를 불러오면 됩니다.

### 7.3 주의 (검증 범위)
* Isaac Sim 백엔드는 Isaac Sim 4.x/5.x 의 `isaacsim.*` API 와 2023.1 의 `omni.isaac.*` API를 모두 시도하도록
  작성했지만, **이 저장소의 개발 환경에는 GPU/Isaac Sim이 없어 실제 실행 검증은 하지 못했습니다.**
  내장 물리 백엔드, 기구학, URDF 일관성은 테스트로 검증되어 있습니다.
* 버전에 따라 모듈 이름·URDF 임포터 설정 이름이 다를 수 있습니다. 에러가 나면 `isaac_backend.py` 상단의
  `_modules()` / `_import_urdf()` 두 함수만 해당 버전에 맞게 고치면 됩니다.
* 실제 UR5/RG2 메시(USD)로 보이게 하려면 URDF의 visual 형상을 메시로 바꾸면 됩니다. 관절 원점은 그대로 두세요.

---

## 8. 참고문헌
* Watkins & Dayan (1992), *Q-learning*. Machine Learning 8.
* Mnih et al. (2015), *Human-level control through deep reinforcement learning*. Nature 518.
* van Hasselt, Guez & Silver (2016), *Deep Reinforcement Learning with Double Q-learning*. AAAI.
* Hessel et al. (2018), *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI.
* Ng, Harada & Russell (1999), *Policy invariance under reward transformations*. ICML.
* Kalashnikov et al. (2018), *QT-Opt: Scalable Deep RL for Vision-Based Robotic Manipulation*. CoRL.
* Sutton & Barto (2018), *Reinforcement Learning: An Introduction* (2nd ed.), ch. 6 (TD/Q-learning), ch. 7 (n-step).
* Universal Robots UR5 / OnRobot RG2 / Intel RealSense D405 데이터시트 (치수·스트로크·파지력·FOV)
