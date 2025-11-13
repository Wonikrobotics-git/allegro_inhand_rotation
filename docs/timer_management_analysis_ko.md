# deploy_ros2.py 타이머 관리 시스템 분석

## 개요

`deploy_ros2.py`의 타이머 관리 시스템은 실시간 로봇 제어에서 가장 핵심적인 부분입니다. 이 문서는 왜 Hz 제어가 성공적인 실행에 결정적이었는지, 그리고 타이머가 어떻게 구현되고 관리되는지를 상세히 설명합니다.

---

## 1. 타이머 기반 제어의 중요성

### 1.1 왜 정확한 Hz 제어가 중요한가?

로봇 제어에서 **일정한 제어 주기(Hz)**는 다음과 같은 이유로 필수적입니다:

1. **안정성**: 불규칙한 제어 주기는 로봇의 불안정한 움직임을 야기
2. **예측 가능성**: 신경망 모델은 일정한 시간 간격으로 학습되었으므로, 동일한 주기로 실행되어야 함
3. **동기화**: 관측(observation) 수집과 액션 실행이 정확히 동기화되어야 함
4. **안전성**: 너무 느린 제어는 반응 지연을, 너무 빠른 제어는 하드웨어 과부하를 초래

### 1.2 이 프로젝트의 제어 주기

```python
hz: float = 20.0  # 기본값: 20Hz (50ms 주기)
```

- **20Hz = 50ms마다 한 번씩 제어 명령 전송**
- 학습 시뮬레이션과 동일한 주기를 유지하여 sim-to-real gap 최소화

---

## 2. 타이머 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│                    HardwarePlayer                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │            ROS2 Timer (20Hz)                       │  │
│  │                    ↓                               │  │
│  │          _control_step() 콜백                      │  │
│  │                    ↓                               │  │
│  │  ┌─────────────────────────────────────────┐      │  │
│  │  │ 1. Normalize observations              │      │  │
│  │  │ 2. Neural network inference            │      │  │
│  │  │ 3. Update target positions             │      │  │
│  │  │ 4. Publish command to hardware         │      │  │
│  │  │ 5. Non-blocking observation update     │      │  │
│  │  └─────────────────────────────────────────┘      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │         AllegroHandIO (Background)                 │  │
│  │  ┌──────────────────────────────────────────┐     │  │
│  │  │  ROS2 Executor (별도 스레드)             │     │  │
│  │  │  - /joint_states 구독                    │     │  │
│  │  │  - /commands 퍼블리시                    │     │  │
│  │  │  - /position_gap 퍼블리시                │     │  │
│  │  └──────────────────────────────────────────┘     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 타이머 생성 코드 (deploy_ros2.py:245-248)

```python
# Timer 등록 (정확 주기)
period = 1.0 / self.hz  # 20Hz → 0.05초 = 50ms
self.timer = self.allegro.create_timer(period, self._control_step)
print(f"Deployment started (timer-based {self.hz:.1f} Hz). Ctrl+C to stop.")
```

**핵심 포인트:**
- `create_timer(period, callback)`: ROS2의 정밀 타이머 사용
- `period = 0.05초`: 정확히 50ms마다 콜백 실행
- `self._control_step`: 타이머가 주기적으로 호출하는 제어 함수

---

## 3. 제어 루프 상세 분석

### 3.1 `_control_step()` 함수 (deploy_ros2.py:159-206)

이 함수는 타이머에 의해 **정확히 20Hz로 호출**됩니다.

```python
@torch.inference_mode()
def _control_step(self):
    t0 = time.perf_counter()  # 시작 시간 측정
```

#### Step 1: 관측 정규화 (deploy_ros2.py:163-164)

```python
# 1) norm
obs_norm = self.running_mean_std(self.obs_buf)
```

- `self.obs_buf`: (1, 96) 크기 = [t-2의 관측(32) | t-1의 관측(32) | 현재 관측(32)]
- 정규화를 통해 신경망 입력 범위를 안정화

#### Step 2: 신경망 추론 (deploy_ros2.py:166-171)

```python
# 2) inference
input_dict = {
    "obs": obs_norm,
    "proprio_hist": self.sa_mean_std(self.proprio_hist_buf),
}
action = torch.clamp(self.model.act_inference(input_dict), -1.0, 1.0)
```

- **GPU 기반 추론** (device="cuda")
- `proprio_hist_buf`: (1, 30, 32) = 과거 30 스텝의 proprioception 이력
- 출력: [-1, 1] 범위로 클램핑된 16차원 액션

#### Step 3: 타겟 업데이트 (deploy_ros2.py:173-174)

```python
# 3) update target
self._pre_physics_step(action)
```

```python
def _pre_physics_step(self, action):
    target = self.prev_target + self.action_scale * action
    self.cur_target = torch.clamp(target, min=self.allegro_dof_lower, max=self.allegro_dof_upper)
    self.prev_target = self.cur_target
```

- **증분 제어 (Incremental Control)**: 이전 타겟에 작은 변화량을 더함
- `action_scale = 1.0 / 24.0`: 급격한 움직임 방지
- 관절 한계 내로 클램핑

#### Step 4: 명령 퍼블리시 (deploy_ros2.py:176-180)

```python
# 4) publish command (CPU로만 내릴 때 변환)
cmd = self.cur_target.detach().to("cpu").numpy()[0]
ros1 = _action_hora2allegro(cmd)      # HORA 순서 → ROS1 순서
ros2 = _reorder_imrt2timr(ros1)       # ROS1 순서 → ROS2 순서
self.allegro.command_joint_position(ros2)
```

**좌표계 변환:**
1. **HORA**: Index, Thumb, Middle, Ring
2. **ROS1**: Index, Middle, Ring, Thumb
3. **ROS2**: Thumb, Index, Middle, Ring

#### Step 5: 비블로킹 관측 업데이트 (deploy_ros2.py:182-194)

```python
# 5) non-blocking obs update (드랍 시 마지막 유효 관측 사용)
q_pos = self.allegro.poll_joint_position(wait=False, timeout=0.0)
if q_pos is not None:
    ros1_q = _reorder_timr2imrt(q_pos)
    hora_q = _obs_allegro2hora(ros1_q)
    obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)
    self._last_obs_q = obs_q
else:
    obs_q = self._last_obs_q  # 이전 관측 재사용
    self._skipped += 1

if obs_q is not None:
    self._post_physics_step(obs_q)
```

**핵심 설계 결정:**
- **비블로킹 폴링**: 타이머 콜백이 블로킹되지 않도록 함
- **Graceful Degradation**: 관측을 못 받으면 이전 값 재사용
- `_skipped` 카운터로 드롭 빈도 추적

#### Step 6: 성능 모니터링 (deploy_ros2.py:196-206)

```python
# 6) light jitter log
if self._last_step_t is None:
    self._last_step_t = t0
else:
    dt = t0 - self._last_step_t
    self._last_step_t = t0
    # 5초마다 한 번만 출력
    if int(time.time()) % 5 == 0:
        hz_est = 1.0 / max(dt, 1e-6)
        print(f"[timer] {hz_est:.2f} Hz, skipped={self._skipped}")
```

- **실제 실행 Hz 측정**: 타이머가 정확히 20Hz로 작동하는지 확인
- **드롭 횟수 추적**: 관측 수신 실패 빈도 모니터링

---

## 4. 관측 버퍼 관리

### 4.1 `_post_physics_step()` 함수 (deploy_ros2.py:135-157)

타이머 콜백 내에서 호출되며, 관측 히스토리를 업데이트합니다.

```python
def _post_physics_step(self, obses):
    # 1) 현재 관측 정규화
    cur_obs = self._unscale(
        obses.view(-1), self.allegro_dof_lower, self.allegro_dof_upper
    ).view(1, 16)

    # 2) obs_buf 롤링 (96 = 32*3)
    #    [0:64] <- [32:96],  [64:80] <- cur_obs,  [80:96] <- cur_target
    src64 = self.obs_buf[:, 32:96].clone()  # 겹침 방지
    self.obs_buf[:, 0:64] = src64
    self.obs_buf[:, 64:80] = cur_obs
    self.obs_buf[:, 80:96] = self.cur_target

    # 3) proprio_hist_buf 롤링 (T=30)
    src_hist = self.proprio_hist_buf[:, 1:, :].clone()
    self.proprio_hist_buf[:, 0:-1, :] = src_hist
    self.proprio_hist_buf[:, -1, :16] = cur_obs
    self.proprio_hist_buf[:, -1, 16:32] = self.cur_target
```

**버퍼 구조:**

```
obs_buf (96차원):
┌──────────┬──────────┬──────────┬──────────┐
│ t-2 obs  │ t-2 tgt  │ t-1 obs  │ t-1 tgt  │  (이전 64차원)
│  (16)    │  (16)    │  (16)    │  (16)    │
└──────────┴──────────┴──────────┴──────────┘
┌──────────┬──────────┐
│  t obs   │  t tgt   │  (새로운 32차원)
│  (16)    │  (16)    │
└──────────┴──────────┘

proprio_hist_buf (30, 32):
각 타임스텝마다 [16차원 obs | 16차원 target] 저장
→ 30 스텝의 이력 유지
```

---

## 5. 타이머 생명주기 관리

### 5.1 초기화 단계 (deploy_ros2.py:208-244)

```python
def deploy(self):
    run_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"🧠 Starting HardwarePlayer deployment at {run_start_time}...")

    # ROS2 I/O 시작(백그라운드 실행기)
    self.allegro = start_allegro_io(side='right')
```

**백그라운드 ROS2 Executor 시작:**
- 별도 스레드에서 ROS2 메시지 처리
- 타이머와 독립적으로 작동

#### 워밍업 단계 (deploy_ros2.py:216-222)

```python
# 워밍업(블로킹) — 하드웨어 settle
warmup = int(self.hz * 4)  # 20Hz * 4초 = 80회
for t in range(warmup):
    tprint(f"setup {t} / {warmup}")
    pose = _reorder_imrt2timr(np.array(self.init_pose, dtype=np.float64))
    self.allegro.command_joint_position(pose)
    time.sleep(1.0 / self.hz)  # 50ms 대기
```

**목적:**
- 하드웨어가 초기 자세로 안정화되도록 4초간 대기
- 타이머 시작 전 수동으로 20Hz 루프 실행

#### 첫 관측 수집 (deploy_ros2.py:224-234)

```python
# 첫 관측(블로킹 1회 — 초기화 안정)
q_pos = self.allegro.poll_joint_position(wait=True, timeout=5.0)
if q_pos is None:
    print("❌ failed to read joint state.")
    stop_allegro_io(self.allegro)
    return
```

- **블로킹 폴링**: 초기화 시에만 사용
- 타이머 시작 전 유효한 관측이 있는지 확인

#### 버퍼 초기화 (deploy_ros2.py:236-243)

```python
# buffers 초기화
cur_obs_buf = self._unscale(obs_q, self.allegro_dof_lower, self.allegro_dof_upper)[None]
self.prev_target = obs_q[None]
for i in range(3):
    self.obs_buf[:, i*32:i*32+16] = cur_obs_buf
    self.obs_buf[:, i*32+16:i*32+32] = self.prev_target
self.proprio_hist_buf[:, :, :16] = cur_obs_buf
self.proprio_hist_buf[:, :, 16:32] = self.prev_target
```

- 모든 타임스텝을 현재 관측으로 채움
- 신경망이 유효한 입력으로 시작하도록 보장

### 5.2 타이머 시작 (deploy_ros2.py:245-248)

```python
# Timer 등록 (정확 주기)
period = 1.0 / self.hz
self.timer = self.allegro.create_timer(period, self._control_step)
print(f"Deployment started (timer-based {self.hz:.1f} Hz). Ctrl+C to stop.")
```

**이 시점부터 자동 제어 시작:**
- ROS2 타이머가 백그라운드에서 50ms마다 `_control_step()` 호출
- 메인 스레드는 시그널 처리만 담당

### 5.3 메인 루프 (deploy_ros2.py:250-260)

```python
# 메인 스레드: 시그널 처리 + 유지
interrupted = False

def _sigint(_sig, _frm):
    nonlocal interrupted
    interrupted = True
signal.signal(signal.SIGINT, _sigint)

try:
    while not interrupted:
        time.sleep(0.2)  # 200ms마다 체크
```

- **메인 스레드는 idle 상태**: 타이머가 모든 제어 담당
- Ctrl+C 신호만 모니터링

### 5.4 정리 단계 (deploy_ros2.py:261-287)

```python
finally:
    try:
        if self.timer is not None:
            self.timer.cancel()  # 타이머 중지
    except Exception:
        pass
    try:
        self.allegro.go_safe()  # 안전 자세로 이동
    except Exception:
        pass
    stop_allegro_io(self.allegro)  # ROS2 I/O 종료
    print("🧠 Deployment stopped cleanly.")
```

**종료 순서:**
1. **타이머 취소**: 더 이상 제어 명령 전송 안 함
2. **안전 자세**: 로봇을 안전한 위치로 이동
3. **ROS2 종료**: 백그라운드 스레드 정리

---

## 6. 타이머의 핵심 장점

### 6.1 정확한 주기 보장

**이전 방식 (while 루프)의 문제점:**
```python
# BAD: 불안정한 주기
while True:
    start = time.time()
    control_step()
    elapsed = time.time() - start
    time.sleep(max(0, 1.0/hz - elapsed))  # 누적 오차 발생
```

**타이머 방식의 장점:**
```python
# GOOD: ROS2가 정확한 주기 보장
self.timer = self.allegro.create_timer(period, self._control_step)
```

- ROS2의 `create_timer()`는 **벽시계 시간 기준**으로 정확히 실행
- 콜백 실행 시간과 무관하게 다음 호출 시점 계산
- 누적 오차 없음

### 6.2 비동기 I/O와의 통합

```
┌───────────────────────────┐
│   Timer Thread (20Hz)     │  ← 제어 루프
│   → _control_step()       │
└───────────────────────────┘
            ↓ publish
┌───────────────────────────┐
│  ROS2 Executor Thread     │  ← 메시지 송수신
│  → /commands pub          │
│  → /joint_states sub      │
└───────────────────────────┘
```

- 타이머는 ROS2 Executor의 일부로 실행
- 메시지 송수신과 자연스럽게 동기화

### 6.3 논블로킹 설계

```python
# 타이머 콜백은 절대 블로킹되지 않음
q_pos = self.allegro.poll_joint_position(wait=False, timeout=0.0)
if q_pos is not None:
    # 새 관측 사용
else:
    # 이전 관측 재사용 (graceful degradation)
```

- 관측 수신이 지연되어도 타이머는 계속 실행
- 실시간성 유지

---

## 7. 성능 튜닝 가이드

### 7.1 Hz 선택 기준

| Hz  | 주기    | 용도                          | 권장 여부 |
|-----|---------|-------------------------------|-----------|
| 10  | 100ms   | 느린 움직임, 저사양 하드웨어  | ⚠️        |
| 20  | 50ms    | **표준 제어 주기 (기본값)**   | ✅        |
| 50  | 20ms    | 빠른 반응, 고사양 GPU 필요    | ⚠️        |
| 100 | 10ms    | 매우 빠른 제어 (오버헤드 큼)  | ❌        |

**20Hz를 선택한 이유:**
1. IsaacGym 시뮬레이션 학습 주기와 일치
2. GPU 추론 시간(~5ms) + 통신 지연(~10ms) 여유
3. 안정적인 실시간 성능

### 7.2 타이머 콜백 최적화

```python
@torch.inference_mode()  # 그래디언트 계산 비활성화
def _control_step(self):
    # ✅ GPU 연산만 수행 (빠름)
    obs_norm = self.running_mean_std(self.obs_buf)
    action = self.model.act_inference(input_dict)

    # ✅ 비블로킹 I/O
    q_pos = self.allegro.poll_joint_position(wait=False, timeout=0.0)

    # ❌ 절대 하지 말 것:
    # - 블로킹 I/O (wait=True)
    # - 무거운 CPU 연산
    # - 디스크 I/O
    # - 빈번한 print() (로깅은 5초마다 한 번)
```

### 7.3 모니터링 메트릭

```python
[timer] 20.15 Hz, skipped=3
```

**해석:**
- `20.15 Hz`: 실제 실행 주기 (20Hz에 매우 근접)
- `skipped=3`: 3회 관측 드롭 발생

**경고 신호:**
- Hz < 18: 타이머 콜백이 너무 느림 → GPU/네트워크 확인
- skipped > 100/분: 관측 수신 불안정 → ROS2 통신 확인

---

## 8. 트러블슈팅

### 8.1 타이머가 불규칙하게 실행됨

**증상:**
```
[timer] 15.3 Hz, skipped=0
[timer] 23.8 Hz, skipped=0
```

**원인:**
- CPU 부하가 높음
- GPU 추론이 50ms 이상 소요

**해결:**
```python
# device를 "cuda"로 설정했는지 확인
agent = HardwarePlayer(hz=20.0, device="cuda")

# GPU 메모리 확인
torch.cuda.empty_cache()
```

### 8.2 관측 드롭이 많음

**증상:**
```
[timer] 20.1 Hz, skipped=250
```

**원인:**
- `/joint_states` 토픽 발행 빈도 낮음
- ROS2 네트워크 지연

**해결:**
```bash
# joint_states 주파수 확인
ros2 topic hz /joint_states

# QoS 설정 확인
ros2 topic info /joint_states -v
```

### 8.3 타이머가 시작되지 않음

**증상:**
```
Deployment started (timer-based 20.0 Hz). Ctrl+C to stop.
[timer] 0.0 Hz, skipped=0  # 아무것도 출력 안 됨
```

**원인:**
- ROS2 Executor가 실행되지 않음

**해결:**
```python
# allegro_ros2_one.py의 _Runner 확인
def start(self):
    self.thread.start()  # 스레드가 시작되었는지 확인
```

---

## 9. 결론

### 9.1 타이머 관리의 핵심 원칙

1. **정확한 주기**: ROS2 Timer를 사용하여 정확히 20Hz 유지
2. **비블로킹 설계**: 콜백 내에서 절대 블로킹 호출 금지
3. **Graceful Degradation**: 관측 드롭 시 이전 값 재사용
4. **분리된 책임**: 제어(타이머)와 I/O(Executor)를 별도 스레드로 분리
5. **모니터링**: 실제 Hz와 드롭 횟수를 지속적으로 추적

### 9.2 성공의 핵심 요소

이 코드가 성공적으로 실행될 수 있었던 이유:

✅ **정밀한 타이머**: ROS2의 고정밀 타이머로 일정한 제어 주기 보장
✅ **비동기 아키텍처**: 제어와 I/O의 완벽한 분리
✅ **강건한 오류 처리**: 관측 드롭, 타이밍 지터에 대한 대응
✅ **sim-to-real 일관성**: 시뮬레이션 학습 주기(20Hz)와 동일한 실행 주기
✅ **최적화된 콜백**: GPU 추론 + 논블로킹 I/O로 50ms 내 완료 보장

### 9.3 핵심 코드 요약

```python
# 타이머 생성 (정확한 주기)
period = 1.0 / self.hz  # 20Hz → 50ms
self.timer = self.allegro.create_timer(period, self._control_step)

# 타이머 콜백 (비블로킹)
@torch.inference_mode()
def _control_step(self):
    # 1. Normalize
    obs_norm = self.running_mean_std(self.obs_buf)

    # 2. Inference (GPU)
    action = self.model.act_inference(input_dict)

    # 3. Update target
    self._pre_physics_step(action)

    # 4. Publish (non-blocking)
    self.allegro.command_joint_position(cmd)

    # 5. Poll observation (non-blocking)
    q_pos = self.allegro.poll_joint_position(wait=False, timeout=0.0)
    if q_pos is not None:
        self._post_physics_step(obs_q)
    else:
        self._post_physics_step(self._last_obs_q)  # fallback
```

---

## 부록: 관련 파일

- **deploy_ros2.py**: 메인 제어 로직 및 타이머 관리
- **allegro_ros2_one.py**: ROS2 I/O 및 백그라운드 Executor
- **models.py**: ActorCritic 신경망 모델
- **running_mean_std.py**: 관측 정규화

---

**작성일**: 2025-11-11
**작성자**: 프로젝트 문서화 팀
