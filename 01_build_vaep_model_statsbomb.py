"""
Build the StatsBomb training corpus (463 men's matches, excluding the 2022 World Cup) and train the XGBoost VAEP model.

Run from the repository root (see README.md for the run order and data locations).
"""

import json
from datetime import date

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
ROOT = Path("statsbomb/data")
OUT_DIR = Path("data/spadl_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = date(2018, 1, 1)  # VAR 도입 이후
QATAR_WC_COMPETITION_ID = 43   # competitions.json에서 확인한 실제 값으로 교체
QATAR_WC_SEASON_ID = 106      # competitions.json에서 확인한 실제 값으로 교체

# 표본 규모 대비 수준 격차가 크거나(예: 최상위 리그 대비 낮은 수준),
# 표본이 너무 적어 실질적 기여가 미미한 대회는 학습 코퍼스에서 제외한다.
EXCLUDE_COMPETITION_NAMES = {
    "Indian Super league",   # 표본 비중 과다(20%) + 최상위 리그 대비 수준 격차
    "Major League Soccer",   # 표본 과소(6경기)
    "Champions League",      # 표본 과소(1~2경기/시즌)
}


# ---------------------------------------------------------------------------
# 1단계: 대회 목록 확인
# ---------------------------------------------------------------------------
def load_competitions() -> pd.DataFrame:
    with open(ROOT / "competitions.json", encoding="utf-8") as f:
        competitions = json.load(f)
    return pd.DataFrame(competitions)


# ---------------------------------------------------------------------------
# 2단계: 남자부 + 카타르 월드컵 제외 + 수준/표본 불균형 대회 제외
# ---------------------------------------------------------------------------
def filter_competitions(df_comp: pd.DataFrame) -> pd.DataFrame:
    is_male = df_comp["competition_gender"] == "male"
    is_qatar_wc = (df_comp["competition_id"] == QATAR_WC_COMPETITION_ID) & (
        df_comp["season_id"] == QATAR_WC_SEASON_ID
    )
    is_excluded = df_comp["competition_name"].isin(EXCLUDE_COMPETITION_NAMES)
    return df_comp[is_male & ~is_qatar_wc & ~is_excluded].copy()


# ---------------------------------------------------------------------------
# 3단계: 대상 경기 ID 수집 (날짜 컷오프 포함)
# ---------------------------------------------------------------------------
def collect_match_ids(df_train_comps: pd.DataFrame) -> pd.DataFrame:
    """match_id, competition_id, season_id, competition_name, match_date 를
    함께 담은 데이터프레임을 반환한다 (나중에 대회 분포 확인용)."""
    rows = []
    for _, row in df_train_comps.iterrows():
        match_file = (
            ROOT / "matches" / str(row["competition_id"]) / f"{row['season_id']}.json"
        )
        if not match_file.exists():
            continue
        with open(match_file, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            match_date = date.fromisoformat(m["match_date"])
            if match_date >= CUTOFF_DATE:
                rows.append(
                    {
                        "match_id": m["match_id"],
                        "competition_id": row["competition_id"],
                        "season_id": row["season_id"],
                        "competition_name": row["competition_name"],
                        "season_name": row["season_name"],
                        "match_date": match_date,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 검증: 카타르 월드컵 경기가 최종 목록에 섞여 있지 않은지 확인
# ---------------------------------------------------------------------------
def verify_no_qatar_wc(df_matches: pd.DataFrame) -> None:
    leaked = df_matches[
        (df_matches["competition_id"] == QATAR_WC_COMPETITION_ID)
        & (df_matches["season_id"] == QATAR_WC_SEASON_ID)
    ]
    assert len(leaked) == 0, f"카타르 월드컵 경기 {len(leaked)}건이 섞여 있습니다!"
    print("검증 통과: 카타르 월드컵 경기 없음")


# ---------------------------------------------------------------------------
# 대회 분포 확인 (특정 대회 쏠림 여부 점검)
# ---------------------------------------------------------------------------
def show_competition_distribution(df_matches: pd.DataFrame) -> None:
    dist = (
        df_matches.groupby(["competition_name", "season_name"])
        .size()
        .sort_values(ascending=False)
    )
    print("\n=== 대회/시즌별 경기 수 분포 (상위 20) ===")
    print(dist.head(20).to_string())
    print(f"\n총 경기 수: {len(df_matches)}")


# ---------------------------------------------------------------------------
# 4단계: 이벤트 JSON 로드 + 평탄화 (경기별로 즉시 parquet 저장)
# ---------------------------------------------------------------------------
def load_and_save_events(match_ids: list[int]) -> None:
    missing = []
    for i, mid in enumerate(match_ids, 1):
        out_path = OUT_DIR / f"{mid}.parquet"
        if out_path.exists():
            continue  # 이미 처리된 경기는 건너뜀 (재실행 시 중복 작업 방지)

        event_path = ROOT / "events" / f"{mid}.json"
        if not event_path.exists():
            missing.append(mid)
            continue

        with open(event_path, encoding="utf-8") as f:
            events = json.load(f)
        df = pd.json_normalize(events, sep="_")
        df["match_id"] = mid
        df.to_parquet(out_path)

        if i % 50 == 0:
            print(f"  {i}/{len(match_ids)} 경기 처리 완료")

    if missing:
        print(f"\n경고: 이벤트 파일이 없는 경기 {len(missing)}건")
        print(missing[:10], "..." if len(missing) > 10 else "")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
3단계: 액션 타입 매핑 + 좌표 통일 + 공격 방향 정렬

입력: data/spadl_ready/*.parquet (2단계에서 저장한 원본 이벤트)
출력: data/actions_ready/*.parquet (VAEP 학습용 정제된 액션)
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/spadl_ready")
OUT_DIR = Path("data/actions_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PITCH_X_MAX = 120.0  # StatsBomb 표준 좌표계
PITCH_Y_MAX = 80.0

# ---------------------------------------------------------------------------
# StatsBomb type_name -> 단순 액션 타입 매핑
# 주의: StatsBomb 'Carry' = SPADL 'dribble'(볼 운반),
#       StatsBomb 'Dribble' = SPADL 'take_on'(상대 제치기 시도) — 이름이 반대로 헷갈리기 쉬움
# ---------------------------------------------------------------------------
ACTION_TYPE_MAP = {
    "Pass": "pass",              # pass_cross=True 인 경우는 이후 별도로 "cross"로 세분화
    "Carry": "dribble",
    "Dribble": "take_on",
    "Shot": "shot",
    "Clearance": "clearance",
    "Interception": "interception",
    "Duel": "tackle",            # 태클성 듀얼만 남기고 싶으면 duel_type_name으로 추가 필터 가능
    "50/50": "tackle",           # 루즈볼 경합 - Duel과 개념상 동일 취급
    "Foul Committed": "foul",
    "Miscontrol": "bad_touch",
    "Dispossessed": "bad_touch",
    "Goal Keeper": "keeper_action",  # 세이브/캐치/펀치 등 세부는 goalkeeper_type_name 참고
    "Block": "block",
    "Ball Recovery": "recovery",
    # 자책골: Shot 이벤트가 아니라 별도 타입으로 기록됨.
    # 4단계(VAEP 라벨링)에서 "득점 여부" 판정 시 이 타입도 반드시 포함시켜야 함.
    "Own Goal Against": "owngoal",
    "Own Goal For": "owngoal",
}
# 명시적으로 학습 대상에서 제외 (오프볼/메타성/판정성 이벤트)
EXCLUDE_TYPES = {
    "Pressure", "Ball Receipt*", "Half Start", "Half End",
    "Tactical Shift", "Starting XI", "Substitution", "Bad Behaviour",
    "Error", "Foul Won",
    "Camera On", "Camera off", "Dribbled Past", "Injury Stoppage",
    "Offside", "Player Off", "Player On", "Referee Ball-Drop",
    "Shield",  # 볼 쉴딩: VAEP 학습엔 제외하지만, 추후 압박 대응 분석 본론에서 재사용 검토
}


def _split_xy(arr, n=2):
    """numpy array 혹은 None을 (x, y)로 안전하게 분리. shot_end_location처럼
    3요소([x, y, z])인 경우도 앞 2개만 사용."""
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return np.nan, np.nan
    if isinstance(arr, np.ndarray) and arr.size >= n:
        return float(arr[0]), float(arr[1])
    return np.nan, np.nan


def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df["start_x"], df["start_y"] = zip(*df["location"].map(_split_xy))

    end_col_by_type = {
        "Pass": "pass_end_location",
        "Carry": "carry_end_location",
        "Shot": "shot_end_location",
        "Goal Keeper": "goalkeeper_end_location",
    }

    end_x = np.full(len(df), np.nan)
    end_y = np.full(len(df), np.nan)

    for sb_type, end_col in end_col_by_type.items():
        if end_col not in df.columns:
            continue
        mask = df["type_name"] == sb_type
        xs, ys = zip(*df.loc[mask, end_col].map(_split_xy)) if mask.any() else ([], [])
        end_x[mask.values] = xs
        end_y[mask.values] = ys

    df["end_x"], df["end_y"] = end_x, end_y

    # 종료 좌표가 없는 타입(Duel, Interception, Clearance, take_on 등)은
    # 발생 지점을 시작=종료 좌표로 간주 (그 자리에서 일어난 액션으로 취급)
    no_end_mask = df["end_x"].isna() & df["start_x"].notna()
    df.loc[no_end_mask, "end_x"] = df.loc[no_end_mask, "start_x"]
    df.loc[no_end_mask, "end_y"] = df.loc[no_end_mask, "start_y"]

    return df


def map_action_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["type_name"].isin(EXCLUDE_TYPES)].copy()
    df["action_type"] = df["type_name"].map(ACTION_TYPE_MAP)

    # Pass 중 크로스는 별도 타입으로 세분화
    if "pass_cross" in df.columns:
        is_cross = (df["action_type"] == "pass") & (df["pass_cross"] == True)  # noqa: E712
        df.loc[is_cross, "action_type"] = "cross"

    # 결과(outcome) 정보를 별도 컬럼으로 보존: 4단계(VAEP 득점 라벨링)에서
    # "이 액션이 골로 이어졌는가"를 판정하려면 반드시 필요하다.
    # - 슈팅: shot_outcome_name 그대로 사용 ("Goal"이면 득점)
    # - 자책골: Against/For를 구분해서 남겨야 함. 득점 판정에는 For만 써야 하므로
    #   (team_name이 득점 혜택 팀으로 정확히 찍히는 쪽), 두 값을 다르게 표기한다.
    df["result_name"] = None
    if "shot_outcome_name" in df.columns:
        shot_mask = df["type_name"] == "Shot"
        df.loc[shot_mask, "result_name"] = df.loc[shot_mask, "shot_outcome_name"]
    df.loc[df["type_name"] == "Own Goal For", "result_name"] = "OwnGoalFor"
    df.loc[df["type_name"] == "Own Goal Against", "result_name"] = "OwnGoalAgainst"

    # 액션 성공 여부(action_success): VAEP 피처의 핵심 신호.
    # pass/cross는 NaN(=값 없음)이 완료를 의미하는 StatsBomb 관례를 따름.
    # take_on/interception/tackle은 outcome 값이 "성공"류에 해당하는지로 판정.
    # (아래 SUCCESS_VALUES는 알려진 StatsBomb 스키마 기준 추정치이므로,
    #  실행 시 실제 값 분포를 출력해 눈으로 재검증한다.)
    # dribble(캐리)/clearance 등 실패 개념이 약한 타입은 항상 성공으로 간주.
    SUCCESS_VALUES = {
        "take_on": {"Complete"},
        "interception": {"Won", "Success In Play", "Success Out"},
        "tackle": {"Won", "Success In Play", "Success Out", "Aerial Won"},
    }
    OUTCOME_COL_BY_TYPE = {
        "take_on": "dribble_outcome_name",
        "interception": "interception_outcome_name",
        "tackle": "duel_outcome_name",
    }

    df["action_success"] = True  # 기본값: 실패 개념이 약한 타입은 항상 성공

    for atype in ["pass", "cross"]:
        col = "pass_outcome_name"
        if col in df.columns:
            mask = df["action_type"] == atype
            df.loc[mask, "action_success"] = df.loc[mask, col].isna()

    for atype, outcome_col in OUTCOME_COL_BY_TYPE.items():
        if outcome_col not in df.columns:
            continue
        mask = df["action_type"] == atype
        df.loc[mask, "action_success"] = df.loc[mask, outcome_col].isin(
            SUCCESS_VALUES[atype]
        )

    df = df[df["action_type"].notna()].copy()
    return df


def process_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = extract_coordinates(df)
    df = map_action_types(df)
    # 공격 방향 정렬 불필요: StatsBomb은 팀 관점(possession team perspective)으로
    # 좌표를 미리 정규화해서 제공하므로, 모든 팀의 이벤트가 이미 항상 x=120
    # 방향을 공격하는 것으로 기록되어 있다. (StatsBomb open-data GitHub issue #7,
    # 다수의 커뮤니티 자료로 확인) 별도 반전 로직을 적용하면 오히려 정상 데이터를
    # 훼손하므로 이 단계는 제거한다.
    keep_cols = [
        "match_id", "period", "minute", "second", "team_name", "player_name",
        "action_type", "result_name", "action_success",
        "start_x", "start_y", "end_x", "end_y",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


if __name__ == "__main__":
    files = sorted(IN_DIR.glob("*.parquet"))
    print(f"처리 대상 파일 수: {len(files)}")

    for i, f in enumerate(files, 1):
        out_path = OUT_DIR / f.name
        if out_path.exists():
            continue
        try:
            df_clean = process_file(f)
            df_clean.to_parquet(out_path)
        except Exception as e:
            print(f"실패: {f.name} - {e}")

        if i % 50 == 0:
            print(f"  {i}/{len(files)} 완료")

    print(f"\n완료: {OUT_DIR}")

    # 정합성 확인 1: 방향 정렬 후 슈팅 종료 x좌표 분포 확인 (120 근처에 몰려야 정상)
    sample = pd.read_parquet(OUT_DIR / files[0].name)
    shots = sample[sample["action_type"] == "shot"]
    if not shots.empty:
        print("\n=== 정렬 후 슈팅 end_x 분포 (120에 가까울수록 정상) ===")
        print(shots["end_x"].describe())

    # 정합성 확인 2: action_success 판정에 쓰인 outcome 값 분포를 직접 눈으로 검산.
    # SUCCESS_VALUES가 실제 데이터와 안 맞으면 여기서 바로 드러난다.
    print("\n=== action_success 판정 검산 (여러 경기 표본) ===")
    check_files = files[:30]
    all_checked = pd.concat([pd.read_parquet(OUT_DIR / f.name) for f in check_files])

    print("\n[전체 action_type별 action_success 비율]")
    print(all_checked.groupby("action_type")["action_success"].mean())

    # 원본(spadl_ready)에서 실제 outcome 값 종류를 다시 불러와 성공/실패 분류가
    # 상식과 맞는지 대조
    print("\n[interception_outcome_name 실제 값과 판정 결과 대조]")
    raw_check = []
    for f in check_files:
        raw = pd.read_parquet(IN_DIR / f.name)
        raw_check.append(raw[raw["type_name"] == "Interception"][["interception_outcome_name"]])
    raw_check_df = pd.concat(raw_check)
    print(raw_check_df["interception_outcome_name"].value_counts(dropna=False))

    print("\n[duel_outcome_name 실제 값과 판정 결과 대조 (Duel + 50/50)]")
    raw_check2 = []
    for f in check_files:
        raw = pd.read_parquet(IN_DIR / f.name)
        raw_check2.append(raw[raw["type_name"].isin(["Duel", "50/50"])][["duel_outcome_name"]])
    raw_check2_df = pd.concat(raw_check2)
    print(raw_check2_df["duel_outcome_name"].value_counts(dropna=False))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
4단계: VAEP 라벨 생성 (scores / concedes)

각 액션에 대해, 그 액션 시점부터 이후 nb_prev_actions(기본 10)개 액션 이내에
- 그 액션을 수행한 팀이 득점했으면 label_scores = True
- 상대팀이 득점했으면 label_concedes = True

입력: data/actions_ready/*.parquet (3단계 산출물, result_name 포함)
출력: data/labeled/*.parquet (label_scores, label_concedes 컬럼 추가)
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/actions_ready")
OUT_DIR = Path("data/labeled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NB_PREV_ACTIONS = 10  # VAEP 표준 lookahead 윈도우 크기


def _rolling_any_forward(bool_arr: np.ndarray, k: int) -> np.ndarray:
    """각 시점 i에서 [i, i+k-1] 구간에 True가 하나라도 있는지 (미래 방향 rolling).
    뒤집어서 rolling max를 적용한 뒤 다시 뒤집는 방식으로 벡터화."""
    reversed_arr = bool_arr[::-1].astype(float)
    rolled = pd.Series(reversed_arr).rolling(window=k, min_periods=1).max().values
    return rolled[::-1].astype(bool)


def compute_labels(df: pd.DataFrame, k: int = NB_PREV_ACTIONS) -> pd.DataFrame:
    """득점/실점 라벨을 생성한다.

    경계 처리:
    - 경기 단위: 이 함수가 경기별 파일 하나씩 호출되므로 자동으로 보장됨.
    - 전/후반(피리어드) 단위: 롤링 윈도우가 피리어드 경계를 넘지 않도록
      period별로 나누어 각각 계산한다. 하프타임처럼 실제로 플레이가
      끊기는 지점을 넘어 득점 기여를 인정하는 건 부적절하기 때문.
    - 포제션(시퀀스) 단위: 의도적으로 끊지 않는다. VAEP의 핵심 취지가
      턴오버를 거쳐도 득점으로 이어지는 흐름 전체에 가치를 나눠주는 것이므로,
      포제션이 바뀌었다고 윈도우를 끊으면 VAEP 설계 의도와 어긋난다.
    """
    df = df.reset_index(drop=True)
    df["label_scores"] = False
    df["label_concedes"] = False
    df["_label_warning"] = ""

    teams = df["team_name"].dropna().unique()
    if len(teams) != 2:
        df["label_scores"] = np.nan
        df["label_concedes"] = np.nan
        df["_label_warning"] = f"team count = {len(teams)}"
        return df

    team_a, team_b = teams[0], teams[1]
    is_goal_all = (
        (df["action_type"] == "shot") & (df["result_name"] == "Goal")
    ) | (df["result_name"] == "OwnGoalFor")

    # 피리어드 경계를 넘지 않도록 구간별로 끊어서 롤링 계산.
    # sort=False로 원본(=시간순) 행 순서를 그대로 유지한다.
    for period, idx in df.groupby("period", sort=False).groups.items():
        idx = idx.sort_values()  # 그룹 내에서도 원본 행 순서(=시간순) 보장
        sub = df.loc[idx]
        is_goal = is_goal_all.loc[idx].values
        goal_team = np.where(is_goal, sub["team_name"].values, "")

        goal_by_a = _rolling_any_forward(goal_team == team_a, k)
        goal_by_b = _rolling_any_forward(goal_team == team_b, k)
        is_team_a_action = sub["team_name"].values == team_a

        df.loc[idx, "label_scores"] = np.where(is_team_a_action, goal_by_a, goal_by_b)
        df.loc[idx, "label_concedes"] = np.where(is_team_a_action, goal_by_b, goal_by_a)

    return df


if __name__ == "__main__":
    files = sorted(IN_DIR.glob("*.parquet"))
    print(f"처리 대상 파일 수: {len(files)}")

    warned = []
    for i, f in enumerate(files, 1):
        out_path = OUT_DIR / f.name
        if out_path.exists():
            continue

        df = pd.read_parquet(f)
        df_labeled = compute_labels(df)

        if (df_labeled["_label_warning"] != "").any():
            warned.append(f.name)

        df_labeled.drop(columns=["_label_warning"]).to_parquet(out_path)

        if i % 50 == 0:
            print(f"  {i}/{len(files)} 완료")

    print(f"\n완료: {OUT_DIR}")
    if warned:
        print(f"경고: 팀 수 이상으로 라벨링 스킵된 파일 {len(warned)}개")
        print(warned[:10])

    # 정합성 확인: 전체 라벨 긍정 비율과 실제 득점 이벤트 수 비교
    sample_files = files[:30]
    all_labeled = pd.concat([pd.read_parquet(OUT_DIR / f.name) for f in sample_files])
    print(f"\n=== 샘플 {len(sample_files)}개 경기 라벨 분포 ===")
    print(f"전체 액션 수: {len(all_labeled)}")
    print(f"label_scores=True 비율: {all_labeled['label_scores'].mean():.4f}")
    print(f"label_concedes=True 비율: {all_labeled['label_concedes'].mean():.4f}")

    n_actual_goals = (
        (all_labeled["action_type"] == "shot") & (all_labeled["result_name"] == "Goal")
    ).sum() + (all_labeled["result_name"] == "OwnGoalFor").sum()
    print(f"실제 득점 이벤트 수(샘플 내): {n_actual_goals}")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
5단계: VAEP 피처 생성

입력: data/labeled/*.parquet (4단계 산출물)
출력: data/features/X.parquet, data/features/y.parquet

피처 구성 (VAEP 원 논문 방식: 현재 + 직전 2개 액션 컨텍스트):
- a0(현재), a1(직전), a2(그 직전) 각각의: action_type 원-핫, action_success,
  시작/종료 좌표, 골대까지 거리·각도
- 게임 상황: 스코어 차이(액션 시점 기준), 경과 시간(초)

경계 처리:
- 직전 액션 컨텍스트(a1, a2): 피리어드 경계를 넘지 않도록 period별로 shift.
- 스코어 차이: 피리어드 경계를 넘어서 누적(하프타임에도 스코어는 유지되므로).
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/labeled")
OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOAL_X, GOAL_Y = 120.0, 40.0  # StatsBomb 팀 관점 좌표계에서 상대 골대는 항상 이 위치
N_PREV_ACTIONS = 2  # 직전 몇 개 액션까지 컨텍스트로 볼지 (현재 액션 포함 총 3개)

ACTION_TYPES = [
    "pass", "cross", "dribble", "take_on", "shot", "tackle",
    "interception", "clearance", "foul", "bad_touch",
    "keeper_action", "block", "recovery", "owngoal",
]


def dist_angle_to_goal(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    dx = GOAL_X - x
    dy = GOAL_Y - y
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy.abs(), dx)
    return dist, angle


def add_location_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    sx, sy = df[f"{prefix}start_x"], df[f"{prefix}start_y"]
    ex, ey = df[f"{prefix}end_x"], df[f"{prefix}end_y"]
    df[f"{prefix}start_dist_goal"], df[f"{prefix}start_angle_goal"] = dist_angle_to_goal(sx, sy)
    df[f"{prefix}end_dist_goal"], df[f"{prefix}end_angle_goal"] = dist_angle_to_goal(ex, ey)
    # VAEP 원 논문(socceraction features.py의 movement())과 동일하게,
    # 거리·각도는 좌우 대칭으로 접되(위 두 줄), 이동 벡터의 부호(dx, dy)는
    # 별도로 보존해 "세로 방향 전환" 같은 비대칭 정보를 완전히 잃지 않도록 함.
    df[f"{prefix}move_dx"] = ex - sx
    df[f"{prefix}move_dy"] = ey - sy
    df[f"{prefix}move_dist"] = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    return df


def add_previous_action_context(df: pd.DataFrame, n_prev: int = N_PREV_ACTIONS) -> pd.DataFrame:
    context_cols = [
        "action_type", "action_success", "start_x", "start_y", "end_x", "end_y",
    ]
    for i in range(1, n_prev + 1):
        # period별로 shift하여 피리어드 경계를 넘지 않도록 함
        shifted = df.groupby("period", sort=False)[context_cols].shift(i)
        shifted.columns = [f"a{i}_{c}" for c in context_cols]
        df = pd.concat([df, shifted], axis=1)
    return df


def add_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    """액션 시점 '이전'까지 누적된 스코어 차이(본인 팀 - 상대 팀).
    피리어드 경계와 무관하게 경기 전체에 걸쳐 누적."""
    teams = df["team_name"].dropna().unique()
    if len(teams) != 2:
        df["score_diff"] = np.nan
        return df
    team_a, team_b = teams[0], teams[1]

    is_goal = (
        (df["action_type"] == "shot") & (df["result_name"] == "Goal")
    ) | (df["result_name"] == "OwnGoalFor")

    # 현재 액션 '이전까지'의 누적 득점이어야 하므로 shift(1) 후 누적합
    goal_a = ((df["team_name"] == team_a) & is_goal).astype(int)
    goal_b = ((df["team_name"] == team_b) & is_goal).astype(int)
    cum_a = goal_a.cumsum().shift(1).fillna(0)
    cum_b = goal_b.cumsum().shift(1).fillna(0)

    is_team_a = df["team_name"] == team_a
    df["score_diff"] = np.where(is_team_a, cum_a - cum_b, cum_b - cum_a)
    return df


def build_features_for_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df = add_score_diff(df)
    df = add_previous_action_context(df)

    # 좌표 기반 피처: 현재(a0) + 직전(a1, a2) 각각
    df = add_location_features(df, prefix="")  # 현재 액션은 접두어 없음(start_x 등 원본 컬럼 사용)
    for i in range(1, N_PREV_ACTIONS + 1):
        df = add_location_features(df, prefix=f"a{i}_")

    # 시간 피처
    df["time_seconds"] = df["minute"] * 60 + df["second"]

    return df


def build_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """action_type 원-핫: 현재(a0) + 직전(a1, a2). 학습 시 항상 동일한 컬럼
    구조를 갖도록 categories를 고정해서 인코딩한다."""
    cat_type = pd.CategoricalDtype(categories=ACTION_TYPES)

    onehots = []
    a0 = pd.get_dummies(df["action_type"].astype(cat_type), prefix="a0_type")
    onehots.append(a0)
    for i in range(1, N_PREV_ACTIONS + 1):
        col = f"a{i}_action_type"
        oh = pd.get_dummies(df[col].astype(cat_type), prefix=f"a{i}_type")
        onehots.append(oh)

    return pd.concat(onehots, axis=1)


if __name__ == "__main__":
    files = sorted(IN_DIR.glob("*.parquet"))
    print(f"처리 대상 파일 수: {len(files)}")

    X_parts, y_parts = [], []
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        df = build_features_for_match(df)
        onehot = build_onehot(df)

        numeric_cols = [
            "start_dist_goal", "start_angle_goal", "end_dist_goal", "end_angle_goal",
            "move_dx", "move_dy", "move_dist",
            "a1_action_success", "a1_start_dist_goal", "a1_end_dist_goal",
            "a1_move_dx", "a1_move_dy", "a1_move_dist",
            "a2_action_success", "a2_start_dist_goal", "a2_end_dist_goal",
            "a2_move_dx", "a2_move_dy", "a2_move_dist",
            "score_diff", "time_seconds",
        ]
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        X = pd.concat([df[["match_id"]], onehot, df[numeric_cols]], axis=1)
        y = df[["match_id", "label_scores", "label_concedes"]]

        X_parts.append(X)
        y_parts.append(y)

        if i % 50 == 0:
            print(f"  {i}/{len(files)} 완료")

    X_full = pd.concat(X_parts, ignore_index=True)
    y_full = pd.concat(y_parts, ignore_index=True)

    print(f"\n최종 X shape: {X_full.shape}")
    print(f"최종 y shape: {y_full.shape}")
    print(f"결측치가 있는 컬럼: {X_full.columns[X_full.isna().any()].tolist()}")

    X_full.to_parquet(OUT_DIR / "X.parquet")
    y_full.to_parquet(OUT_DIR / "y.parquet")
    print(f"\n완료: {OUT_DIR}")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
6단계: VAEP 가치 모델 학습 (XGBoost)

- label_scores, label_concedes 각각에 대해 별도의 XGBoost 이진분류기 학습
  (VAEP 원 논문/socceraction 레퍼런스 구현의 기본 러너와 동일하게 맞춤)
- 경기(match_id) 단위 GroupKFold로 분할 (동일 경기 액션이 학습/검증에 동시에 섞이지 않도록)
- 평가지표: Brier score, log loss (단순 정확도는 극단적 클래스 불균형 때문에 부적합)
- 최종적으로 전체 데이터로 재학습한 모델을 저장 (PFF FC 추론에 사용할 모델)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

FEATURES_DIR = Path("data/features")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
N_ESTIMATORS = 200


class TqdmCallback(xgb.callback.TrainingCallback):
    """XGBoost 학습 라운드마다 tqdm 막대그래프를 갱신하는 콜백."""

    def __init__(self, total: int, desc: str = ""):
        self.pbar = tqdm(total=total, desc=desc, unit="round")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # False = 학습 계속 (True를 반환하면 조기 종료됨)

    def after_training(self, model):
        self.pbar.close()
        return model


def load_data():
    X = pd.read_parquet(FEATURES_DIR / "X.parquet")
    y = pd.read_parquet(FEATURES_DIR / "y.parquet")
    groups = X["match_id"]
    X_feat = X.drop(columns=["match_id"])

    # a1_/a2_ action_success는 shift로 인해 NaN이 섞여 bool -> object로 승격된 상태.
    # int 변환은 NaN을 처리 못 하므로 float으로 변환(True->1.0, False->0.0, NaN은 유지).
    success_cols = [c for c in X_feat.columns if c.endswith("action_success")]
    X_feat[success_cols] = X_feat[success_cols].astype(float)

    # 나머지 원-핫 인코딩 등 순수 bool 컬럼(NaN 없음)은 int로 변환
    bool_cols = X_feat.select_dtypes(include="bool").columns
    X_feat[bool_cols] = X_feat[bool_cols].astype(int)

    return X_feat, y, groups


def make_model(callbacks=None):
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",  # 대용량 데이터 학습 속도를 위한 히스토그램 기반 분할
        random_state=42,
        eval_metric="logloss",
        callbacks=callbacks,  # 최신 XGBoost sklearn API: callbacks는 fit()이 아니라 생성자 인자
    )


def evaluate_groupkfold(X: pd.DataFrame, y: pd.Series, groups: pd.Series, label_name: str):
    gkf = GroupKFold(n_splits=N_FOLDS)
    brier_scores, log_losses = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = make_model(
            callbacks=[TqdmCallback(N_ESTIMATORS, desc=f"{label_name} fold {fold}/{N_FOLDS}")],
        )
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]

        brier = brier_score_loss(y_val, pred)
        ll = log_loss(y_val, pred, labels=[0, 1])
        brier_scores.append(brier)
        log_losses.append(ll)
        print(f"  [{label_name}] fold {fold}: Brier={brier:.5f}, LogLoss={ll:.5f}")

    print(f"[{label_name}] 평균 Brier={np.mean(brier_scores):.5f} (+-{np.std(brier_scores):.5f})")
    print(f"[{label_name}] 평균 LogLoss={np.mean(log_losses):.5f} (+-{np.std(log_losses):.5f})")
    return brier_scores, log_losses


def train_final_model(X: pd.DataFrame, y: pd.Series, label_name: str):
    """전체 데이터로 재학습한 최종 모델 (PFF FC 추론용).

    XGBoost 자체 저장 포맷(save_model)을 사용한다. joblib/pickle은 모델에
    딸려있는 콜백 객체(tqdm 등 파이썬 전용 객체)까지 직렬화하려다 실패할 수 있는데,
    save_model은 학습된 트리 구조만 저장하므로 이런 문제가 없다.
    """
    model = make_model(
        callbacks=[TqdmCallback(N_ESTIMATORS, desc=f"{label_name} 최종 학습")],
    )
    model.fit(X, y)

    save_path = MODEL_DIR / f"vaep_{label_name}.json"
    model.save_model(save_path)
    print(f"저장 완료: {save_path}")
    return model


if __name__ == "__main__":
    print("데이터 로딩...")
    X, y, groups = load_data()
    print(f"X shape: {X.shape}, 경기 수: {groups.nunique()}")

    # 결측치는 XGBoost가 네이티브로 처리하므로 별도 대체 불필요
    for label_col in ["label_scores", "label_concedes"]:
        print(f"\n{'=' * 50}")
        print(f"{label_col} 모델 평가 (GroupKFold, {N_FOLDS}-fold)")
        print("=" * 50)
        y_target = y[label_col].astype(int)
        evaluate_groupkfold(X, y_target, groups, label_col)

        print(f"\n{label_col}: 전체 데이터로 최종 모델 학습 중...")
        train_final_model(X, y_target, label_col.replace("label_", ""))

    print("\n모든 모델 학습 완료.")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
7단계 준비: VAEP 값 계산

- 저장된 scores/concedes 모델로 모든 액션의 확률 예측
- 같은 (match_id, period) 그룹 내에서, 연속된 두 행의 확률 차이로 VAEP 값 산출
- 피리어드 첫 액션은 비교 대상(직전 상태)이 없어 값 없음(NaN)으로 남김
- 식별 컬럼(period, team_name, player_name)은 data/labeled에서 같은 순서로 재결합
  (모델 재학습 없이 이미 저장된 X.parquet/모델을 그대로 활용)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES_DIR = Path("data/features")
LABELED_DIR = Path("data/labeled")
MODEL_DIR = Path("models")
OUT_DIR = Path("data/vaep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    m_scores = xgb.XGBClassifier()
    m_scores.load_model(MODEL_DIR / "vaep_scores.json")
    m_concedes = xgb.XGBClassifier()
    m_concedes.load_model(MODEL_DIR / "vaep_concedes.json")
    return m_scores, m_concedes


def reconstruct_identifiers() -> pd.DataFrame:
    """11단계와 정확히 동일한 파일 순서로 data/labeled를 읽어,
    X.parquet의 각 행과 1:1 대응하는 식별 컬럼(period, team_name, player_name,
    action_type, match_id)을 재구성한다."""
    files = sorted(LABELED_DIR.glob("*.parquet"))  # 11단계와 동일한 정렬 기준
    id_cols = ["match_id", "period", "team_name", "player_name", "action_type"]
    parts = [pd.read_parquet(f, columns=id_cols) for f in files]
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    print("데이터 로딩...")
    X = pd.read_parquet(FEATURES_DIR / "X.parquet")
    ids = reconstruct_identifiers()

    # 정합성 확인: 길이와 match_id가 완전히 일치하는지 반드시 검증
    assert len(X) == len(ids), f"행 수 불일치: X={len(X)}, ids={len(ids)}"
    mismatch = (X["match_id"].values != ids["match_id"].values).sum()
    assert mismatch == 0, f"match_id 정렬이 어긋난 행이 {mismatch}개 있습니다"
    print(f"정합성 확인 통과: {len(X)}행, match_id 완전 일치")

    X_feat = X.drop(columns=["match_id"])
    bool_cols = X_feat.select_dtypes(include="bool").columns
    X_feat[bool_cols] = X_feat[bool_cols].astype(int)
    success_cols = [c for c in X_feat.columns if c.endswith("action_success")]
    X_feat[success_cols] = X_feat[success_cols].astype(float)

    print("모델 로딩 및 예측...")
    m_scores, m_concedes = load_models()
    p_scores = m_scores.predict_proba(X_feat)[:, 1]
    p_concedes = m_concedes.predict_proba(X_feat)[:, 1]

    result = ids.copy()
    result["p_scores"] = p_scores
    result["p_concedes"] = p_concedes

    print("VAEP 값 계산 (피리어드 경계 내에서 연속 상태 차이)...")
    result["offensive_value"] = np.nan
    result["defensive_value"] = np.nan

    for (match_id, period), idx in result.groupby(["match_id", "period"], sort=False).groups.items():
        idx = idx.sort_values()
        p_s = result.loc[idx, "p_scores"].values
        p_c = result.loc[idx, "p_concedes"].values

        off_val = np.full(len(idx), np.nan)
        def_val = np.full(len(idx), np.nan)
        if len(idx) > 1:
            off_val[1:] = p_s[1:] - p_s[:-1]
            def_val[1:] = p_c[:-1] - p_c[1:]

        result.loc[idx, "offensive_value"] = off_val
        result.loc[idx, "defensive_value"] = def_val

    result["vaep_value"] = result["offensive_value"] + result["defensive_value"]

    result.to_parquet(OUT_DIR / "vaep_values.parquet")
    print(f"\n완료: {OUT_DIR / 'vaep_values.parquet'}")

    # face validity 체크: 선수별 VAEP 합산 상위 15명
    print("\n=== 선수별 VAEP 합산 상위 15명 (face validity 체크용) ===")
    top_players = (
        result.dropna(subset=["vaep_value"])
        .groupby("player_name")["vaep_value"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    print(top_players)
