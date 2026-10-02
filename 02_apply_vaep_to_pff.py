"""
Convert PFF FC 2022 World Cup events to the common action schema, align coordinates and attacking direction, and apply the trained VAEP model without retraining.

Run from the repository root (see README.md for the run order and data locations).
"""

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
PFF FC 카타르 월드컵 파이프라인 통합본 (23+28+30+33번 파일을 하나로 합침)

앞으로는 이 파일 하나만 실행하시면 됩니다 - 버전이 헷갈릴 일이 없도록
전체 흐름을 한 스크립트에 담았습니다.

실행 순서:
  A. 이벤트 JSON -> 액션 매핑 + 좌표(StatsBomb 격자 변환) + 승부차기 제외
     -> data/pff_actions_ready/*.parquet
  B. 피처 생성(StatsBomb 학습 모델과 동일한 63컬럼 구조) + 등급/압박/시간 보존
     -> data/pff_features/X_pff.parquet, ids_pff.parquet
  C. 저장된 VAEP 모델(vaep_scores.json, vaep_concedes.json)로 추론
     -> data/pff_vaep/pff_vaep_values.parquet
  D. 직전 행위 대비 시간차(time_gap) 계산
     -> data/pff_vaep/pff_vaep_with_timegap.parquet  (최종 산출물)

중요: eventTime 정렬을 A단계 맨 끝에서 한 번에 처리하므로, 이후 모든 단계는
"행 순서 = 시간 순서"라는 전제를 안전하게 쓸 수 있다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from mplsoccer import Standardizer

PFF_DIR = Path("FIFA World Cup 2022/Event Data")
ACTIONS_DIR = Path("data/pff_actions_ready")
FEATURES_DIR = Path("data/pff_features")
VAEP_DIR = Path("data/pff_vaep")
MODEL_DIR = Path("models")

for d in [ACTIONS_DIR, FEATURES_DIR, VAEP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# A. 액션 매핑 + 좌표 처리
# =============================================================================

CONFIRMED_OWN_GOALS = [
    {"file": "10503.json", "conceding_team": "Argentina", "benefiting_team": "Australia"},
    {"file": "3853.json", "conceding_team": "Morocco", "benefiting_team": "Canada"},
]
OWN_GOAL_FILES = {og["file"]: og for og in CONFIRMED_OWN_GOALS}

TACKLE_TYPES = {"B", "H", "L", "S", "T"}
TAKE_ON_TYPES = {"D"}
DUEL_TYPES = {"A", "FIFTY"}

EXCLUDE_GAME_EVENT_TYPES = {"OUT", "SUB", "END", "OFF", "OTB", "ON", "FIRSTKICKOFF",
                             "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "G"}

SIMPLE_TYPE_MAP = {
    "PA": "pass", "CR": "cross", "SH": "shot",
    "CL": "clearance", "BC": "dribble", "FO": "foul",
}


def extract_ball_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    def _get_xy(ball_field):
        if isinstance(ball_field, list) and len(ball_field) > 0:
            return ball_field[0].get("x"), ball_field[0].get("y")
        return None, None

    df["start_x"], df["start_y"] = zip(*df["ball"].map(_get_xy))

    period_col = "gameEvents_period"
    df["end_x"] = float("nan")
    df["end_y"] = float("nan")
    if period_col in df.columns:
        for period, idx in df.groupby(period_col, sort=False).groups.items():
            idx = idx.sort_values()
            df.loc[idx, "end_x"] = df.loc[idx, "start_x"].shift(-1)
            df.loc[idx, "end_y"] = df.loc[idx, "start_y"].shift(-1)
    else:
        df["end_x"] = df["start_x"].shift(-1)
        df["end_y"] = df["start_y"].shift(-1)
    return df


def normalize_attacking_direction(df: pd.DataFrame) -> pd.DataFrame:
    dir_col = "stadiumMetadata_teamAttackingDirection"
    if dir_col not in df.columns:
        return df
    flip_mask = df[dir_col] == "L"
    for col in ["start_x", "end_x"]:
        df.loc[flip_mask, col] = -df.loc[flip_mask, col]
    for col in ["start_y", "end_y"]:
        df.loc[flip_mask, col] = -df.loc[flip_mask, col]
    return df


def convert_to_statsbomb_grid(df: pd.DataFrame) -> pd.DataFrame:
    pitch_len_col = "stadiumMetadata_pitchLength"
    pitch_wid_col = "stadiumMetadata_pitchWidth"
    length = df[pitch_len_col].dropna().iloc[0] if pitch_len_col in df.columns and df[pitch_len_col].notna().any() else 105.0
    width = df[pitch_wid_col].dropna().iloc[0] if pitch_wid_col in df.columns and df[pitch_wid_col].notna().any() else 68.0

    standardizer = Standardizer(pitch_from="custom", length_from=length, width_from=width, pitch_to="statsbomb")

    x_col, y_col, ex_col, ey_col = "start_x", "start_y", "end_x", "end_y"
    x_corner = df[x_col] + length / 2
    y_corner = df[y_col] + width / 2
    df[x_col], df[y_col] = standardizer.transform(x_corner.values, y_corner.values)
    ex_corner = df[ex_col] + length / 2
    ey_corner = df[ey_col] + width / 2
    df[ex_col], df[ey_col] = standardizer.transform(ex_corner.values, ey_corner.values)
    return df


def detect_and_exclude_shootout(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    shot_mask = df["possessionEvents_possessionEventType"] == "SH"
    if shot_mask.sum() == 0:
        return df
    shots = df[shot_mask]
    coord_counts = shots.groupby(["start_x", "start_y"]).size()
    shootout_coords = set(coord_counts[coord_counts >= 3].index)
    if len(shootout_coords) == 0:
        return df
    is_shootout_coord = shots.apply(lambda r: (r["start_x"], r["start_y"]) in shootout_coords, axis=1)
    shootout_shot_idx = shots.index[is_shootout_coord]
    cutoff_idx = shootout_shot_idx.min()
    n_before = len(df)
    df = df[df.index < cutoff_idx]
    print(f"  -> 승부차기 탐지: {filename}에서 {cutoff_idx}번째 행부터 끝까지 "
          f"{n_before - len(df)}건 제외")
    return df


def expand_challenge_events(df: pd.DataFrame) -> pd.DataFrame:
    ch = df[df["possessionEvents_possessionEventType"] == "CH"].copy()
    ct = ch["possessionEvents_challengeType"]
    rows = []

    tackle_mask = ct.isin(TACKLE_TYPES)
    tackle_rows = ch[tackle_mask].copy()
    tackle_rows["action_type"] = "tackle"
    tackle_rows["actor_player_id"] = tackle_rows["possessionEvents_challengerPlayerId"]
    tackle_rows["opponent_player_id"] = tackle_rows["possessionEvents_carrierPlayerId"]
    tackle_rows["action_success"] = (
        tackle_rows["actor_player_id"] == tackle_rows["possessionEvents_challengeWinnerPlayerId"]
    ).astype(float)
    tackle_rows.loc[tackle_rows["possessionEvents_challengeWinnerPlayerId"].isna(), "action_success"] = float("nan")
    rows.append(tackle_rows)

    take_on_mask = ct.isin(TAKE_ON_TYPES)
    take_on_rows = ch[take_on_mask].copy()
    take_on_rows["action_type"] = "take_on"
    take_on_rows["actor_player_id"] = take_on_rows["possessionEvents_dribblerPlayerId"]
    take_on_rows["opponent_player_id"] = take_on_rows["possessionEvents_challengerPlayerId"]
    take_on_rows["action_success"] = (
        take_on_rows["actor_player_id"] == take_on_rows["possessionEvents_challengeWinnerPlayerId"]
    ).astype(float)
    take_on_rows.loc[take_on_rows["possessionEvents_challengeWinnerPlayerId"].isna(), "action_success"] = float("nan")
    rows.append(take_on_rows)

    duel_mask = ct.isin(DUEL_TYPES)
    duel_base = ch[duel_mask]
    for side_col in ["possessionEvents_homeDuelPlayerId", "possessionEvents_awayDuelPlayerId"]:
        side_rows = duel_base.copy()
        side_rows["action_type"] = "tackle"
        side_rows["actor_player_id"] = side_rows[side_col]
        side_rows["opponent_player_id"] = None
        side_rows["action_success"] = (
            side_rows["actor_player_id"] == side_rows["possessionEvents_challengeWinnerPlayerId"]
        ).astype(float)
        side_rows.loc[side_rows["possessionEvents_challengeWinnerPlayerId"].isna(), "action_success"] = float("nan")
        rows.append(side_rows)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def map_pff_events(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    type_col = "possessionEvents_possessionEventType"
    df = extract_ball_coordinates(df)
    df = normalize_attacking_direction(df)
    df = convert_to_statsbomb_grid(df)
    df = detect_and_exclude_shootout(df, filename)
    df = df[~(df[type_col].isna() & df.get("gameEvents_gameEventType", pd.Series()).isin(EXCLUDE_GAME_EVENT_TYPES))]

    rows = []

    simple_mask = df[type_col].isin(SIMPLE_TYPE_MAP.keys())
    simple_rows = df[simple_mask].copy()
    simple_rows["action_type"] = simple_rows[type_col].map(SIMPLE_TYPE_MAP)
    simple_rows["actor_player_id"] = simple_rows.get("gameEvents_playerId")
    if "possessionEvents_shotOutcomeType" in simple_rows.columns:
        simple_rows["result_name"] = None
        shot_mask2 = simple_rows["action_type"] == "shot"
        simple_rows.loc[shot_mask2, "result_name"] = simple_rows.loc[shot_mask2, "possessionEvents_shotOutcomeType"]
    rows.append(simple_rows)

    it_rows = df[df[type_col] == "IT"].copy()
    tt_col = "initialTouch_initialTouchType"
    if tt_col in it_rows.columns:
        bad_it = it_rows[it_rows[tt_col].isin(["M", "B"])].copy()
        bad_it["action_type"] = "bad_touch"
        bad_it["actor_player_id"] = bad_it.get("gameEvents_playerId")
        bad_it["action_success"] = 0.0
        rows.append(bad_it)

    tc_rows = df[df[type_col] == "TC"].copy()
    tc_rows["action_type"] = "bad_touch"
    tc_rows["actor_player_id"] = tc_rows.get("possessionEvents_touchPlayerId")
    tc_rows["action_success"] = 0.0
    rows.append(tc_rows)

    re_rows = df[df[type_col] == "RE"].copy()
    if len(re_rows) > 0:
        re_rows["action_type"] = "recovery"
        re_rows["actor_player_id"] = re_rows.get("possessionEvents_rebounderPlayerId")
        outcome = re_rows["possessionEvents_reboundOutcomeType"]
        success = pd.Series(float("nan"), index=re_rows.index)
        success[outcome == "R"] = 1.0
        success[outcome == "T"] = 0.0
        ambiguous_idx = re_rows.index[outcome == "P"]
        for idx in ambiguous_idx:
            pos = df.index.get_loc(idx) if idx in df.index else None
        re_rows["action_success"] = success
        rows.append(re_rows)

    challenge_expanded = expand_challenge_events(df)
    if len(challenge_expanded) > 0:
        rows.append(challenge_expanded)

    rows = [r for r in rows if len(r) > 0]
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # 핵심 수정: 타입별로 잘라 이어 붙였으므로 이 시점 행 순서는 시간 순서가
    # 아니다. eventTime으로 정렬해 "행 순서 = 시간 순서"를 이후 단계에 보장한다.
    if "eventTime" in result.columns and len(result) > 0:
        result = result.sort_values("eventTime").reset_index(drop=True)

    if filename in OWN_GOAL_FILES and len(result) > 0:
        og = OWN_GOAL_FILES[filename]
        credit_row = pd.DataFrame([{
            "action_type": "owngoal_credit",
            "actor_player_id": None,
            "gameEvents_teamName": og["benefiting_team"],
            "result_name": "OwnGoalFor",
            "eventTime": result["eventTime"].max() if "eventTime" in result.columns else None,
        }])
        result = pd.concat([result, credit_row], ignore_index=True)
        print(f"  -> 자책골 크레딧 추가: {filename} ({og['conceding_team']} 실점 -> {og['benefiting_team']} 득점)")

    return result


def run_stage_a():
    files = sorted(PFF_DIR.glob("*.json"))
    print(f"[A단계] 처리 대상 경기 파일 수: {len(files)}")
    failed = []
    for i, f in enumerate(files, 1):
        try:
            with open(f, encoding="utf-8") as fh:
                events = json.load(fh)
            df = pd.json_normalize(events, sep="_")
            mapped = map_pff_events(df, f.name)
            mapped["match_file"] = f.name
            mapped.to_parquet(ACTIONS_DIR / f"{f.stem}.parquet")
        except Exception as e:
            failed.append((f.name, str(e)))
        if i % 50 == 0:
            print(f"  {i}/{len(files)} 완료")
    print(f"[A단계] 완료: {ACTIONS_DIR}")
    if failed:
        print(f"[A단계] 실패 {len(failed)}건: {failed[:5]}")


# =============================================================================
# B. 피처 생성
# =============================================================================

GOAL_X, GOAL_Y = 120.0, 40.0
N_PREV_ACTIONS = 2
ACTION_TYPES = [
    "pass", "cross", "dribble", "take_on", "shot", "tackle",
    "interception", "clearance", "foul", "bad_touch",
    "keeper_action", "block", "recovery", "owngoal",
]


def dist_angle_to_goal(x, y):
    dx = GOAL_X - x
    dy = GOAL_Y - y
    return np.sqrt(dx**2 + dy**2), np.arctan2(dy.abs(), dx)


def add_location_features(df, prefix):
    sx, sy = df[f"{prefix}start_x"], df[f"{prefix}start_y"]
    ex, ey = df[f"{prefix}end_x"], df[f"{prefix}end_y"]
    df[f"{prefix}start_dist_goal"], df[f"{prefix}start_angle_goal"] = dist_angle_to_goal(sx, sy)
    df[f"{prefix}end_dist_goal"], df[f"{prefix}end_angle_goal"] = dist_angle_to_goal(ex, ey)
    df[f"{prefix}move_dx"] = ex - sx
    df[f"{prefix}move_dy"] = ey - sy
    df[f"{prefix}move_dist"] = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    return df


def add_previous_action_context(df, n_prev=N_PREV_ACTIONS):
    context_cols = ["action_type", "action_success", "start_x", "start_y", "end_x", "end_y"]
    for i in range(1, n_prev + 1):
        shifted = df.groupby("period", sort=False)[context_cols].shift(i)
        shifted.columns = [f"a{i}_{c}" for c in context_cols]
        df = pd.concat([df, shifted], axis=1)
    return df


def add_score_diff(df):
    teams = df["team_name"].dropna().unique()
    if len(teams) != 2:
        df["score_diff"] = np.nan
        return df
    team_a, team_b = teams[0], teams[1]
    is_goal = ((df["action_type"] == "shot") & (df["result_name"] == "G")) | (df["action_type"] == "owngoal")
    goal_a = ((df["team_name"] == team_a) & is_goal).astype(int)
    goal_b = ((df["team_name"] == team_b) & is_goal).astype(int)
    cum_a = goal_a.cumsum().shift(1).fillna(0)
    cum_b = goal_b.cumsum().shift(1).fillna(0)
    is_team_a = df["team_name"] == team_a
    df["score_diff"] = np.where(is_team_a, cum_a - cum_b, cum_b - cum_a)
    return df


def build_onehot(df):
    cat_type = pd.CategoricalDtype(categories=ACTION_TYPES)
    onehots = [pd.get_dummies(df["action_type"].astype(cat_type), prefix="a0_type")]
    for i in range(1, N_PREV_ACTIONS + 1):
        onehots.append(pd.get_dummies(df[f"a{i}_action_type"].astype(cat_type), prefix=f"a{i}_type"))
    return pd.concat(onehots, axis=1)


def extract_actor_grade(df):
    grade = pd.Series(np.nan, index=df.index)
    simple_map = {
        "pass": "grades_passerGrade", "cross": "grades_crosserGrade",
        "shot": "grades_shooterGrade", "clearance": "grades_clearerGrade",
        "dribble": "grades_ballCarrierGrade", "take_on": "grades_dribblerGrade",
    }
    for atype, gcol in simple_map.items():
        if gcol in df.columns:
            m = df["action_type"] == atype
            grade.loc[m] = df.loc[m, gcol]

    if "grades_touchGrade" in df.columns:
        m = df["action_type"] == "bad_touch"
        grade.loc[m] = df.loc[m, "grades_touchGrade"]

    tackle_mask = df["action_type"] == "tackle"
    is_duel_style = tackle_mask & df.get("opponent_player_id", pd.Series(index=df.index)).isna()
    is_regular_tackle = tackle_mask & ~is_duel_style

    if "grades_challengerGrade" in df.columns:
        grade.loc[is_regular_tackle] = df.loc[is_regular_tackle, "grades_challengerGrade"]

    home_id_col, away_id_col = "possessionEvents_homeDuelPlayerId", "possessionEvents_awayDuelPlayerId"
    home_grade_col, away_grade_col = "grades_homeDuelGrade", "grades_awayDuelGrade"
    if all(c in df.columns for c in [home_id_col, home_grade_col]):
        home_match = is_duel_style & (df["actor_player_id"] == df[home_id_col])
        grade.loc[home_match] = df.loc[home_match, home_grade_col]
    if all(c in df.columns for c in [away_id_col, away_grade_col]):
        away_match = is_duel_style & (df["actor_player_id"] == df[away_id_col])
        grade.loc[away_match] = df.loc[away_match, away_grade_col]
    return grade


def prepare_clean_schema(df):
    df = df.copy()
    df["action_type"] = df["action_type"].replace({"owngoal_credit": "owngoal"})

    for src, dst in [("gameEvents_period", "period"), ("gameEvents_teamName", "team_name")]:
        if src in df.columns:
            df[dst] = df[src]

    if "result_name" not in df.columns:
        df["result_name"] = None
    if "action_success" not in df.columns:
        df["action_success"] = np.nan

    time_col = "possessionEvents_gameClock"
    df["time_seconds"] = df[time_col] if time_col in df.columns else np.nan

    for src, dst in [
        ("initialTouch_initialPressureType", "initial_pressure_type"),
        ("possessionEvents_pressureType", "event_pressure_type"),
    ]:
        df[dst] = df[src] if src in df.columns else None

    df["actor_grade"] = extract_actor_grade(df)

    keep_cols = [
        "match_file", "period", "team_name", "actor_player_id",
        "action_type", "result_name", "action_success",
        "start_x", "start_y", "end_x", "end_y", "time_seconds",
        "eventTime", "initial_pressure_type", "event_pressure_type", "actor_grade",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def run_stage_b():
    files = sorted(ACTIONS_DIR.glob("*.parquet"))
    print(f"\n[B단계] 처리 대상 파일 수: {len(files)}")
    X_parts, id_parts = [], []
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        df = prepare_clean_schema(df)
        df = df.dropna(subset=["period"]).reset_index(drop=True)
        if len(df) == 0:
            continue

        df = add_score_diff(df)
        df = add_previous_action_context(df)
        df = add_location_features(df, prefix="")
        for i2 in range(1, N_PREV_ACTIONS + 1):
            df = add_location_features(df, prefix=f"a{i2}_")

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

        X = pd.concat([onehot, df[numeric_cols]], axis=1)
        ids = df[[
            "match_file", "period", "team_name", "actor_player_id", "action_type",
            "eventTime", "initial_pressure_type", "event_pressure_type", "actor_grade",
        ]]
        X_parts.append(X)
        id_parts.append(ids)
        if i % 50 == 0:
            print(f"  {i}/{len(files)} 완료")

    X_full = pd.concat(X_parts, ignore_index=True)
    id_full = pd.concat(id_parts, ignore_index=True)
    print(f"[B단계] X shape: {X_full.shape}")

    X_full.to_parquet(FEATURES_DIR / "X_pff.parquet")
    id_full.to_parquet(FEATURES_DIR / "ids_pff.parquet")
    print(f"[B단계] 완료: {FEATURES_DIR}")


# =============================================================================
# C. VAEP 추론
# =============================================================================

def run_stage_c():
    print(f"\n[C단계] 데이터 로딩...")
    X = pd.read_parquet(FEATURES_DIR / "X_pff.parquet")
    ids = pd.read_parquet(FEATURES_DIR / "ids_pff.parquet")
    assert len(X) == len(ids), f"행 수 불일치: X={len(X)}, ids={len(ids)}"
    print(f"[C단계] 정합성 확인 통과: {len(X)}행")

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    success_cols = [c for c in X.columns if c.endswith("action_success")]
    X[success_cols] = X[success_cols].astype(float)

    m_scores = xgb.XGBClassifier()
    m_scores.load_model(MODEL_DIR / "vaep_scores.json")
    m_concedes = xgb.XGBClassifier()
    m_concedes.load_model(MODEL_DIR / "vaep_concedes.json")

    p_scores = m_scores.predict_proba(X)[:, 1]
    p_concedes = m_concedes.predict_proba(X)[:, 1]

    result = ids.copy()
    result["p_scores"] = p_scores
    result["p_concedes"] = p_concedes
    result["offensive_value"] = np.nan
    result["defensive_value"] = np.nan

    for (match_file, period), idx in result.groupby(["match_file", "period"], sort=False).groups.items():
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
    result.to_parquet(VAEP_DIR / "pff_vaep_values.parquet")
    print(f"[C단계] 완료: {VAEP_DIR / 'pff_vaep_values.parquet'}")
    return result


# =============================================================================
# D. 시간차 계산
# =============================================================================

def run_stage_d(result=None):
    print(f"\n[D단계] 시간차 계산...")
    if result is None:
        result = pd.read_parquet(VAEP_DIR / "pff_vaep_values.parquet")

    result["time_gap"] = np.nan
    for (match_file, period), idx in result.groupby(["match_file", "period"], sort=False).groups.items():
        idx = idx.sort_values()
        et = result.loc[idx, "eventTime"].values
        gap = np.full(len(idx), np.nan)
        gap[1:] = et[1:] - et[:-1]
        result.loc[idx, "time_gap"] = gap

    result.to_parquet(VAEP_DIR / "pff_vaep_with_timegap.parquet")
    print(f"[D단계] 완료: {VAEP_DIR / 'pff_vaep_with_timegap.parquet'}")

    print("\n=== 최종 정합성 확인 ===")
    print(f"time_gap 음수 개수(0이어야 정상): {(result['time_gap'] < 0).sum()}")
    print(f"time_gap 결측 개수(138건 근처여야 정상): {result['time_gap'].isna().sum()}")
    print(result["time_gap"].describe())
    return result


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    run_stage_a()
    run_stage_b()
    result = run_stage_c()
    run_stage_d(result)
    print("\n=== 전체 파이프라인 완료 ===")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import json
from pathlib import Path

import numpy as np
import pandas as pd

PFF_DIR = Path("FIFA World Cup 2022/Event Data")
RESULT_PATH = Path("data/pff_vaep/pff_vaep_with_timegap.parquet")
OUT_PATH = Path("data/pff_vaep/pff_vaep_final.parquet")

EXCLUDE_GAME_EVENT_TYPES = {"OUT", "SUB", "END", "OFF", "OTB", "ON", "FIRSTKICKOFF",
                             "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "G"}


def build_full_timeline(filename: str) -> pd.DataFrame:
    """IT(S)를 포함한 모든 possessionEvent(메타 이벤트만 제외)의 eventTime을
    시간순으로 뽑는다. 좌표 변환이나 액션 타입 정교화는 전혀 하지 않는다 -
    오직 '전체 흐름에서의 시간차'만 정확히 계산하기 위한 용도."""
    path = PFF_DIR / filename
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    df = pd.json_normalize(events, sep="_")

    type_col = "possessionEvents_possessionEventType"
    ge_col = "gameEvents_gameEventType"
    period_col = "gameEvents_period"

    # 메타 이벤트(OUT/SUB 등)만 제외, IT(S)를 포함한 나머지는 전부 유지
    df = df[~(df[type_col].isna() & df.get(ge_col, pd.Series()).isin(EXCLUDE_GAME_EVENT_TYPES))]
    df = df.dropna(subset=[period_col, "eventTime"])

    df = df.sort_values("eventTime").reset_index(drop=True)

    df["full_time_gap"] = np.nan
    for period, idx in df.groupby(period_col, sort=False).groups.items():
        idx = idx.sort_values()
        et = df.loc[idx, "eventTime"].values
        gap = np.full(len(idx), np.nan)
        gap[1:] = et[1:] - et[:-1]
        df.loc[idx, "full_time_gap"] = gap

    return df[[period_col, "eventTime", "full_time_gap"]].rename(columns={period_col: "period"})


if __name__ == "__main__":
    result = pd.read_parquet(RESULT_PATH)
    print(f"기존 결과 행 수: {len(result)}")

    files = sorted(result["match_file"].unique())
    print(f"전체 타임라인을 재구성할 파일 수: {len(files)}")

    timelines = []
    for i, filename in enumerate(files, 1):
        tl = build_full_timeline(filename)
        tl["match_file"] = filename
        timelines.append(tl)
        if i % 20 == 0:
            print(f"  {i}/{len(files)} 완료")

    full_timeline = pd.concat(timelines, ignore_index=True)

    # eventTime(소수점까지 정밀) + match_file + period로 병합
    # (부동소수점 정밀도 문제를 피하기 위해 round로 안전하게 매칭)
    result["_et_key"] = result["eventTime"].round(6)
    full_timeline["_et_key"] = full_timeline["eventTime"].round(6)

    # 같은 (match_file, period, eventTime) 조합이 중복되면 병합 시 행이 불어나므로
    # 먼저 제거한다. 동시에 여러 이벤트가 기록된 경우 첫 번째 값만 사용.
    before_dedup = len(full_timeline)
    full_timeline = full_timeline.drop_duplicates(
        subset=["match_file", "period", "_et_key"], keep="first"
    )
    print(f"full_timeline 중복 제거: {before_dedup} -> {len(full_timeline)}건")

    merged = result.merge(
        full_timeline[["match_file", "period", "_et_key", "full_time_gap"]],
        on=["match_file", "period", "_et_key"],
        how="left",
    )
    merged = merged.drop(columns=["_et_key"])

    print(f"\n병합 후 행 수: {len(merged)} (기존과 같아야 정상)")
    print(f"full_time_gap 매칭 실패(NaN) 건수: {merged['full_time_gap'].isna().sum()}")

    merged.to_parquet(OUT_PATH)
    print(f"\n완료: {OUT_PATH}")

    print("\n=== 기존 time_gap vs 새 full_time_gap 비교 ===")
    print(merged[["time_gap", "full_time_gap"]].describe())
