"""
Compute minutes played per player and reconstruct possession episodes (continuous possession by a single player).

Run from the repository root (see README.md for the run order and data locations).
"""

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
선수별 출전 시간 계산.

로직:
- 각 피리어드의 킥오프 이벤트(FIRSTKICKOFF/SECONDKICKOFF/THIRDKICKOFF/
  FOURTHKICKOFF)로 피리어드 시작 시각을 잡는다.
- 선발(started=True) 선수는 그 경기 첫 킥오프부터 출전 시작.
- 교체 투입(playerOnId)은 그 이벤트 시각부터, 교체/퇴장(playerOffId)은 그
  이벤트 시각까지 출전한 것으로 본다.
- 끝까지 안 빠진 선수는 마지막 피리어드의 마지막 이벤트 시각까지 뛴 것으로 계산.
"""

import json
from pathlib import Path

import pandas as pd

PFF_DIR = Path("FIFA World Cup 2022/Event Data")
ROSTER_DIR = Path("FIFA World Cup 2022/Rosters")

KICKOFF_TYPES = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}


def compute_minutes_for_match(match_file: str) -> pd.DataFrame:
    with open(PFF_DIR / match_file, encoding="utf-8") as f:
        events = json.load(f)
    df = pd.json_normalize(events, sep="_")

    roster_path = ROSTER_DIR / match_file
    with open(roster_path, encoding="utf-8") as f:
        roster = json.load(f)
    roster_df = pd.json_normalize(roster, sep="_")
    roster_df["player_id"] = roster_df["player_id"].astype(str)

    ge_type_col = "gameEvents_gameEventType"
    period_col = "gameEvents_period"

    # 경기 시작(첫 킥오프)과 종료(마지막 이벤트) 시각
    match_start = df["eventTime"].min()
    match_end = df["eventTime"].max()

    # 교체/퇴장 이벤트: playerOnId/playerOffId가 채워진 행
    on_col, off_col = "gameEvents_playerOnId", "gameEvents_playerOffId"
    sub_events = df[df[on_col].notna() | df[off_col].notna()][
        ["eventTime", on_col, off_col]
    ].dropna(how="all", subset=[on_col, off_col])

    minutes = {}
    for _, row in roster_df.iterrows():
        pid = row["player_id"]
        started = row.get("started", False)

        # 이 선수가 투입된 시각
        on_events = sub_events[sub_events[on_col] == float(pid)]
        off_events = sub_events[sub_events[off_col] == float(pid)]

        if started:
            enter_time = match_start
        elif len(on_events) > 0:
            enter_time = on_events["eventTime"].min()
        else:
            minutes[pid] = 0.0  # 출전 자체를 안 함
            continue

        if len(off_events) > 0:
            exit_time = off_events["eventTime"].min()
        else:
            exit_time = match_end

        minutes[pid] = max(0.0, (exit_time - enter_time) / 60.0)

    result = roster_df[["player_id", "player_nickname", "positionGroupType", "started"]].copy()
    result["minutes_played"] = result["player_id"].map(minutes)
    result["match_file"] = match_file
    return result


if __name__ == "__main__":
    files = sorted(f.name for f in PFF_DIR.glob("*.json"))
    print(f"처리 대상 경기 수: {len(files)}")

    all_minutes = []
    failed = []
    for i, fname in enumerate(files, 1):
        try:
            m = compute_minutes_for_match(fname)
            all_minutes.append(m)
        except Exception as e:
            failed.append((fname, str(e)))
        if i % 20 == 0:
            print(f"  {i}/{len(files)} 완료")

    minutes_df = pd.concat(all_minutes, ignore_index=True)
    print(f"\n총 행 수: {len(minutes_df)}")
    if failed:
        print(f"실패 {len(failed)}건: {failed[:5]}")

    print("\n=== minutes_played 기술통계 ===")
    print(minutes_df["minutes_played"].describe())

    # 선수별 누적 출전시간(대회 전체)
    total_minutes = minutes_df.groupby("player_id")["minutes_played"].sum().reset_index()
    total_minutes.columns = ["player_id", "total_minutes"]

    minutes_df.to_parquet("data/pff_vaep/minutes_by_match.parquet")
    total_minutes.to_parquet("data/pff_vaep/minutes_total.parquet")
    print("\n완료: data/pff_vaep/minutes_by_match.parquet, minutes_total.parquet")

    print("\n=== 대회 전체 누적 출전시간 상위 10명 ===")
    top = total_minutes.merge(
        minutes_df[["player_id", "player_nickname"]].drop_duplicates("player_id"),
        on="player_id",
    ).sort_values("total_minutes", ascending=False)
    print(top.head(10))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
에피소드(볼 점유 구간) 단위 재구성 + 최종 압박 대응 분석

핵심 변경: "행(row) 단위" -> "에피소드(같은 선수가 연속으로 볼을 소유한 구간) 단위"

에피소드 정의:
- 원본 전체 이벤트(IT(S) 포함)를 시간순으로 훑으며, "행위자가 같은 채로 이어지는
  연속 구간"을 하나의 에피소드로 묶는다.
- Challenge(CH)/Foul(FO)는 행위자가 모호하므로 에피소드 연결 판단에서 제외
  (건너뛰고 그 앞뒤로 같은 선수가 이어지는지만 본다).
- 시간: 에피소드 첫 행 eventTime ~ 마지막 행 eventTime
- 이동거리: 첫 행 시작좌표 -> 마지막 행 종료좌표의 직선거리(변위)
- VAEP: 에피소드에 속한 모든 행의 vaep_value 합산
  (연속된 상태 확률차 정의상 텔레스코핑되어 "에피소드 시작~끝의 순가치"가 됨)
- 압박 여부: 에피소드 첫 행(수신 시점)의 initial_pressure_type 사용
  (원래도 "받는 순간의 압박"을 뜻하므로 에피소드 시작 시점과 정확히 일치)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mplsoccer import Standardizer

PFF_DIR = Path("FIFA World Cup 2022/Event Data")
VAEP_PATH = Path("data/pff_vaep/pff_vaep_final.parquet")
OUT_DIR = Path("data/pff_vaep")

EXCLUDE_GAME_EVENT_TYPES = {"OUT", "SUB", "END", "OFF", "OTB", "ON", "FIRSTKICKOFF",
                             "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "G"}

# 에피소드 구성에 포함할 타입과, 각 타입의 행위자 필드
ACTOR_FIELD_BY_TYPE = {
    "IT": "gameEvents_playerId",
    "TC": "possessionEvents_touchPlayerId",
    "RE": "possessionEvents_rebounderPlayerId",
    "PA": "gameEvents_playerId",
    "CR": "gameEvents_playerId",
    "SH": "gameEvents_playerId",
    "CL": "gameEvents_playerId",
    "BC": "gameEvents_playerId",
}
# FO는 행위자 모호(가해자 필드가 거의 비어있음 - 이전 검증에서 확인)로 계속 제외.
# CH(Challenge)는 원래 전부 제외했으나, take_on(challengeType == "D")은 행위자가
# 실제 볼 소유자(dribblerPlayerId)이므로 예외적으로 에피소드에 포함시킨다.
# tackle 계열/순수 듀얼은 행위자가 볼 소유자가 아니므로 여전히 제외.
TAKE_ON_CHALLENGE_TYPES = {"D"}


# ---------------------------------------------------------------------------
# 좌표 처리 함수 (기존 파이프라인과 동일 - 일관성 유지를 위해 재사용)
# ---------------------------------------------------------------------------
def extract_ball_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    def _get_xy(ball_field):
        if isinstance(ball_field, list) and len(ball_field) > 0:
            return ball_field[0].get("x"), ball_field[0].get("y")
        return None, None

    df["start_x"], df["start_y"] = zip(*df["ball"].map(_get_xy))
    period_col = "gameEvents_period"
    df["end_x"] = float("nan")
    df["end_y"] = float("nan")
    for period, idx in df.groupby(period_col, sort=False).groups.items():
        idx = idx.sort_values()
        df.loc[idx, "end_x"] = df.loc[idx, "start_x"].shift(-1)
        df.loc[idx, "end_y"] = df.loc[idx, "start_y"].shift(-1)
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

    x_corner = df["start_x"] + length / 2
    y_corner = df["start_y"] + width / 2
    df["start_x"], df["start_y"] = standardizer.transform(x_corner.values, y_corner.values)
    ex_corner = df["end_x"] + length / 2
    ey_corner = df["end_y"] + width / 2
    df["end_x"], df["end_y"] = standardizer.transform(ex_corner.values, ey_corner.values)
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
    cutoff_idx = shots.index[is_shootout_coord].min()
    return df[df.index < cutoff_idx]


# ---------------------------------------------------------------------------
# 에피소드 구성
# ---------------------------------------------------------------------------
def build_episodes_for_match(filename: str) -> pd.DataFrame:
    with open(PFF_DIR / filename, encoding="utf-8") as f:
        events = json.load(f)
    df = pd.json_normalize(events, sep="_")

    df = extract_ball_coordinates(df)
    df = normalize_attacking_direction(df)
    df = convert_to_statsbomb_grid(df)
    df = detect_and_exclude_shootout(df, filename)

    type_col = "possessionEvents_possessionEventType"
    ge_col = "gameEvents_gameEventType"
    period_col = "gameEvents_period"
    pressure_col = "initialTouch_initialPressureType"

    # 메타 이벤트 제외
    df = df[~(df[type_col].isna() & df.get(ge_col, pd.Series()).isin(EXCLUDE_GAME_EVENT_TYPES))]

    # CH 중 take_on만 남기고 나머지(태클/듀얼)는 제외, 그 외 지정된 타입은 그대로 유지
    ct_col = "possessionEvents_challengeType"
    is_ch = df[type_col] == "CH"
    is_take_on = is_ch & df.get(ct_col, pd.Series(index=df.index)).isin(TAKE_ON_CHALLENGE_TYPES)
    keep_mask = df[type_col].isin(ACTOR_FIELD_BY_TYPE.keys()) | is_take_on
    df = df[keep_mask].copy()

    df = df.dropna(subset=[period_col, "eventTime"])
    df = df.sort_values("eventTime").reset_index(drop=True)

    # 타입별 행위자 필드에서 실제 행위자 id 추출 (take_on은 드리블러가 행위자)
    def get_actor(row):
        if row[type_col] == "CH":
            return row.get("possessionEvents_dribblerPlayerId")
        field = ACTOR_FIELD_BY_TYPE.get(row[type_col])
        return row.get(field) if field else None

    df["actor_id"] = df.apply(get_actor, axis=1)

    episodes = []
    for period, idx in df.groupby(period_col, sort=False).groups.items():
        idx = idx.sort_values()
        sub = df.loc[idx].reset_index(drop=True)

        # 행위자가 바뀌는 지점마다 새 에피소드 그룹 번호 부여
        actor_changed = sub["actor_id"] != sub["actor_id"].shift(1)
        episode_group = actor_changed.cumsum()

        for _, ep in sub.groupby(episode_group):
            if ep["actor_id"].isna().all():
                continue
            episodes.append({
                "match_file": filename,
                "period": period,
                "actor_player_id": ep["actor_id"].iloc[0],
                "team_name": ep["gameEvents_teamName"].iloc[0] if "gameEvents_teamName" in ep.columns else None,
                "episode_start_time": ep["eventTime"].iloc[0],
                "episode_end_time": ep["eventTime"].iloc[-1],
                "start_x": ep["start_x"].iloc[0],
                "start_y": ep["start_y"].iloc[0],
                "end_x": ep["end_x"].iloc[-1],
                "end_y": ep["end_y"].iloc[-1],
                "initial_pressure_type": ep[pressure_col].iloc[0] if pressure_col in ep.columns else None,
                "n_touches": len(ep),
                "member_event_times": ep["eventTime"].tolist(),  # VAEP 합산용 매칭 키
            })

    return pd.DataFrame(episodes)


if __name__ == "__main__":
    files = sorted(f.name for f in PFF_DIR.glob("*.json"))
    print(f"처리 대상 경기 수: {len(files)}")

    all_episodes = []
    for i, fname in enumerate(files, 1):
        ep = build_episodes_for_match(fname)
        all_episodes.append(ep)
        if i % 20 == 0:
            print(f"  {i}/{len(files)} 완료")

    episodes_df = pd.concat(all_episodes, ignore_index=True)
    print(f"\n전체 에피소드 수: {len(episodes_df)}")

    # 파생 변수: 시간, 변위
    episodes_df["duration"] = episodes_df["episode_end_time"] - episodes_df["episode_start_time"]
    episodes_df["displacement"] = np.sqrt(
        (episodes_df["end_x"] - episodes_df["start_x"]) ** 2
        + (episodes_df["end_y"] - episodes_df["start_y"]) ** 2
    )

    # 기존 VAEP 결과와 매칭해서 에피소드별 vaep_value 합산
    vaep_data = pd.read_parquet(VAEP_PATH)
    vaep_data["_et_key"] = vaep_data["eventTime"].round(6)
    vaep_lookup = vaep_data.set_index(["match_file", "period", "_et_key"])["vaep_value"].to_dict()

    def sum_episode_vaep(row):
        total = 0.0
        found_any = False
        for t in row["member_event_times"]:
            key = (row["match_file"], row["period"], round(t, 6))
            val = vaep_lookup.get(key)
            if val is not None and not pd.isna(val):
                total += val
                found_any = True
        return total if found_any else np.nan

    print("\nVAEP 합산 중 (시간이 다소 걸릴 수 있습니다)...")
    episodes_df["vaep_sum"] = episodes_df.apply(sum_episode_vaep, axis=1)
    episodes_df = episodes_df.drop(columns=["member_event_times"])

    print(f"vaep_sum 결측 비율: {episodes_df['vaep_sum'].isna().mean():.2%} "
          f"(에피소드가 전부 IT(S)/제외타입으로만 구성된 경우 - 정상)")

    episodes_df.to_parquet(OUT_DIR / "episodes.parquet")
    print(f"\n완료: {OUT_DIR / 'episodes.parquet'}")

    print("\n=== duration 기술통계 ===")
    print(episodes_df["duration"].describe())
    print("\n=== displacement 기술통계 ===")
    print(episodes_df["displacement"].describe())
    print("\n=== vaep_sum 기술통계 ===")
    print(episodes_df["vaep_sum"].describe())
