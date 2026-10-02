"""
StatsBomb pass completion rate under pressure, player matching across datasets, rank comparison with VAEP, and statistics (Spearman, Kendall, Cohen's kappa).

Run from the repository root (see README.md for the run order and data locations).
"""

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
StatsBomb Open Data(2022 카타르 월드컵, competition_id=43, season_id=106)에서
압박 시(under_pressure) 패스 성공률을 직접 추출.

StatsBomb 폴더 구조 가정: STATSBOMB_DIR/matches/43/106.json,
STATSBOMB_DIR/events/<match_id>.json
"""

import json
from pathlib import Path

import pandas as pd

STATSBOMB_DIR = Path("statsbomb/data")
matches_path = STATSBOMB_DIR / "matches" / "43" / "106.json"
with open(matches_path, encoding="utf-8") as f:
    matches = json.load(f)

match_ids = [m["match_id"] for m in matches]
print(f"2022 카타르 월드컵 경기 수: {len(match_ids)}")

all_passes = []
for match_id in match_ids:
    events_path = STATSBOMB_DIR / "events" / f"{match_id}.json"
    with open(events_path, encoding="utf-8") as f:
        events = json.load(f)

    for ev in events:
        if ev.get("type", {}).get("name") != "Pass":
            continue
        player = ev.get("player", {})
        pass_info = ev.get("pass", {})
        all_passes.append({
            "match_id": match_id,
            "player_name": player.get("name"),
            "player_id": player.get("id"),
            "under_pressure": ev.get("under_pressure", False),
            "success": "outcome" not in pass_info,  # outcome 없음 = 완료 (StatsBomb 관례)
        })

passes_df = pd.DataFrame(all_passes)
print(f"전체 패스 이벤트 수: {len(passes_df)}")

pressured = passes_df[passes_df["under_pressure"]]
print(f"압박 시 패스 수: {len(pressured)}")

completion_sb = pressured.groupby(["player_id", "player_name"]).agg(
    n_pressured_passes=("success", "count"),
    completion_rate=("success", "mean"),
).reset_index()

completion_sb = completion_sb.sort_values("n_pressured_passes", ascending=False)
print("\n=== 압박 시 패스 수 상위 15명 (검산용) ===")
print(completion_sb.head(15).to_string())

completion_sb.to_parquet("data/pff_vaep/statsbomb_completion_rate_wc2022.parquet")
print("\n완료: data/pff_vaep/statsbomb_completion_rate_wc2022.parquet")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
StatsBomb 라인업 데이터에서 선수별 실제 출전 포지션을 추출해,
압박 시 패스 성공률 데이터를 미드필더로 한정.
동시에 PFF FC 포지션 분류와 다르게 나오는 선수(로드리 같은 케이스)를
자동으로 찾아 표시.
"""

import json
from pathlib import Path

import pandas as pd

STATSBOMB_DIR = Path("statsbomb/data")

# StatsBomb 포지션명 -> 대분류 매핑
SB_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Right Back": "D", "Left Back": "D", "Right Center Back": "D",
    "Left Center Back": "D", "Center Back": "D",
    "Right Wing Back": "D", "Left Wing Back": "D",
    "Right Defensive Midfield": "M", "Left Defensive Midfield": "M",
    "Center Defensive Midfield": "M",
    "Right Center Midfield": "M", "Left Center Midfield": "M", "Center Midfield": "M",
    "Right Midfield": "M", "Left Midfield": "M",
    "Right Attacking Midfield": "M", "Left Attacking Midfield": "M",
    "Center Attacking Midfield": "M",
    "Right Wing": "F", "Left Wing": "F",
    "Center Forward": "F", "Right Center Forward": "F", "Left Center Forward": "F",
    "Secondary Striker": "F",
}

matches_path = STATSBOMB_DIR / "matches" / "43" / "106.json"
with open(matches_path, encoding="utf-8") as f:
    matches = json.load(f)
match_ids = [m["match_id"] for m in matches]

position_records = []
for match_id in match_ids:
    lineups_path = STATSBOMB_DIR / "lineups" / f"{match_id}.json"
    with open(lineups_path, encoding="utf-8") as f:
        lineups = json.load(f)

    for team in lineups:
        for player in team.get("lineup", []):
            player_name = player.get("player_name")
            player_id = player.get("player_id")
            for pos in player.get("positions", []):
                position_records.append({
                    "player_id": player_id,
                    "player_name": player_name,
                    "position_name": pos.get("position"),
                })

pos_df = pd.DataFrame(position_records)
print(f"전체 포지션 기록 수: {len(pos_df)}")

pos_df["position_broad_sb"] = pos_df["position_name"].map(SB_POSITION_MAP)
unmapped = pos_df[pos_df["position_broad_sb"].isna()]["position_name"].unique()
if len(unmapped) > 0:
    print(f"매핑 안 된 포지션명: {list(unmapped)}")

# 선수별로 가장 많이 뛴 포지션(최빈값)을 대표 포지션으로
player_position = (
    pos_df.dropna(subset=["position_broad_sb"])
    .groupby(["player_id", "player_name"])["position_broad_sb"]
    .agg(lambda s: s.value_counts().idxmax())
    .reset_index()
)

print(f"\n=== StatsBomb 기준 포지션 대분류 분포 ===")
print(player_position["position_broad_sb"].value_counts())

player_position.to_parquet("data/pff_vaep/statsbomb_player_position.parquet")
print("\n완료: data/pff_vaep/statsbomb_player_position.parquet")

# 미드필더로 압박 시 패스 성공률 필터링
completion = pd.read_parquet("data/pff_vaep/statsbomb_completion_rate_wc2022.parquet")
completion_m = completion.merge(player_position, on=["player_id", "player_name"], how="inner")
completion_m = completion_m[completion_m["position_broad_sb"] == "M"]
print(f"\nStatsBomb 기준 미드필더로 한정된 선수 수: {completion_m['player_id'].nunique()}")

completion_m.to_parquet("data/pff_vaep/statsbomb_completion_rate_M.parquet")
print("완료: data/pff_vaep/statsbomb_completion_rate_M.parquet")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
에피소드 데이터에 '종료 행위 타입'을 붙여서, 패스로 종결된 에피소드만
필터링한 뒤 저희 지표(VAEP 절단평균) 선수 순위를 재산출.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_POSITION = "M"
MIN_TOTAL_MINUTES = 90
MIN_PRESSURED_EPISODES = 20
MIN_DISTINCT_MATCHES = 3
EXCLUDE_NICKNAMES = {"Rodri"}


def trimmed_mean(s: pd.Series, trim: float = 0.05) -> float:
    if len(s) < 3:
        return s.mean()
    lower, upper = s.quantile(trim), s.quantile(1 - trim)
    trimmed = s[(s >= lower) & (s <= upper)]
    return trimmed.mean() if len(trimmed) > 0 else s.mean()


# ---------------------------------------------------------------------------
# 1) 에피소드에 종료 행위 타입 붙이기
# ---------------------------------------------------------------------------
episodes = pd.read_parquet("data/pff_vaep/episodes.parquet")

ACTIONS_DIR = Path("data/pff_actions_ready")
type_parts = []
for f in sorted(ACTIONS_DIR.glob("*.parquet")):
    d = pd.read_parquet(f, columns=["match_file", "gameEvents_period", "eventTime", "action_type"])
    d = d.rename(columns={"gameEvents_period": "period"})
    type_parts.append(d)
type_lookup = pd.concat(type_parts, ignore_index=True)
type_lookup["_et_key"] = type_lookup["eventTime"].round(6)
type_lookup = type_lookup.drop_duplicates(subset=["match_file", "period", "_et_key"], keep="first")

episodes["_et_key"] = episodes["episode_end_time"].round(6)
episodes = episodes.merge(
    type_lookup[["match_file", "period", "_et_key", "action_type"]].rename(
        columns={"action_type": "final_action_type"}
    ),
    on=["match_file", "period", "_et_key"], how="left",
).drop(columns=["_et_key"])

print(f"final_action_type 매칭 실패 비율: {episodes['final_action_type'].isna().mean():.2%}")
print("\n=== 종료 행위 타입 분포 ===")
print(episodes["final_action_type"].value_counts())

episodes.to_parquet("data/pff_vaep/episodes_with_final_type.parquet")

# ---------------------------------------------------------------------------
# 2) 패스로 종결된 에피소드만 필터링 + 기존과 동일한 파이프라인
# ---------------------------------------------------------------------------
episodes = episodes[episodes["final_action_type"] == "pass"].copy()
episodes = episodes.dropna(subset=["vaep_sum", "duration", "displacement"])
print(f"\n패스로 종결된 에피소드(결측 제외): {len(episodes)}")

players = pd.read_csv("FIFA World Cup 2022/players.csv").drop_duplicates(subset="id")
players["id"] = players["id"].astype(str)
POSITION_GROUP_MAP = {
    "D": "D", "RCB": "D", "LCB": "D", "RB": "D", "LB": "D", "CB": "D",
    "M": "M", "CM": "M", "DM": "M", "AM": "M",
    "F": "F", "CF": "F", "RW": "F", "LW": "F",
    "GK": "GK",
}
players["position_broad"] = players["positionGroupType"].map(POSITION_GROUP_MAP)
episodes["actor_id_str"] = episodes["actor_player_id"].astype("Int64").astype(str)
episodes = episodes.merge(players[["id", "position_broad", "nickname"]],
                            left_on="actor_id_str", right_on="id", how="left")
episodes = episodes[~episodes["nickname"].isin(EXCLUDE_NICKNAMES)]

minutes_total = pd.read_parquet("data/pff_vaep/minutes_total.parquet")
qualified_players = minutes_total[minutes_total["total_minutes"] >= MIN_TOTAL_MINUTES]["player_id"]
episodes = episodes[episodes["actor_id_str"].isin(qualified_players)]

episodes = episodes[episodes["initial_pressure_type"].isin(["N", "P"])].copy()
episodes["is_pressured"] = episodes["initial_pressure_type"] == "P"

pos_data = episodes[episodes["position_broad"] == TARGET_POSITION]
pressured_pos = pos_data[pos_data["is_pressured"]]

counts = pressured_pos.groupby("actor_player_id").agg(
    n_pressured=("vaep_sum", "count"), n_distinct_matches=("match_file", "nunique")
)
qualified_ids = counts[(counts["n_pressured"] >= MIN_PRESSURED_EPISODES) &
                        (counts["n_distinct_matches"] >= MIN_DISTINCT_MATCHES)].index

pass_only_summary = pressured_pos[pressured_pos["actor_player_id"].isin(qualified_ids)].groupby(
    "actor_player_id"
).agg(
    nickname=("nickname", "first"),
    n_pressured=("vaep_sum", "count"),
    n_distinct_matches=("match_file", "nunique"),
    vaep_trimmed_mean=("vaep_sum", trimmed_mean),
    vaep_mean=("vaep_sum", "mean"),  # 강건성 비교용 원본(비절단) 평균
).sort_values("vaep_trimmed_mean", ascending=False)

print(f"\n{TARGET_POSITION} 포지션군, 패스 종결 에피소드 기준 최종 선수 수: {len(pass_only_summary)}")
print(pass_only_summary.to_string())

pass_only_summary.to_parquet(f"data/pff_vaep/pass_only_vaep_summary_{TARGET_POSITION}.parquet")
print(f"\n완료: data/pff_vaep/pass_only_vaep_summary_{TARGET_POSITION}.parquet")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
강건성 확인용: 절단평균이 아닌 원본(비절단) 평균으로 동일한 비교를 재수행.
나머지 로직(매칭·중복해결·포지션 이중검증·최소표본필터)은 64번과 완전히 동일.
"""

import pandas as pd
from rapidfuzz import fuzz, process
from scipy import stats

# ---------------------------------------------------------------------------
# 1) 이름 매칭 (pass_only 목록 기준으로 재실행)
# ---------------------------------------------------------------------------
MIN_SB_PRESSURED_PASSES = 20  # PFF 쪽 MIN_PRESSURED_EPISODES=20과 일관성 맞춤

pff = pd.read_parquet("data/pff_vaep/pass_only_vaep_summary_M.parquet").reset_index()
sb = pd.read_parquet("data/pff_vaep/statsbomb_completion_rate_M.parquet")

n_before = sb["player_id"].nunique()
sb = sb[sb["n_pressured_passes"] >= MIN_SB_PRESSURED_PASSES]
print(f"StatsBomb 측 최소 표본 필터 적용: {n_before}명 -> {sb['player_id'].nunique()}명")

sb_names = sb["player_name"].tolist()

MIN_MATCH_SCORE = 60  # 이 미만은 '매칭 실패'로 간주 (아래에서 수동 검증 대상으로 별도 출력)

matches = []
for _, row in pff.iterrows():
    result = process.extractOne(row["nickname"], sb_names, scorer=fuzz.token_sort_ratio)
    if result is None:
        continue
    sb_name, score, _ = result
    matches.append({
        "actor_player_id": row["actor_player_id"],
        "pff_nickname": row["nickname"],
        "sb_name": sb_name,
        "match_score": score,
    })
matches_df = pd.DataFrame(matches)

print(f"매칭 시도: {len(matches_df)}명")

# 같은 StatsBomb 이름이 두 명 이상의 PFF 선수와 매칭된 경우:
# 점수가 가장 높은 매칭 하나만 살리고 나머지(낮은 점수 = 억지 매칭일 가능성)는 제거.
# (예: 진짜 정답인 100점짜리 매칭이, 다른 선수의 진짜 짝이 필터링으로 사라지며
#  생긴 40점대 억지 매칭 때문에 같이 지워지는 걸 방지)
dup_names = matches_df["sb_name"].value_counts()
dup_names = dup_names[dup_names > 1].index.tolist()
if dup_names:
    print(f"\n경고: 중복 매칭된 StatsBomb 이름 발견 (최고점만 유지): {dup_names}")
    for name in dup_names:
        dup_rows = matches_df[matches_df["sb_name"] == name].sort_values("match_score", ascending=False)
        print(f"  '{name}': {list(zip(dup_rows['pff_nickname'], dup_rows['match_score'].round(1)))}"
              f" -> '{dup_rows.iloc[0]['pff_nickname']}' 유지")
        loser_idx = dup_rows.index[1:]
        matches_df = matches_df.drop(index=loser_idx)

# 낮은 점수 매칭은 자동 채택하지 않고 별도로 보여준 뒤 제거
low_score = matches_df[matches_df["match_score"] < MIN_MATCH_SCORE]
print(f"\n=== 점수 {MIN_MATCH_SCORE} 미만이라 자동 제외된 매칭 "
      f"(수동으로 맞는지 확인 후 필요시 직접 추가하세요) ===")
print(low_score.sort_values("match_score").to_string(index=False))

matches_df = matches_df[matches_df["match_score"] >= MIN_MATCH_SCORE]

# 수동 검증 완료: 점수는 낮았지만 실제로 정답인 매칭 (정식 이름 형식 차이 때문)
MANUAL_OVERRIDES = {
    "Casemiro": "Carlos Henrique Casimiro",
    "Bernardo Silva": "Bernardo Mota Veiga de Carvalho e Silva",
    "Luis Chávez": "Luis Gerardo Chávez Magallón",
    "Rúben Neves": "Rúben Diogo Da Silva Neves",
}
override_rows = []
for pff_name, sb_name in MANUAL_OVERRIDES.items():
    pid = pff.loc[pff["nickname"] == pff_name, "actor_player_id"]
    if len(pid) == 0:
        continue
    override_rows.append({
        "actor_player_id": pid.iloc[0], "pff_nickname": pff_name,
        "sb_name": sb_name, "match_score": 100.0,  # 수동 검증이므로 신뢰도 최고로 표시
    })
matches_df = pd.concat([matches_df, pd.DataFrame(override_rows)], ignore_index=True)

print(f"\n최종 신뢰 가능한 매칭(수동 복원 {len(override_rows)}건 포함): {len(matches_df)}명")

# ---------------------------------------------------------------------------
# 2) 세 데이터(PFF VAEP, 이름매칭, StatsBomb 성공률) 병합
# ---------------------------------------------------------------------------
compare = matches_df.merge(pff, on="actor_player_id", how="left")
compare = compare.merge(sb, left_on="sb_name", right_on="player_name", how="left")
compare = compare.dropna(subset=["completion_rate", "vaep_mean"])

print(f"\n최종 비교 가능한 선수 수: {len(compare)}")
print(f"(참고: 이 {len(compare)}명 전원이 PFF FC 기준 M이자 StatsBomb 기준 M으로 "
      f"이중 확인된 선수들이며, 이후 순위는 이 집합 안에서만 계산됩니다)")

# ---------------------------------------------------------------------------
# 3) 순위 비교 - 상위 20명씩 나란히 표로
# ---------------------------------------------------------------------------
compare["vaep_rank"] = compare["vaep_mean"].rank(ascending=False).astype(int)
compare["completion_rank"] = compare["completion_rate"].rank(ascending=False).astype(int)

cols_vaep_view = ["vaep_rank", "pff_nickname", "vaep_mean", "n_pressured",
                  "completion_rank", "completion_rate", "n_pressured_passes"]
cols_comp_view = ["completion_rank", "pff_nickname", "completion_rate", "n_pressured_passes",
                  "vaep_rank", "vaep_mean", "n_pressured"]

print(f"\n{'=' * 70}")
print("VAEP 기준 상위 20명 (각 선수의 성공률 순위도 같이 표시)")
print("=" * 70)
top20_vaep = compare.sort_values("vaep_rank").head(20)[cols_vaep_view]
print(top20_vaep.to_string(index=False))

print(f"\n{'=' * 70}")
print("압박 시 패스 성공률 기준 상위 20명 (각 선수의 VAEP 순위도 같이 표시)")
print("=" * 70)
top20_comp = compare.sort_values("completion_rank").head(20)[cols_comp_view]
print(top20_comp.to_string(index=False))

compare.to_parquet("data/pff_vaep/final_comparison_M_rawmean.parquet")
print(f"\n완료: data/pff_vaep/final_comparison_M_rawmean.parquet")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
VAEP 순위와 압박 시 패스 성공률 순위 간의 일치도를,
Spearman / Kendall's Tau / Cohen's Kappa 세 가지로 한 번에 계산.
"""

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

compare = pd.read_parquet("data/pff_vaep/final_comparison_M_rawmean.parquet")
print(f"분석 대상 선수 수: {len(compare)}")

# ---------------------------------------------------------------------------
# 1) Spearman
# ---------------------------------------------------------------------------
spearman_r, spearman_p = stats.spearmanr(compare["vaep_mean"], compare["completion_rate"])
print(f"\n[Spearman] rho={spearman_r:.4f}, p={spearman_p:.4f}")
# ---------------------------------------------------------------------------
# 2) Cohen's Kappa (상위 절반 여부를 이진 분류로 변환 후 일치도)
# ---------------------------------------------------------------------------
n = len(compare)
half = n // 2
compare["vaep_top_half"] = compare["vaep_mean"].rank(ascending=False) <= half
compare["completion_top_half"] = compare["completion_rate"].rank(ascending=False) <= half

kappa = cohen_kappa_score(compare["vaep_top_half"], compare["completion_top_half"])
n_agree = (compare["vaep_top_half"] == compare["completion_top_half"]).sum()
print(f"[Cohen's Kappa] kappa={kappa:.4f} "
      f"(상위/하위 절반 일치 {n_agree}/{n}명, 단순 일치율={n_agree/n:.1%})")

print("\n=== 요약 (초록용 문장 예시) ===")
print(f"Spearman's rho = {spearman_r:.3f} (p = {spearman_p:.3f}), "
      f"Cohen's kappa = {kappa:.3f}")
