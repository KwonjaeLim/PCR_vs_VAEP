"""
Mean pass starting x-coordinate per player and its Spearman correlations with both metrics.

Run from the repository root (see README.md for the run order and data locations).
"""

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
분석 1: 역할 변수(전체 패스 평균 출발 x)와 성공률·VAEP 순위의 Spearman 상관.
VAEP·성공률은 기존 45명 결과를 그대로 쓰고, 역할 변수만 새로 계산한다.
"""

from pathlib import Path

import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1) 기존 45명 결과 로드
# ---------------------------------------------------------------------------
comp = pd.read_parquet("data/pff_vaep/final_comparison_M_rawmean.parquet").reset_index()
print(f"분석 대상: {len(comp)}명")

target_ids = set(comp["actor_player_id"].astype("Int64"))

# ---------------------------------------------------------------------------
# 2) 역할 변수 계산: 전체 패스(압박 여부 무관) 평균 출발 x
# ---------------------------------------------------------------------------
ACTIONS_DIR = Path("data/pff_actions_ready")
parts = []
for f in sorted(ACTIONS_DIR.glob("*.parquet")):
    d = pd.read_parquet(f, columns=["action_type", "actor_player_id", "start_x"])
    d = d[(d["action_type"] == "pass") & (d["actor_player_id"].astype("Int64").isin(target_ids))]
    parts.append(d)

passes = pd.concat(parts, ignore_index=True)
passes = passes.dropna(subset=["start_x"])
print(f"전체 패스 수(45명, 좌표 결측 제외): {len(passes)}")

role = passes.groupby("actor_player_id")["start_x"].agg(
    role_mean_start_x="mean", role_n_passes="count"
).reset_index()
role["actor_player_id"] = role["actor_player_id"].astype("Int64")

# ---------------------------------------------------------------------------
# 3) 병합 및 상관 분석
# ---------------------------------------------------------------------------
comp["actor_player_id"] = comp["actor_player_id"].astype("Int64")
m = comp.merge(role, on="actor_player_id", how="left")

missing = m["role_mean_start_x"].isna().sum()
print(f"역할 변수 결측: {missing}명" + (" (전체 패스 자체가 없는 선수 - 확인 필요)" if missing else ""))
m = m.dropna(subset=["role_mean_start_x"])
print(f"최종 분석 대상: {len(m)}명")

print("\n=== 역할 변수(평균 출발 x) 기술통계 ===")
print(m["role_mean_start_x"].describe())

r1, p1 = stats.spearmanr(m["role_mean_start_x"], m["completion_rate"])
r2, p2 = stats.spearmanr(m["role_mean_start_x"], m["vaep_mean"])

print(f"\n[역할 vs 성공률] Spearman rho={r1:.3f}, p={p1:.4f}")
print(f"[역할 vs VAEP]   Spearman rho={r2:.3f}, p={p2:.4f}")

# 세 변수 각각의 순위(1=가장 큰 값). 상관과 같은 방향으로 바로 대조하기 위함.
m["role_rank"] = m["role_mean_start_x"].rank(ascending=False).astype(int)
m["completion_rank"] = m["completion_rate"].rank(ascending=False).astype(int)
m["vaep_rank"] = m["vaep_mean"].rank(ascending=False).astype(int)

cols = ["pff_nickname", "role_mean_start_x", "role_rank",
        "completion_rate", "completion_rank", "vaep_mean", "vaep_rank"]

print("\n=== 역할 변수 상위 10명 (가장 공격적 진영 시작) ===")
print(m.nlargest(10, "role_mean_start_x")[cols].to_string(index=False))

print("\n=== 역할 변수 하위 10명 (가장 수비적 진영 시작) ===")
print(m.nsmallest(10, "role_mean_start_x")[cols].to_string(index=False))

m.to_parquet("data/pff_vaep/role_correlation_M.parquet")
print("\n완료: data/pff_vaep/role_correlation_M.parquet")
