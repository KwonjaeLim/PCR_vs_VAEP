"""
Figure 1: PCR vs. VAEP scatter (color = mean pass starting x-coordinate; labels = PCR and VAEP ranks).

Run from the repository root (see README.md for the run order and data locations).
"""

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
"""
Figure 1 (단일 패널):
성공률(PCR) vs VAEP 산점도. 색 = 평균 패스 출발 x좌표, 크기 = 압박 시 패스 횟수.
순위 차이가 큰 선수를 강조하고, 라벨에 PCR 순위와 VAEP 순위를 함께 표기한다.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

m = pd.read_parquet("data/pff_vaep/role_correlation_M.parquet")
n = len(m)

# 의도: 순위 열이 정수가 아니면(동순위 평균 순위 등) 아래 ordinal 표기가 틀어지므로 미리 검사
assert (m[["completion_rank", "vaep_rank"]] % 1 == 0).all().all(), "순위에 소수(동순위)가 있음"

# 의도: 문서(본문 폭 약 6.27인치)에 확대·축소 없이 그대로 넣도록 최종 크기로 그림.
# 이렇게 하면 7pt 글자가 인쇄 크기에서도 그대로 읽힘
fig, ax = plt.subplots(figsize=(6.27, 4.9))

# ---------------------------------------------------------------------------
# 강조 선수 선정 (방향별 상위 3명)
# ---------------------------------------------------------------------------
# 의도: 부호 있는 순위 차이. 양수 = VAEP 순위가 성공률 순위보다 높음(숫자가 작을수록 상위)
m["rank_gap"] = m["completion_rank"] - m["vaep_rank"]

# 의도: 한쪽 방향으로 쏠리지 않도록 방향별로 N명씩 선정
N_PER_SIDE = 3
vaep_higher = m.nlargest(N_PER_SIDE, "rank_gap")    # VAEP 순위가 훨씬 높은 선수
comp_higher = m.nsmallest(N_PER_SIDE, "rank_gap")   # 성공률 순위가 훨씬 높은 선수
HIGHLIGHT = list(dict.fromkeys([*vaep_higher["pff_nickname"], *comp_higher["pff_nickname"]]))

# 의도: 선정 결과, 표본 크기(노이즈 여부), 경계 동순위를 눈으로 점검
print(m[m["pff_nickname"].isin(HIGHLIGHT)]
      [["pff_nickname", "completion_rank", "vaep_rank", "rank_gap", "n_pressured"]]
      .sort_values("rank_gap"))
print("경계 다음 순위(동순위 확인):")
print(m.nlargest(N_PER_SIDE + 1, "rank_gap")[["pff_nickname", "rank_gap"]].tail(2))
print(m.nsmallest(N_PER_SIDE + 1, "rank_gap")[["pff_nickname", "rank_gap"]].tail(2))

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#222222"]
assert len(HIGHLIGHT) <= len(PALETTE), "강조 선수 수가 팔레트보다 많음"
HIGHLIGHT_COLORS = dict(zip(HIGHLIGHT, PALETTE))

# 의도: 순위를 1st, 2nd, 3rd, 4th…로 표기하는 함수(11~13위는 예외적으로 th)
def ordinal(k):
    k = int(k)
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"

# ---------------------------------------------------------------------------
# 산점도
# ---------------------------------------------------------------------------
x = m["completion_rate"] * 100
y = m["vaep_mean"]
c = m["role_mean_start_x"]
s = m["n_pressured"] * 1.5          # 의도: 작아진 그림에 맞춰 점 크기 배율을 2.5 → 1.5로 축소

# 의도: 색 중앙을 60(하프라인)으로 고정해 '자기 진영 쪽 / 상대 진영 쪽'이 색으로 나뉘게 함
norm = mcolors.TwoSlopeNorm(vmin=c.min(), vcenter=60, vmax=c.max())

# 의도: 전체 선수 점을 먼저 그림. 옅은 점이 흰 배경에 묻히지 않도록 테두리는 회색
scatter = ax.scatter(x, y, c=c, s=s, cmap="coolwarm", norm=norm, alpha=0.75,
                     edgecolor="0.3", linewidth=0.4, zorder=2)

# 의도: 강조 선수의 테두리를 점 바깥의 별도 고리로 분리해, 작은 점에서도 채움색이 보이게 함
hl = m[m["pff_nickname"].isin(HIGHLIGHT)]
hl_x = hl["completion_rate"] * 100
hl_s = hl["n_pressured"] * 1.5
ring_s = (np.sqrt(hl_s) + 5) ** 2   # 의도: 점 지름에 5pt를 더해 점 크기와 무관하게 일정한 틈 확보

# 의도: 1) 선수별 색의 속이 빈 고리
ax.scatter(hl_x, hl["vaep_mean"], s=ring_s, facecolors="none",
           edgecolors=[HIGHLIGHT_COLORS[nm] for nm in hl["pff_nickname"]],
           linewidths=1.5, zorder=3)
# 의도: 2) 고리 안에 원래 점을 다시 그려 채움색(출발 위치)만 보이게 함(norm 공유로 색 막대와 일치)
ax.scatter(hl_x, hl["vaep_mean"], c=hl["role_mean_start_x"], s=hl_s,
           cmap=scatter.cmap, norm=norm, alpha=0.95,
           edgecolor="white", linewidth=0.6, zorder=4)

# 의도: 라벨마다 (dx, dy, 정렬)을 직접 지정. dx의 부호로 정렬을 정하면
# 오른쪽 끝 선수의 라벨 방향을 따로 제어할 수 없어서 정렬(ha)도 함께 지정
DEFAULT_OFFSET = (-10, 14, "right")
LABEL_OFFSET = {
    "Kevin De Bruyne": (10, -2, "left"),    # 왼쪽 위 구석이라 오른쪽으로 보내 y축선에 걸리지 않게 함
    "Yunus Musah": (0, 26, "right"),        # 라이스 라벨과 분리하기 위해 점 위쪽의 빈 공간으로 이동
    "Declan Rice": (8, -26, "left"),        # 오른쪽 아래의 빈 공간으로 이동(무사 라벨과 분리)
    "Celso Borges": (14, -2, "left"),       # 왼쪽 위 거품 점과 겹치지 않도록 오른쪽 빈 공간으로 이동
}

# 의도: 라벨 = 이름 + 두 지표의 순위(두 줄). 색은 고리와 같게, 가는 지시선으로 어느 점인지 분명히 함
for _, row in hl.iterrows():
    name = row["pff_nickname"]
    dx, dy, ha = LABEL_OFFSET.get(name, DEFAULT_OFFSET)
    label = (f"{name}\n"
             f"PCR {ordinal(row['completion_rank'])} · VAEP {ordinal(row['vaep_rank'])}")
    ax.annotate(label, (row["completion_rate"] * 100, row["vaep_mean"]),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center",
                fontsize=7, fontweight="medium", linespacing=1.15,
                color=HIGHLIGHT_COLORS[name], zorder=5,
                arrowprops=dict(arrowstyle="-", color=HIGHLIGHT_COLORS[name],
                                lw=0.6, shrinkA=0, shrinkB=5))

# 의도: VAEP = 0 기준선(자명하므로 라벨 없음)과 선수별 성공률 평균선
ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.7, zorder=1)
ax.axvline(x.mean(), color="gray", linestyle="--", linewidth=0.6, alpha=0.7, zorder=1)
# 의도: 압박 패스 수로 가중하지 않은 선수별 단순 평균임을 "Player mean"으로 명시
ax.text(x.mean() + 0.4, 0.98, f"Player mean = {x.mean():.1f}%",
        transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7, color="gray")

# 의도: 오른쪽 끝(성공률 상위) 점과 라벨이 잘리지 않도록 x축 여백 확보
ax.margins(x=0.06)

# 의도: 본문과 같은 약어(PCR)를 축 제목에 넣어 초록 본문과 대응이 읽히게 함
ax.set_xlabel("Pass Completion Rate (PCR) Under Pressure (%)", fontsize=8)
ax.set_ylabel("Mean VAEP per Pressured Pass-Ending Episode", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=7)

# 의도: 그림 안 제목은 두지 않음(제목은 Word 캡션에 둠). 그림 안에도 제목을 넣고 싶으면
# 아래 TITLE에 문자열을 넣으면 됨
TITLE = None
if TITLE:
    ax.set_title(TITLE, fontsize=9, fontweight="bold")

# 의도: 화살표 대신 색 막대 제목에 방향을 직접 적어, 작은 그림에서도 글자와 겹치지 않게 함
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Mean Pass Starting X-Coordinate\n(higher = closer to opponent's goal)", fontsize=7)
cbar.ax.tick_params(labelsize=7)

# 의도: 점 크기 범례(압박 시 패스 횟수). 오른쪽 위의 빈 공간에 두되 평균선 텍스트와 겹치지 않게 아래로 내림
for n_ref in [20, 60, 120]:
    ax.scatter([], [], s=n_ref * 1.5, facecolor="gray", edgecolor="white",
               alpha=0.7, label=f"{n_ref} passes")
ax.legend(title="Pressured Passes", loc="upper right", bbox_to_anchor=(1.0, 0.92),
          fontsize=7, title_fontsize=7, labelspacing=1.0, borderpad=0.8, frameon=False)

plt.tight_layout()
# 의도: 제출용 고해상도 저장(300dpi, 잘림 방지). 경로는 상황에 맞게 변경
plt.savefig("figure1.png", dpi=300, bbox_inches="tight")
plt.show()
