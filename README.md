# Pass Completion Rate vs. VAEP Under Pressure: Midfielders at the 2022 FIFA World Cup

Code for the research abstract submitted to the SSAC27 Research Paper Competition.

## What this repository does
1. Trains an XGBoost-based VAEP model on StatsBomb Open Data (463 men's matches; 2022 World Cup excluded).
2. Applies the model, without retraining, to PFF FC 2022 World Cup event data.
3. Reconstructs possession episodes, selects midfielders, and compares pass completion rate under pressure (PCR; StatsBomb) with episode-based VAEP (PFF FC).
4. Reports Spearman's rho, Kendall's tau and Cohen's kappa, and the correlation of each metric with the mean pass starting x-coordinate.

## Data (not included; links only)
| Source | How to obtain | Version / date used |
|---|---|---|
| PFF FC 2022 FIFA World Cup dataset (events, rosters, players) | https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset (free access by request) | PFF FC Event Data Specification v2.5 (June 2025) |
| StatsBomb Open Data (events, lineups, matches) | https://github.com/statsbomb/open-data | Retrieved 2026-09-08 |

Expected local layout (relative to the repository root; these folders are not tracked):
```
statsbomb/data/                       # clone of statsbomb/open-data (data/ folder)
FIFA World Cup 2022/Event Data/*.json # PFF FC event files (one per match)
FIFA World Cup 2022/Rosters/          # PFF FC rosters
FIFA World Cup 2022/players.csv       # PFF FC player table
```
Data credit: Data provided by StatsBomb; data provided by PFF FC. Please follow each provider's terms of use.

## Environment
Python 3.11. Install with `pip install -r requirements.txt`.

## How to reproduce (run from the repository root, in this order)
| Step | Script | Main inputs | Main outputs |
|---|---|---|---|
| 1 | `01_build_vaep_model_statsbomb.py` | `statsbomb/data/` | `models/vaep_scores.json`, `models/vaep_concedes.json`, `data/features/`, `data/vaep/` |
| 2 | `02_apply_vaep_to_pff.py` | PFF event files, `models/` | `data/pff_actions_ready/`, `data/pff_features/`, `data/pff_vaep/pff_vaep_final.parquet` |
| 3 | `03_prepare_pff_tables.py` | PFF events and rosters, `pff_vaep_final.parquet` | `data/pff_vaep/minutes_total.parquet`, `data/pff_vaep/episodes.parquet` |
| 4 | `04_compare_pcr_vs_vaep.py` | StatsBomb events/lineups, `episodes.parquet`, `minutes_total.parquet`, `players.csv` | `data/pff_vaep/final_comparison_M_rawmean.parquet` (prints rho, tau, kappa) |
| 5 | `05_role_correlation.py` | `final_comparison_M_rawmean.parquet`, `data/pff_actions_ready/` | `data/pff_vaep/role_correlation_M.parquet` |
| 6 | `06_make_figure1.py` | `role_correlation_M.parquet` | `figure1_B_only.png` |

Sanity check (n = 45 players): Spearman's rho = -0.111 (p = 0.470); Kendall's tau = -0.075 (p = 0.469); Cohen's kappa = 0.022 (51.1% agreement, top/bottom half); mean pass starting x-coordinate vs. VAEP rho = 0.572 (p < 0.001) and vs. PCR rho = -0.104 (p = 0.497).

## Processing choices
- **Training corpus:** men's competitions, matches from 2018-01-01, excluding the 2022 World Cup (competition 43, season 106) and three competitions with very small or very large sample shares (Indian Super League, Major League Soccer, Champions League). Result: 463 matches.
- **Pressure (PFF FC):** `initialTouch_initialPressureType` at the first row of an episode (ball reception). `P` (Player Pressured) = pressured, `N` (No Pressure) = not pressured. `A` (Attempted) and `L` (Passing Lane) are excluded.
- **Episodes:** continuous possession by one player; episode VAEP is the sum of the VAEP values of its actions. Only episodes ending in a pass are used (`final_action_type == "pass"`; crosses are a separate action type and are not included).
- **Coordinates:** PFF FC coordinates are standardized to the StatsBomb 120 x 80 system (mplsoccer `Standardizer`), and coordinates are flipped when `stadiumMetadata_teamAttackingDirection == "L"` so that x increases in the attacking direction.
- **Sample (n = 45):** midfielders in both datasets; at least 90 minutes played; at least 20 pressured pass-ending episodes in at least 3 different matches (PFF FC); at least 20 pressured passes (StatsBomb).
- **Player matching across datasets:** fuzzy name matching (rapidfuzz, token sort ratio, minimum score 60), with 4 manually restored matches.
- **Reported results:** the raw (untrimmed) mean of episode VAEP per player (`final_comparison_M_rawmean`). A 5%-trimmed-mean variant was also computed during development; it is not part of this repository.
- **PCR under pressure (StatsBomb):** pass events tagged `under_pressure`; a pass is successful if it has no `outcome` field.

## Known limitations
- PCR (StatsBomb `under_pressure` tag, which includes crosses as passes) and VAEP (PFF FC initial-touch pressure, crosses excluded) come from different providers with different pressure definitions.
- Player-name matching across the two datasets relies on fuzzy matching plus 4 manual corrections.

## License
Code: MIT (see `LICENSE`). Data are subject to the providers' terms and are not redistributed here.
