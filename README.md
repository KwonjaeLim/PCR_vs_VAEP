# PCR_vs_VAEP
# Pass Completion Rate vs. VAEP Under Pressure: Midfielders at the 2022 FIFA World Cup

Code repository for the SSAC27 research abstract. Work in progress: code is being cleaned and will be updated.

## Data (not included; links only)
- PFF FC 2022 World Cup dataset: [https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data]. Access by request. 
- StatsBomb Open Data: https://github.com/statsbomb/open-data.

## Method summary
- VAEP model (XGBoost) trained on StatsBomb Open Data, applied without retraining to PFF FC data.
- Pressure: PFF FC initialPressureType [P].
- Sample: midfielders in both datasets with >= 90 min, >= 20 passes under pressure, >= 3 matches (n = 45).
- Statistics: Spearman's rho, Cohen's kappa, alpha = .05, Python 3.11.

## Reproduction
Scripts will be listed here in run order after cleanup.
