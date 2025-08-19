# Task 3: 10-Month LSTM Forecast Analysis

## Trend Analysis
- Direction: Downward
- Total change over horizon: -20.64% (start $316.58 → end $251.23)
- Approx. annualized change (based on horizon length): -24.23%

## Confidence Intervals and Uncertainty
- Initial 95% interval width: $43.94 (13.88% of price)
- Final 95% interval width:   $136.28 (54.24% of price)
- Average interval width:     $131.09 (49.21% of average price)
- Interval width change over horizon: 210.14%
- Interval width vs time correlation: 0.51 (interval increases over time)

### Interpretation of Uncertainty
- Confidence intervals typically widen with horizon in data-driven forecasts; a positive correlation indicates growing uncertainty farther into the future.
- Wider intervals imply reduced reliability in long-horizon point estimates; decisions should account for the range, not just the mean forecast.

## Market Opportunities and Risks
- Risk: Downward/flat trend suggests caution for new long exposure; consider hedging or defensive positioning.
- Risk: Uncertainty increases over the 10-month horizon; long-dated decisions face higher forecast dispersion.
- Tactically, monitoring for regime shifts, earnings surprises, and macro catalysts is essential; these can invalidate recent dynamics captured by the LSTM.

## Notes
- Intervals are derived via Monte Carlo dropout (approximate predictive uncertainty).
- Results reflect learned patterns and may not capture exogenous shocks.
