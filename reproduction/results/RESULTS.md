# YAP+ paper-result reproduction

Aggregate report assembled at current `yap+` commit: `7fde3ad07891852b4796863fd5a4733a7b259f0f`.
Each result JSON records the exact commit used for its calculation; unaffected figure packages may retain an earlier calculation commit.
Pre-summer checkpoint inspected: `6e80bc8` (2026-04-02).

| Figure | Coverage | Comparison to paper plot |
|---|---|---|
| 13 | W2W + D2W, all 300/300 legacy-sweep points (299/300 inside the 0.5–1.0 plot window) | MSE W2W `1.256e-04` vs `1.188e-04`; D2W `2.301e-05` vs `1.449e-05` |
| 15 | W2W, all 12 configurations | **PASS** with December-2025 inputs in current code; dedicated-package gating RMSE `0.0031`, max error `0.0110` |
| 16 | D2W, all 12 configurations + system yield | **NOT PASS** with all five quantities gated; best historical profile passes 8/12, RMSE `0.00922`, max error `0.04312` |
| 17 | W2W + D2W, all 4 layouts | RMSE W2W `0.0031`, D2W `0.0060` |
| 19 | W2W + D2W, all spacings and 20:1 | RMSE W2W `0.0754`, D2W `0.1516` |

Fig. 13/15/16/17 references are approximate readbacks from published raster plots. Fig. 19 instead uses exact numeric arrays recovered from the uploaded December-2025 MATLAB source. Fig. 13 uses the recovered 300-point parameter vectors and was regenerated after restoring sample-wise worst-corner overlay aggregation.

## Main findings

- Fig. 15: all twelve W2W configurations pass with the uploaded December-2025 parameters evaluated by current code. See `../fig15/results/summary.json`.
- Fig. 16: the paper-era sample-wise worst-corner aggregation is restored and all five plotted quantities are gated. The best recovered profile passes 8/12; cases 2/9/11/12 fail through amplified system-yield residuals. The May-2025 wafer/die radial scale is audited separately and does not recover the paper's die-area trend.
- Fig. 17: the explicit paper-era reconstruction passes all 8 layouts (RMSE 0.00475, max error 0.01288). It uses current code with sample-wise aggregation and opt-in D2W wafer scaling. The no-scaling raw-current plot is retained separately; provenance remains conditional because no recovered commit contains both the May scale and later layout boundaries.
- Fig. 19: latest code is 12/24 against the exact recovered bars. D2W at 1 cm^-2 and zero spacing is 0.35294 versus 0.6974. The recovered values satisfy Poisson density scaling, so geometry/critical-area/bitmap provenance—not density-label inconsistency—remains unresolved. The 20:1 columns are explicitly labeled compact-grid approximations.
