# M-HySMap v1.3 Optional Add-on

v1.3 keeps the v1.0/v1.2 final algorithm intact and adds one optional, safe portfolio arm:

- `mhysmap_v13_congestion_place_polish`: starts from the strong Activity+QAP / hypergraph-refined seed and runs a congestion-biased multicast placement polish.
- `mhysmap_v13_final_best`: safe selector between `mhysmap_v10_portfolio_best` and the v1.3 polish under the original published objective.

The add-on is deliberately conservative: if the congestion polish does not improve the original objective, `mhysmap_v13_final_best` simply falls back to the v1.0 portfolio result.

## PowerShell global run style

```powershell
$env:PYTHONPATH="."; py -3.12 .\run_heavy_v12.py --config .\configs\smoke_v12.json --out-dir .\results\smoke_v13 --timeout-per-run 300
$env:PYTHONPATH="."; py -3.12 .\plot_heavy_v12.py --results-dir .\results\smoke_v13
```

For a bigger run:

```powershell
$env:PYTHONPATH="."; py -3.12 .\run_heavy_v12.py --config .\configs\paperlite_10seed_v12.json --out-dir .\results\paperlite_v13 --timeout-per-run 1800
$env:PYTHONPATH="."; py -3.12 .\plot_heavy_v12.py --results-dir .\results\paperlite_v13
```

## How to interpret

Use `mhysmap_v10_portfolio_best` as the locked main-paper method unless `mhysmap_v13_final_best` shows consistent improvement on the heavy run. If v13 improves only a few stress cases, present it as an optional congestion-polish ablation or future direction, not as the main algorithm.
