# M-HySMap v1.1 Paper Tables and Ablation Pack

v1.1 packages the v1.0 stable evidence into paper-ready tables, ablations, and figures. It does **not** claim final camera-ready evidence yet; it is the draft-evidence pack for ASP-DAC writing.

## Main result table

| config         | mesh   | method                  |   edge_cut |   weighted_cut |   routed_hops |   max_link |   objective |   runs |
|:---------------|:-------|:------------------------|-----------:|---------------:|--------------:|-----------:|------------:|-------:|
| s=0.0015, k=16 | 4x4    | Edge-cut + QAP          |      164.5 |         1063.9 |        1791.8 |      131.4 |      1816.1 |      2 |
| s=0.0015, k=16 | 4x4    | Activity-edge + QAP     |      162.5 |          917.3 |        1193.8 |      106.4 |      1212.1 |      2 |
| s=0.0015, k=16 | 4x4    | QAP-seeded Hyper-refine |      163.5 |          904.8 |        1042.9 |      106.5 |      1061.3 |      2 |
| s=0.0015, k=16 | 4x4    | M-HySMap portfolio      |      170.5 |          923.8 |         952.6 |       90.8 |       967.9 |      2 |
| s=0.0020, k=16 | 4x4    | Edge-cut + QAP          |      327.5 |         5229.0 |        7688.0 |      397.0 |      7795.6 |      2 |
| s=0.0020, k=16 | 4x4    | Activity-edge + QAP     |      320.0 |         4660.8 |        6473.3 |      394.0 |      6584.0 |      2 |
| s=0.0020, k=16 | 4x4    | QAP-seeded Hyper-refine |      335.5 |         5014.0 |        5326.6 |      359.5 |      5427.5 |      2 |
| s=0.0020, k=16 | 4x4    | M-HySMap portfolio      |      338.0 |         5023.4 |        5220.2 |      318.9 |      5315.2 |      2 |
| s=0.0020, k=25 | 5x5    | Edge-cut + QAP          |      329.5 |         5272.7 |        9079.5 |      309.4 |      9177.1 |      2 |
| s=0.0020, k=25 | 5x5    | Activity-edge + QAP     |      326.5 |         4704.6 |        7233.2 |      325.2 |      7326.6 |      2 |
| s=0.0020, k=25 | 5x5    | QAP-seeded Hyper-refine |      344.0 |         5011.0 |        6392.7 |      309.5 |      6476.1 |      2 |
| s=0.0020, k=25 | 5x5    | M-HySMap portfolio      |      344.0 |         5011.0 |        6392.7 |      309.5 |      6476.1 |      2 |
| s=0.0020, k=36 | 6x6    | Edge-cut + QAP          |      354.5 |         5704.5 |       13016.8 |      371.9 |     13111.9 |      2 |
| s=0.0020, k=36 | 6x6    | Activity-edge + QAP     |      353.5 |         5086.2 |       10157.1 |      325.2 |     10245.9 |      2 |
| s=0.0020, k=36 | 6x6    | QAP-seeded Hyper-refine |      363.5 |         5342.5 |        8963.0 |      290.8 |      9037.8 |      2 |
| s=0.0020, k=36 | 6x6    | M-HySMap portfolio      |      363.5 |         5342.5 |        8963.0 |      290.8 |      9037.8 |      2 |


## Improvement table

| config         | mesh   |   hop_impr_vs_edge_qap_pct |   hop_impr_vs_activity_qap_pct |   obj_impr_vs_activity_qap_pct |   maxlink_impr_vs_activity_qap_pct |   incremental_speedup_x |   portfolio_joint_selection_rate |   portfolio_pairwise_seed_rate |
|:---------------|:-------|---------------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|------------------------:|---------------------------------:|-------------------------------:|
| s=0.0015, k=16 | 4x4    |                      46.84 |                          20.21 |                          20.15 |                              14.64 |                    6.34 |                             1.00 |                           0.00 |
| s=0.0020, k=16 | 4x4    |                      32.10 |                          19.36 |                          19.27 |                              19.06 |                    8.05 |                             0.50 |                           0.50 |
| s=0.0020, k=25 | 5x5    |                      29.59 |                          11.62 |                          11.61 |                               4.81 |                    7.50 |                             0.00 |                           1.00 |
| s=0.0020, k=36 | 6x6    |                      31.14 |                          11.76 |                          11.79 |                              10.58 |                    6.49 |                             0.00 |                           1.00 |


## Ablation table

| config         |   edge_qap_hops |   activity_qap_hops |   qap_seeded_hyper_hops |   joint_default_hops |   portfolio_hops |   activity_gain_over_edge_pct |   hyper_refine_gain_over_activity_pct |   portfolio_gain_over_activity_pct |   portfolio_gain_over_edge_pct |   incremental_speedup_x |
|:---------------|----------------:|--------------------:|------------------------:|---------------------:|-----------------:|------------------------------:|--------------------------------------:|-----------------------------------:|-------------------------------:|------------------------:|
| s=0.0015, k=16 |         1791.80 |             1193.85 |                 1042.88 |               974.37 |           952.57 |                         33.37 |                                 12.65 |                              20.21 |                          46.84 |                    6.34 |
| s=0.0020, k=16 |         7688.02 |             6473.33 |                 5326.62 |              5457.30 |          5220.15 |                         15.80 |                                 17.71 |                              19.36 |                          32.10 |                    8.05 |
| s=0.0020, k=25 |         9079.47 |             7233.23 |                 6392.67 |              7023.23 |          6392.67 |                         20.33 |                                 11.62 |                              11.62 |                          29.59 |                    7.50 |
| s=0.0020, k=36 |        13016.80 |            10157.08 |                 8962.98 |              9103.10 |          8962.98 |                         21.97 |                                 11.76 |                              11.76 |                          31.14 |                    6.49 |


## Runtime / incremental-update table

| config         |   full_recompute_partition_s |   incremental_partition_s |   speedup_x |   avg_affected_sources |   max_validation_error |
|:---------------|-----------------------------:|--------------------------:|------------:|-----------------------:|-----------------------:|
| s=0.0015, k=16 |                        0.713 |                     0.112 |       6.341 |                  2.612 |                  0.000 |
| s=0.0020, k=16 |                        1.930 |                     0.240 |       8.045 |                  2.523 |                  0.000 |
| s=0.0020, k=25 |                        1.882 |                     0.251 |       7.504 |                  2.605 |                  0.000 |
| s=0.0020, k=36 |                        1.976 |                     0.305 |       6.486 |                  2.690 |                  0.000 |


## Draft interpretation

- The final portfolio beats Edge-cut + QAP on all configs in routed multicast hops.

- The final portfolio also beats Activity-edge + QAP on all configs in this stable evidence set, including the larger 6x6/k=36 case.

- The QAP-seeded hypergraph path is the key robustness fix: the method starts from the strongest graph placement and then applies multicast-aware moves.

- Incremental gain updates give roughly 6-8x partition-refinement speedup while keeping validation error near numerical noise.


## Final run checklist before submission

1. Expand seeds to at least 10 for k=16 and k=25.

2. Expand k=36 to at least 5 seeds, ideally 10 if runtime allows.

3. Add one synthetic fanout/congestion stress benchmark.

4. Freeze alpha/beta/gamma and report sensitivity in appendix/supplement if space allows.

5. Add link-load heatmap examples for one representative seed.
