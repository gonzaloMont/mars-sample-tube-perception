# IC 95 % del recall: Wilson frente a bootstrap por imagen (test v3)

| corrida | n | recall % | Wilson | bootstrap por imagen (4000 réplicas) |
|---|---|---|---|---|
| d2co_gt_sinumbral | 222 | 0.9 | 0.2–3.2 | 0.0–2.3 |
| d2co_gt_umbral | 222 | 0.5 | 0.1–2.5 | 0.0–1.4 |
| d2co_yolo_umbral | 222 | 0.5 | 0.1–2.5 | 0.0–1.4 |
| icp_depthgt_gt_B_defaults_nodo | 222 | 84.2 | 78.9–88.4 | 79.3–88.9 |
| icp_depthgt_yolo_B_defaults_nodo | 222 | 80.2 | 74.4–84.9 | 74.8–85.1 |
| icp_sgbmwls500_gt_B_defaults_nodo | 222 | 23.0 | 17.9–28.9 | 17.4–28.5 |
| icp_sgbmwls500_yolo_B_defaults_nodo | 222 | 23.0 | 17.9–28.9 | 17.6–28.5 |
