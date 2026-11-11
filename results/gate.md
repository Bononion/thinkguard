PASS

R under our harness vs ThinkGuard in the paper (protocol.md, Harness gate). Ours is rounded to one decimal;
a gated row passes if |Δ| <= 2.0. Written by `python -m thinkguard.report gate` from scores.csv.

| benchmark | metric | paper | ours | Δ | pass (- = not gated) | invalid % |
|---|---|---|---|---|---|---|
| beavertails_30k_test | F1 (reading 4) | 82.7 | 81.8 | -0.9 | yes | 0.0 |
| beavertails_30k_test | Accuracy (reading 4) | 81.6 | 82.9 | +1.3 | yes | 0.0 |
| beavertails_30k_test | AUPRC | 91.7 | 95.6 | +3.9 | - | 0.0 |
| openai_moderation | F1 | 77.3 | 73.5 | -3.8 | - | 0.1 |
| openai_moderation | AUPRC | 80.7 | 87.2 | +6.5 | - | 0.1 |
| toxicchat_0124_test | F1 | 63.5 | 65.0 | +1.5 | yes | 0.1 |
| toxicchat_0124_test | AUPRC | 65.3 | 75.5 | +10.2 | - | 0.1 |
| wildguardtest | F1 | 78.6 | 78.5 | -0.1 | yes | 0.1 |
| wildguardtest | AUPRC | 80.3 | 85.5 | +5.2 | - | 0.1 |

BeaverTails under the four readings of reports/paper_protocol.md (reading 4 chosen):

| reading | F1 | accuracy |
|---|---|---|
| 1 | 62.3 | 64.9 |
| 2 | 50.5 | 64.9 |
| 3 | 84.5 | 82.9 |
| 4 | 81.8 | 82.9 |

| benchmark | run_id | config_hash |
|---|---|---|
| beavertails_30k_test | 20261110-0302-eval-c8a4a2c8 | c8a4a2c8941cf1809896f5551a51a49bfe1c20483cc1eac036d9254e9790d44a |
| openai_moderation | 20261110-0413-eval-4e7d16a5 | 4e7d16a508017bbf927b9577b8abceb6b7ba9ba87c93638181cbe1324e49a75e |
| toxicchat_0124_test | 20261110-0335-eval-23ae6493 | 23ae64938755ccf8b40782c5f04703ca868cb5ed0d125086ac93fde4cba827ca |
| wildguardtest | 20261110-0428-eval-e7753281 | e7753281ffd92ebf4b30bc142135508c15ef120f1ec716c7311db27a644f8c53 |
