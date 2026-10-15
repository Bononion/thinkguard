# Paper inconsistencies

Checked against the ACL Anthology PDF (Findings of ACL 2025, pp. 13698–13713). Page numbers are 1-based PDF pages. "LG3" = LLaMA Guard 3 zero-shot.

## 1. ThinkGuard BeaverTails F1: 82.7 vs 82.3

- **Claim:** Table 1 reports 82.7; Fig. 3 / Table 4 report 82.3.
- **PDF shows:** Table 1 lists ThinkGuard BeaverTails F1 = 82.7 (p. 6). The Fig. 3 discussion says "At 15k training examples, ThinkGuard achieves an F1 score of 82.3% and a macro F1 of 76.4%" (p. 8), and Fig. 3(a)'s 15k point sits at about 82.3 (p. 7). Table 4's "15k Cla-Exp SFT" row, which Appendix A says "corresponds to ThinkGuard", reports accuracy 81.2 / F1 82.3 / macro F1 76.4 (p. 13). Table 3 reports accuracy 81.6 / macro F1 77.5 for the same model (p. 7).
- **Verdict:** confirmed. The paper reports two sets of ThinkGuard BeaverTails numbers: Table 1/3 (F1 82.7, acc 81.6, macro F1 77.5) and Fig. 3/Table 4 (F1 82.3, acc 81.2, macro F1 76.4).
- **Pages:** 6, 7, 8, 13

## 2. ToxicChat gain over LLaMA Guard 3: +14.1% vs +20.1 points

- **Claim:** the text says +14.1%; Table 1 shows +20.1 points.
- **PDF shows:** Section 4.2 says ThinkGuard "enhances F1 by 14.1% on Toxic Chat and 6.1% on WildGuardMix" (p. 7); the Conclusion repeats it (p. 8). Table 1 ToxicChat F1: LG3 43.4 → ThinkGuard 63.5 = +20.1 points (+46.3% relative), so 14.1 matches neither reading (p. 6). The WildGuardMix figure in the same sentence is also off: 72.6 → 78.6 = +6.0 points, not 6.1.
- **Verdict:** confirmed
- **Pages:** 6, 7, 8

## 3. Table 3 gains: +16.1% / +27.0% vs +16.4 / +28.1 points

- **Claim:** the text says accuracy +16.1% and macro F1 +27.0%; Table 3 shows +16.4 / +28.1 points.
- **PDF shows:** the abstract (p. 1), introduction (p. 2) and Section 4.2 (p. 8) all say ThinkGuard "improves accuracy by 16.1% and macro F1 by 27.0%" over LG3. Table 3 (p. 7): accuracy LG3 65.2 → ThinkGuard 81.6 = +16.4 points; macro F1 49.4 → 77.5 = +28.1 points. Against Table 4's 15k Cla-Exp row (p. 13) the gains are 81.2 − 65.2 = +16.0 and 76.4 − 49.4 = +27.0, and the text's "+2.1% accuracy over label-only" (p. 8) is exactly 81.2 − 79.1. The text gains appear to come from the Table 4 numbers, not Table 3.
- **Verdict:** confirmed
- **Pages:** 1, 2, 7, 8, 13

## 4. Appendix A "consistently outperforms" vs lower F1 at 9k

- **Claim:** Appendix A says critique training always outperforms, but F1 is lower at 9k examples.
- **PDF shows:** Appendix A compares two critique formats, not critique vs label-only: "Cla-Exp consistently outperforms Exp-Cla, particularly at 15k" (p. 13). Table 4 at 9k: Cla-Exp F1 69.9 < Exp-Cla F1 71.0, while Cla-Exp leads on accuracy (69.9 vs 67.2) and macro F1 (62.6 vs 60.9) (p. 13). Separately, in Fig. 3(a) (critique vs label-only) ThinkGuard's F1 is above label-only at 9k by eye but below it at 12k and 15k (83.7 vs 82.3 at 15k), although the p. 8 text claims "a clear advantage over the label-only model as dataset size increases".
- **Verdict:** confirmed. The "consistently outperforms" claim fails on F1 at 9k, but the comparison is Cla-Exp vs Exp-Cla (both critique formats).
- **Pages:** 7, 8, 13
