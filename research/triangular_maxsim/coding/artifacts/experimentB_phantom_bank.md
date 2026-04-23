# Experiment B (isolated): phantom-response bank

> Swaps the hand-edited `wrong` tier for genuinely-phantom responses that use fabricated imports (banana-ml, photon-labs, skylab-metrics, crystal-lattice, cheetah-profiler, mochi-router, prismatic-bench, cactus-codegen, basilisk-schema, skyjournal). Same 8 base scenarios, same cached context, same scorer. No production code modified.

## Per-case scores

### chunker = `sentence_packed`

| base | rc(correct) | rc(wrong) | rc(ambig avg) | rc(phantom) | correct_min | phantom_min | correct_p10 | phantom_p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_01_pooler_helper | 0.9795 | 0.9738 | 0.9814 | 0.9523 | 0.9109 | 0.8721 | 0.9313 | 0.8982 |
| base_02_nli_source_init | 0.9824 | 0.9788 | 0.9833 | 0.9630 | 0.9107 | 0.8692 | 0.9650 | 0.9005 |
| base_03_benchmark_measure | 0.9817 | 0.9742 | n/a | 0.9519 | 0.8877 | 0.8725 | 0.9635 | 0.8959 |
| base_04_structured_evidence_detector | 0.9755 | 0.9707 | 0.9744 | 0.9645 | 0.8992 | 0.8856 | 0.9408 | 0.9139 |
| base_05_maxsim_timing_block | 0.9857 | 0.9810 | n/a | 0.9620 | 0.9173 | 0.8644 | 0.9655 | 0.8974 |
| base_06_benchmark_invocation | 0.9799 | 0.9766 | n/a | 0.9585 | 0.8924 | 0.8823 | 0.9399 | 0.8999 |
| base_07_wire_marker_module | 0.9724 | 0.9660 | n/a | 0.9659 | 0.8686 | 0.8850 | 0.9256 | 0.9038 |
| base_08_cleanup_audit_entry | 0.9615 | 0.9572 | 0.9557 | 0.9471 | 0.8427 | 0.8575 | 0.9041 | 0.8848 |

### chunker = `colgrep`

| base | rc(correct) | rc(wrong) | rc(ambig avg) | rc(phantom) | correct_min | phantom_min | correct_p10 | phantom_p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_01_pooler_helper | 0.9791 | 0.9742 | 0.9809 | 0.9521 | 0.9116 | 0.8701 | 0.9333 | 0.8978 |
| base_02_nli_source_init | 0.9807 | 0.9786 | 0.9820 | 0.9620 | 0.9103 | 0.8699 | 0.9645 | 0.8979 |
| base_03_benchmark_measure | 0.9831 | 0.9759 | n/a | 0.9524 | 0.9125 | 0.8781 | 0.9631 | 0.8951 |
| base_04_structured_evidence_detector | 0.9747 | 0.9689 | 0.9735 | 0.9615 | 0.8968 | 0.8826 | 0.9357 | 0.9107 |
| base_05_maxsim_timing_block | 0.9848 | 0.9801 | n/a | 0.9576 | 0.9183 | 0.8643 | 0.9616 | 0.8894 |
| base_06_benchmark_invocation | 0.9792 | 0.9757 | n/a | 0.9586 | 0.8924 | 0.8820 | 0.9407 | 0.9003 |
| base_07_wire_marker_module | 0.9605 | 0.9559 | n/a | 0.9571 | 0.8574 | 0.8584 | 0.9073 | 0.8913 |
| base_08_cleanup_audit_entry | 0.9603 | 0.9563 | 0.9551 | 0.9473 | 0.8451 | 0.8577 | 0.9061 | 0.8864 |

## AUROC — correct vs phantom vs original wrong

| chunker | aggregate | contrast | AUROC | separation | pos mean | neg mean | pairs |
|---|---|---|---:|---:|---:|---:|---:|
| colgrep | consensus_hardened | correct vs ambiguous (baseline) | 0.5000 | 0.00080 | 0.97355 | 0.97275 | 4 |
| colgrep | consensus_hardened | correct vs phantom (exp B) | 0.9219 | 0.01908 | 0.97389 | 0.95481 | 8 |
| colgrep | consensus_hardened | correct vs wrong (baseline) | 0.6875 | 0.00453 | 0.97389 | 0.96936 | 8 |
| colgrep | echo_mean | correct vs ambiguous (baseline) | 0.5625 | 0.00100 | 0.95771 | 0.95671 | 4 |
| colgrep | echo_mean | correct vs phantom (exp B) | 0.8906 | 0.01652 | 0.95083 | 0.93432 | 8 |
| colgrep | echo_mean | correct vs wrong (baseline) | 0.6875 | 0.00436 | 0.95083 | 0.94647 | 8 |
| colgrep | groundedness_v2 | correct vs ambiguous (baseline) | 0.4375 | -0.01324 | 0.88958 | 0.90282 | 4 |
| colgrep | groundedness_v2 | correct vs phantom (exp B) | 0.6406 | 0.02794 | 0.91188 | 0.88394 | 8 |
| colgrep | groundedness_v2 | correct vs wrong (baseline) | 0.7031 | 0.03868 | 0.91188 | 0.87320 | 8 |
| colgrep | literal_guarded | correct vs ambiguous (baseline) | 0.4062 | -0.04840 | 0.67925 | 0.72764 | 4 |
| colgrep | literal_guarded | correct vs phantom (exp B) | 0.6406 | 0.04969 | 0.75329 | 0.70361 | 8 |
| colgrep | literal_guarded | correct vs wrong (baseline) | 0.7031 | 0.12386 | 0.75329 | 0.62944 | 8 |
| colgrep | max_top_evidence_score | correct vs ambiguous (baseline) | 0.6875 | 0.00059 | 0.99815 | 0.99756 | 4 |
| colgrep | max_top_evidence_score | correct vs phantom (exp B) | 0.4062 | 0.00024 | 0.99852 | 0.99828 | 8 |
| colgrep | max_top_evidence_score | correct vs wrong (baseline) | 0.6250 | 0.00201 | 0.99852 | 0.99651 | 8 |
| colgrep | per_token_mean_weakest_5 | correct vs phantom (exp B) | 0.7500 | 0.02790 | 0.90587 | 0.87797 | 8 |
| colgrep | per_token_min | correct vs phantom (exp B) | 0.7500 | 0.02267 | 0.89304 | 0.87038 | 8 |
| colgrep | per_token_p10 | correct vs phantom (exp B) | 0.9688 | 0.04291 | 0.93902 | 0.89611 | 8 |
| colgrep | per_token_p5 | correct vs phantom (exp B) | 0.8750 | 0.03520 | 0.92246 | 0.88726 | 8 |
| colgrep | reverse_context (aggregate) | correct vs ambiguous (baseline) | 0.4375 | 0.00082 | 0.97371 | 0.97289 | 4 |
| colgrep | reverse_context (aggregate) | correct vs phantom (exp B) | 0.9375 | 0.01924 | 0.97531 | 0.95607 | 8 |
| colgrep | reverse_context (aggregate) | correct vs wrong (baseline) | 0.7188 | 0.00461 | 0.97531 | 0.97070 | 8 |
| colgrep | reverse_query_context | correct vs ambiguous (baseline) | 0.5000 | 0.00131 | 0.97743 | 0.97612 | 4 |
| colgrep | reverse_query_context | correct vs phantom (exp B) | 0.9844 | 0.02085 | 0.97956 | 0.95871 | 8 |
| colgrep | reverse_query_context | correct vs wrong (baseline) | 0.7031 | 0.00482 | 0.97956 | 0.97474 | 8 |
| colgrep | triangular | correct vs ambiguous (baseline) | 0.5000 | 0.00056 | 0.96615 | 0.96559 | 4 |
| colgrep | triangular | correct vs phantom (exp B) | 0.9531 | 0.01434 | 0.96298 | 0.94864 | 8 |
| colgrep | triangular | correct vs wrong (baseline) | 0.7031 | 0.00338 | 0.96298 | 0.95960 | 8 |
| sentence_packed | consensus_hardened | correct vs ambiguous (baseline) | 0.5000 | 0.00101 | 0.97468 | 0.97368 | 4 |
| sentence_packed | consensus_hardened | correct vs phantom (exp B) | 0.9375 | 0.01912 | 0.97717 | 0.95805 | 8 |
| sentence_packed | consensus_hardened | correct vs wrong (baseline) | 0.7344 | 0.00497 | 0.97717 | 0.97220 | 8 |
| sentence_packed | echo_mean | correct vs ambiguous (baseline) | 0.5625 | 0.00100 | 0.95771 | 0.95671 | 4 |
| sentence_packed | echo_mean | correct vs phantom (exp B) | 0.8906 | 0.01652 | 0.95083 | 0.93432 | 8 |
| sentence_packed | echo_mean | correct vs wrong (baseline) | 0.6875 | 0.00436 | 0.95083 | 0.94647 | 8 |
| sentence_packed | groundedness_v2 | correct vs ambiguous (baseline) | 0.4375 | -0.01307 | 0.89053 | 0.90360 | 4 |
| sentence_packed | groundedness_v2 | correct vs phantom (exp B) | 0.6562 | 0.02768 | 0.91359 | 0.88592 | 8 |
| sentence_packed | groundedness_v2 | correct vs wrong (baseline) | 0.7031 | 0.03907 | 0.91359 | 0.87452 | 8 |
| sentence_packed | literal_guarded | correct vs ambiguous (baseline) | 0.4062 | -0.04833 | 0.67998 | 0.72831 | 4 |
| sentence_packed | literal_guarded | correct vs phantom (exp B) | 0.6406 | 0.04894 | 0.75424 | 0.70531 | 8 |
| sentence_packed | literal_guarded | correct vs wrong (baseline) | 0.7031 | 0.12415 | 0.75424 | 0.63009 | 8 |
| sentence_packed | max_top_evidence_score | correct vs ambiguous (baseline) | 0.6250 | 0.00051 | 0.99820 | 0.99769 | 4 |
| sentence_packed | max_top_evidence_score | correct vs phantom (exp B) | 0.3281 | 0.00035 | 0.99861 | 0.99826 | 8 |
| sentence_packed | max_top_evidence_score | correct vs wrong (baseline) | 0.6250 | 0.00219 | 0.99861 | 0.99642 | 8 |
| sentence_packed | per_token_mean_weakest_5 | correct vs phantom (exp B) | 0.8594 | 0.02859 | 0.90876 | 0.88017 | 8 |
| sentence_packed | per_token_min | correct vs phantom (exp B) | 0.7812 | 0.01760 | 0.89119 | 0.87359 | 8 |
| sentence_packed | per_token_p10 | correct vs phantom (exp B) | 0.9844 | 0.04268 | 0.94195 | 0.89927 | 8 |
| sentence_packed | per_token_p5 | correct vs phantom (exp B) | 0.9844 | 0.03818 | 0.92630 | 0.88812 | 8 |
| sentence_packed | reverse_context (aggregate) | correct vs ambiguous (baseline) | 0.5000 | 0.00104 | 0.97474 | 0.97371 | 4 |
| sentence_packed | reverse_context (aggregate) | correct vs phantom (exp B) | 0.9375 | 0.01917 | 0.97733 | 0.95816 | 8 |
| sentence_packed | reverse_context (aggregate) | correct vs wrong (baseline) | 0.7344 | 0.00503 | 0.97733 | 0.97230 | 8 |
| sentence_packed | reverse_query_context | correct vs ambiguous (baseline) | 0.5000 | 0.00134 | 0.97785 | 0.97651 | 4 |
| sentence_packed | reverse_query_context | correct vs phantom (exp B) | 0.9375 | 0.02012 | 0.98065 | 0.96052 | 8 |
| sentence_packed | reverse_query_context | correct vs wrong (baseline) | 0.7500 | 0.00479 | 0.98065 | 0.97586 | 8 |
| sentence_packed | triangular | correct vs ambiguous (baseline) | 0.5000 | 0.00059 | 0.96623 | 0.96565 | 4 |
| sentence_packed | triangular | correct vs phantom (exp B) | 0.9688 | 0.01404 | 0.96360 | 0.94956 | 8 |
| sentence_packed | triangular | correct vs wrong (baseline) | 0.7031 | 0.00355 | 0.96360 | 0.96005 | 8 |

## Headline

- **sentence_packed**: `reverse_context` AUROC correct-vs-wrong = 0.7344 → correct-vs-phantom = 0.9375 (separation 0.00503 → 0.01917)
- **colgrep**: `reverse_context` AUROC correct-vs-wrong = 0.7188 → correct-vs-phantom = 0.9375 (separation 0.00461 → 0.01924)
