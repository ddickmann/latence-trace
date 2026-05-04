# Runtime Verification Definition Of Done

Overall pass: `True`

## Gates

- No-regression: `True`
- Six-class coverage + thresholds: `True`
- Stable runtime record schema: `True`
- Latency budget: `True` (mean=480.57 ms, p95=1692.38 ms)
- Shadow/canary rates: `True` (false_allow=0.0106, false_block=0.0)
- Rollback-safe defaults: `True`

## Notes

- Classes where the learned calibrator would regress v1 are configured as `v1_passthrough` with learned weight `0.0`.
- The learned channel is therefore safe to keep disabled by default and only enabled per class when the optimizer proves no regression.
