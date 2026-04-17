"""CLI wrapper for the threshold calibration harness.

Delegates to :mod:`research.triangular_maxsim.calibrate_thresholds` so the
full evaluation harness stays in one place. Re-exposed under ``scripts/`` so
it can be invoked via the ``latence-trace-calibrate`` console entry point.
"""

from research.triangular_maxsim.calibrate_thresholds import main

if __name__ == "__main__":
    main()
