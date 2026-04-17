"""CLI wrapper for the offline fusion-weight sweep.

Delegates to :mod:`research.triangular_maxsim.sweep_fusion_weights` so the
full sweep harness stays in one place. Re-exposed under ``scripts/`` so it
can be invoked via the ``latence-trace-sweep`` console entry point.
"""

from research.triangular_maxsim.sweep_fusion_weights import main

if __name__ == "__main__":
    main()
