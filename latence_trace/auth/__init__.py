"""Commercial license enforcement (L3).

The latence-trace runtime is a commercially licensed product. Production
deployments verify a customer-issued JWT license token signed by the
latence.ai signing key. The token carries the customer identifier,
entitlement tier, feature toggles, max QPS, and a hard ``exp`` claim.

This package exposes:

- :class:`LicenseClaims` -- typed accessor over the JWT body.
- :func:`verify_license` -- pure function, no FastAPI dependency, used
  by tests and the CLI.
- :class:`LicenseMiddleware` -- ASGI middleware that loads the license
  once at startup, exposes it via :attr:`Request.state.license`, and
  refuses traffic when the license is missing, malformed, expired,
  unknown, or fingerprint-mismatched.
- :func:`load_license_from_env` -- helper used by both the middleware
  and ``latence-trace license inspect``.

Defaults: enforcement is **enabled** (``LATENCE_TRACE_LICENSE_REQUIRE``
defaults to ``true``); CI / dev environments opt out by setting
``LATENCE_TRACE_LICENSE_REQUIRE=false``.
"""

from latence_trace.auth.license import (
    DEFAULT_LICENSE_FILES,
    LicenseClaims,
    LicenseError,
    LicenseExpired,
    LicenseInvalid,
    LicenseMissing,
    LicenseNotYetValid,
    decode_license,
    load_license_from_env,
    public_key_pem,
    verify_license,
)
from latence_trace.auth.middleware import (
    LicenseMiddleware,
    license_required,
)

__all__ = [
    "DEFAULT_LICENSE_FILES",
    "LicenseClaims",
    "LicenseError",
    "LicenseExpired",
    "LicenseInvalid",
    "LicenseMiddleware",
    "LicenseMissing",
    "LicenseNotYetValid",
    "decode_license",
    "license_required",
    "load_license_from_env",
    "public_key_pem",
    "verify_license",
]
