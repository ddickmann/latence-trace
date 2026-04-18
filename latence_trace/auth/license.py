"""Ed25519-signed JWT license verification.

The license token (`LATENCE_TRACE_LICENSE` env or
`/etc/latence-trace/license` / `~/.latence-trace/license` file) is a
JWT signed with EdDSA / Ed25519 by the latence.ai signing key. The
public counterpart ships with the wheel at
``latence_trace/auth/license_pubkey.pem``.

Required claims:

- ``iss``: must equal ``"latence.ai"`` (issuer).
- ``aud``: must equal ``"latence-trace"`` (audience).
- ``sub``: customer identifier (string, e.g. ``acme-corp``).
- ``iat``: issued-at (numeric date).
- ``exp``: expiry (numeric date). Hard expiry — verified to the second
  with a 60 s clock skew tolerance.
- ``tier``: one of ``trial`` / ``team`` / ``enterprise``.

Optional claims:

- ``nbf``: not-before timestamp.
- ``features``: list of capability tokens this license unlocks
  (``nli``, ``reranker``, ``atomic_claims``, ``semantic_entropy``,
  ``structured_source``).
- ``profiles``: list of profile names the customer may select
  (``fast``, ``balanced``, ``quality``).
- ``max_qps``: advisory rate-limit cap (the rate-limit middleware
  enforces it on its own; this is the contractual ceiling).
- ``max_workers``: maximum FastAPI worker count for this license.
- ``customer``: free-form metadata block (``{"name", "contact", ...}``).
- ``fingerprint``: SHA-256 of the deployment id; if set, must match the
  ``LATENCE_TRACE_DEPLOYMENT_FINGERPRINT`` env at runtime.

The verification path is deliberately pure (no FastAPI dependency, no
file IO once the public key is loaded) so it is easy to unit-test and
to call from the CLI ``license inspect`` subcommand.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import jwt
from jwt import InvalidTokenError

EXPECTED_ISSUER = "latence.ai"
EXPECTED_AUDIENCE = "latence-trace"
ALLOWED_ALGORITHMS: Tuple[str, ...] = ("EdDSA",)
ALLOWED_TIERS: Tuple[str, ...] = ("trial", "team", "enterprise")
DEFAULT_CLOCK_SKEW_SECONDS = 60

DEFAULT_LICENSE_FILES: Tuple[Path, ...] = (
    Path("/etc/latence-trace/license"),
    Path.home() / ".latence-trace" / "license",
)


# --- exceptions ----------------------------------------------------------


class LicenseError(RuntimeError):
    """Base exception for license validation failures.

    Carries a stable ``code`` so the FastAPI middleware can surface it
    in the structured error envelope (``code``, ``message``, ``hint``)
    that the rest of the API uses.
    """

    code: str = "license_error"
    http_status: int = 403


class LicenseMissing(LicenseError):
    code = "license_missing"
    http_status = 402  # Payment Required - the canonical license-missing code.


class LicenseInvalid(LicenseError):
    code = "license_invalid"
    http_status = 403


class LicenseExpired(LicenseError):
    code = "license_expired"
    http_status = 402


class LicenseNotYetValid(LicenseError):
    code = "license_not_yet_valid"
    http_status = 403


# --- claims --------------------------------------------------------------


@dataclass(frozen=True)
class LicenseClaims:
    """Validated, normalised view of a license JWT body.

    Frozen on purpose: a license is supposed to be immutable once issued.
    Mutating values at runtime would silently elevate entitlements and
    is a thing static analyzers should flag.
    """

    subject: str
    tier: str
    issued_at: int
    expires_at: int
    not_before: Optional[int] = None
    features: Tuple[str, ...] = ()
    profiles: Tuple[str, ...] = ()
    max_qps: Optional[int] = None
    max_workers: Optional[int] = None
    fingerprint: Optional[str] = None
    customer: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def has_feature(self, name: str) -> bool:
        return name in self.features

    def allows_profile(self, name: str) -> bool:
        # Empty profile list = customer is allowed everything (legacy
        # entitlement). Listing a profile narrows the entitlement.
        if not self.profiles:
            return True
        return name in self.profiles

    @property
    def days_until_expiry(self) -> float:
        return (self.expires_at - time.time()) / 86400.0

    def is_expiring_soon(self, days: int = 30) -> bool:
        return 0 < self.days_until_expiry <= float(days)

    def to_inspect_dict(self) -> dict:
        return {
            "subject": self.subject,
            "tier": self.tier,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "not_before": self.not_before,
            "features": list(self.features),
            "profiles": list(self.profiles),
            "max_qps": self.max_qps,
            "max_workers": self.max_workers,
            "fingerprint": self.fingerprint,
            "customer": dict(self.customer),
            "days_until_expiry": round(self.days_until_expiry, 2),
            "expiring_soon": self.is_expiring_soon(),
        }


# --- public key loading --------------------------------------------------


def public_key_pem() -> str:
    """Resolve the PEM-encoded Ed25519 public key.

    Resolution order:

    1. ``LATENCE_TRACE_LICENSE_PUBKEY`` (raw PEM string or path) -- lets
       customers pin a different signing key for staging vs production
       without rebuilding the wheel.
    2. ``LATENCE_TRACE_LICENSE_PUBKEY_FILE`` (explicit file path).
    3. The bundled ``latence_trace/auth/license_pubkey.pem`` shipped in
       the wheel (default for production deployments).
    """

    env = os.environ.get("LATENCE_TRACE_LICENSE_PUBKEY")
    if env:
        env = env.strip()
        if env.startswith("-----BEGIN "):
            return env
        path = Path(env)
        if path.is_file():
            return path.read_text(encoding="utf-8")

    file_env = os.environ.get("LATENCE_TRACE_LICENSE_PUBKEY_FILE")
    if file_env:
        return Path(file_env).read_text(encoding="utf-8")

    bundled = Path(__file__).resolve().parent / "license_pubkey.pem"
    return bundled.read_text(encoding="utf-8")


# --- verification --------------------------------------------------------


def decode_license(token: str, *, public_key: Optional[str] = None) -> Mapping[str, Any]:
    """Decode and signature-verify ``token`` without claim normalisation.

    Raises :class:`LicenseInvalid` on signature, algorithm, audience,
    or issuer mismatch; :class:`LicenseExpired` / :class:`LicenseNotYetValid`
    when the temporal claims fail; :class:`LicenseMissing` when the
    token string is empty.
    """

    if not token or not token.strip():
        raise LicenseMissing("license token is empty")
    pem = public_key or public_key_pem()
    try:
        return jwt.decode(
            token,
            key=pem,
            algorithms=list(ALLOWED_ALGORITHMS),
            audience=EXPECTED_AUDIENCE,
            issuer=EXPECTED_ISSUER,
            leeway=DEFAULT_CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise LicenseExpired("license has expired; renew at https://latence.ai/trace/renew") from exc
    except jwt.ImmatureSignatureError as exc:
        raise LicenseNotYetValid("license is not yet valid (nbf in future)") from exc
    except InvalidTokenError as exc:
        raise LicenseInvalid(f"license token invalid: {exc}") from exc


def verify_license(
    token: str,
    *,
    public_key: Optional[str] = None,
    expected_fingerprint: Optional[str] = None,
) -> LicenseClaims:
    """Decode + normalise + cross-check a license token.

    On top of :func:`decode_license`, validates:

    - ``tier`` is one of the ALLOWED_TIERS;
    - ``features`` and ``profiles`` are lists of strings;
    - ``max_qps`` / ``max_workers`` are positive ints if present;
    - the optional ``fingerprint`` claim matches
      ``expected_fingerprint`` (typically the
      ``LATENCE_TRACE_DEPLOYMENT_FINGERPRINT`` env hash).
    """

    payload = decode_license(token, public_key=public_key)

    tier = str(payload.get("tier") or "").strip().lower()
    if tier not in ALLOWED_TIERS:
        raise LicenseInvalid(
            f"license tier {tier!r} not recognised; expected one of {ALLOWED_TIERS}"
        )

    def _string_seq(name: str) -> Tuple[str, ...]:
        raw = payload.get(name) or ()
        if not isinstance(raw, (list, tuple)):
            raise LicenseInvalid(f"claim {name!r} must be a list of strings")
        out: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry:
                raise LicenseInvalid(f"claim {name!r} must contain non-empty strings only")
            out.append(entry)
        return tuple(out)

    def _opt_pos_int(name: str) -> Optional[int]:
        raw = payload.get(name)
        if raw is None:
            return None
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise LicenseInvalid(f"claim {name!r} must be a positive integer")
        value = int(raw)
        if value <= 0:
            raise LicenseInvalid(f"claim {name!r} must be > 0; got {value}")
        return value

    fingerprint = payload.get("fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise LicenseInvalid("claim 'fingerprint' must be a string when set")

    if fingerprint and expected_fingerprint and fingerprint != expected_fingerprint:
        raise LicenseInvalid(
            "license fingerprint does not match the deployment fingerprint; "
            "regenerate the license against this deployment id"
        )

    customer = payload.get("customer")
    if customer is None:
        customer = {}
    if not isinstance(customer, Mapping):
        raise LicenseInvalid("claim 'customer' must be a JSON object when set")

    return LicenseClaims(
        subject=str(payload["sub"]),
        tier=tier,
        issued_at=int(payload["iat"]),
        expires_at=int(payload["exp"]),
        not_before=int(payload["nbf"]) if payload.get("nbf") is not None else None,
        features=_string_seq("features"),
        profiles=_string_seq("profiles"),
        max_qps=_opt_pos_int("max_qps"),
        max_workers=_opt_pos_int("max_workers"),
        fingerprint=fingerprint or None,
        customer=dict(customer),
        raw=dict(payload),
    )


# --- env loading ---------------------------------------------------------


def _read_token_from_path(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        raise LicenseInvalid(
            f"license file at {path} is not readable: {exc}"
        ) from exc
    return text or None


def _candidate_paths() -> Sequence[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("LATENCE_TRACE_LICENSE_FILE")
    if explicit:
        paths.append(Path(explicit))
    paths.extend(DEFAULT_LICENSE_FILES)
    return tuple(paths)


def load_license_from_env(*, require: Optional[bool] = None) -> Optional[LicenseClaims]:
    """Resolve and verify a license token from the runtime environment.

    Resolution order for the token:

    1. ``LATENCE_TRACE_LICENSE`` env -- raw JWT string OR a file path.
    2. ``LATENCE_TRACE_LICENSE_FILE`` env -- explicit file path.
    3. ``/etc/latence-trace/license`` (system-wide).
    4. ``~/.latence-trace/license`` (per-user).

    If a token is found it is verified end-to-end. If no token is found:

    - When ``require`` (or ``LATENCE_TRACE_LICENSE_REQUIRE``) is truthy
      (default), :class:`LicenseMissing` is raised.
    - Otherwise ``None`` is returned (developer mode).
    """

    if require is None:
        env_require = os.environ.get("LATENCE_TRACE_LICENSE_REQUIRE", "true").strip().lower()
        require = env_require in {"1", "true", "yes", "on"}

    expected_fingerprint = os.environ.get("LATENCE_TRACE_DEPLOYMENT_FINGERPRINT") or None

    token: Optional[str] = None
    inline = os.environ.get("LATENCE_TRACE_LICENSE")
    if inline:
        inline = inline.strip()
        if inline.count(".") == 2 and not inline.startswith("/") and not inline.startswith("~"):
            token = inline
        else:
            token = _read_token_from_path(Path(inline).expanduser())

    if not token:
        for path in _candidate_paths():
            token = _read_token_from_path(path.expanduser())
            if token:
                break

    if not token:
        if require:
            raise LicenseMissing(
                "no license token found; set LATENCE_TRACE_LICENSE or place the "
                "token at /etc/latence-trace/license or ~/.latence-trace/license. "
                "Set LATENCE_TRACE_LICENSE_REQUIRE=false to disable enforcement "
                "in development environments."
            )
        return None

    return verify_license(token, expected_fingerprint=expected_fingerprint)


# --- helpers used by CLI / smoke tests -----------------------------------


def deployment_fingerprint(seed: str) -> str:
    """Compute the fingerprint claim from a deployment seed string.

    Operators run::

        latence-trace license fingerprint --seed "$(uname -m)-$(cat /etc/machine-id)"

    and ship the resulting hash back to latence.ai when requesting a
    pinned license. The default seed is the literal string the operator
    passes in; we don't guess values from the environment to keep the
    fingerprint reproducible across container restarts.
    """

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()
