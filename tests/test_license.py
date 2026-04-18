"""Unit tests for the L3 license verifier + middleware."""

from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from latence_trace.auth import (
    LicenseClaims,
    LicenseExpired,
    LicenseInvalid,
    LicenseMiddleware,
    LicenseMissing,
    decode_license,
    verify_license,
)
from latence_trace.auth.license import deployment_fingerprint


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def keypair() -> tuple[str, str]:
    """Fresh Ed25519 keypair scoped to the test module so signing key
    material never leaks across tests."""

    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    sk_pem = sk.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    ).decode()
    pk_pem = pk.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return sk_pem, pk_pem


def _mint(sk_pem: str, **overrides) -> str:
    """Mint a JWT with sensible defaults; overrides win per-claim."""

    now = int(time.time())
    payload = {
        "iss": "latence.ai",
        "aud": "latence-trace",
        "sub": "acme-corp",
        "iat": now,
        "exp": now + 3600,
        "tier": "enterprise",
        "features": ["nli", "reranker", "atomic_claims"],
        "profiles": ["fast", "balanced", "quality"],
        "max_qps": 50,
        "max_workers": 4,
        "customer": {"name": "Acme Corp", "contact": "ops@acme.test"},
    }
    payload.update(overrides)
    return jwt.encode(payload, sk_pem, algorithm="EdDSA")


# ---------------------------------------------------------------------------
# decode + verify
# ---------------------------------------------------------------------------


def test_decode_license_round_trip(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem)
    payload = decode_license(token, public_key=pk_pem)
    assert payload["sub"] == "acme-corp"
    assert payload["aud"] == "latence-trace"
    assert payload["tier"] == "enterprise"


def test_verify_license_normalises_claims(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem)
    claims = verify_license(token, public_key=pk_pem)
    assert isinstance(claims, LicenseClaims)
    assert claims.subject == "acme-corp"
    assert claims.tier == "enterprise"
    assert "nli" in claims.features
    assert claims.allows_profile("fast") is True
    assert claims.allows_profile("invented_profile") is False
    assert claims.has_feature("nli") is True
    assert claims.max_qps == 50


def test_expired_token_raises_license_expired(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    # exp 10 minutes in the past so even with the 60s leeway it fails.
    token = _mint(sk_pem, exp=int(time.time()) - 600)
    with pytest.raises(LicenseExpired):
        verify_license(token, public_key=pk_pem)


def test_wrong_audience_rejected(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem, aud="someone-else")
    with pytest.raises(LicenseInvalid):
        verify_license(token, public_key=pk_pem)


def test_wrong_issuer_rejected(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem, iss="evil.ai")
    with pytest.raises(LicenseInvalid):
        verify_license(token, public_key=pk_pem)


def test_unknown_tier_rejected(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem, tier="hyperscale")
    with pytest.raises(LicenseInvalid):
        verify_license(token, public_key=pk_pem)


def test_negative_max_qps_rejected(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem, max_qps=-5)
    with pytest.raises(LicenseInvalid):
        verify_license(token, public_key=pk_pem)


def test_fingerprint_mismatch_rejected(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    token = _mint(sk_pem, fingerprint="abc123")
    with pytest.raises(LicenseInvalid):
        verify_license(
            token,
            public_key=pk_pem,
            expected_fingerprint="something-else",
        )


def test_fingerprint_match_accepted(keypair: tuple[str, str]) -> None:
    sk_pem, pk_pem = keypair
    fp = deployment_fingerprint("test-cluster-7")
    token = _mint(sk_pem, fingerprint=fp)
    claims = verify_license(token, public_key=pk_pem, expected_fingerprint=fp)
    assert claims.fingerprint == fp


def test_empty_token_raises_license_missing() -> None:
    with pytest.raises(LicenseMissing):
        decode_license("")


# ---------------------------------------------------------------------------
# middleware (end-to-end ASGI)
# ---------------------------------------------------------------------------


def _build_app(license_claims=None, *, require=False) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        LicenseMiddleware,
        license_claims=license_claims,
        require=require,
        exempt_paths=("/healthz",),
    )

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/protected")
    def protected() -> dict:
        return {"ok": True}

    return TestClient(app)


def test_middleware_lets_health_through_without_license() -> None:
    client = _build_app(license_claims=None, require=True)
    assert client.get("/healthz").status_code == 200


def test_middleware_returns_402_when_license_required_but_missing() -> None:
    client = _build_app(license_claims=None, require=True)
    response = client.get("/protected")
    assert response.status_code == 402
    body = response.json()
    assert body["detail"]["code"] == "license_missing"


def test_middleware_passes_request_with_valid_claims(
    keypair: tuple[str, str],
) -> None:
    sk_pem, pk_pem = keypair
    claims = verify_license(_mint(sk_pem), public_key=pk_pem)
    client = _build_app(license_claims=claims, require=True)
    assert client.get("/protected").status_code == 200


def test_middleware_dev_mode_allows_unlicensed_traffic() -> None:
    client = _build_app(license_claims=None, require=False)
    assert client.get("/protected").status_code == 200
