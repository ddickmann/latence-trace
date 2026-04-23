"""Hand-crafted coding-agent groundedness fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CodeContextFile:
    path: str
    content: str


@dataclass(frozen=True)
class CodingCase:
    id: str
    source: str
    query: str
    response: str
    label: str
    subcategory: Optional[str]
    notes: str
    context_files: List[CodeContextFile]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _file(path: str, content: str) -> CodeContextFile:
    return CodeContextFile(path=path, content=dedent(content).strip() + "\n")


HANDCRAFTED_CASES: List[CodingCase] = [
    CodingCase(
        id="CG1",
        source="handcrafted:python_client_timeout",
        query="Honor request_timeout when creating the async HTTP client.",
        response=dedent(
            """
            diff --git a/src/service/http.py b/src/service/http.py
            @@
            -    return httpx.AsyncClient(base_url=settings.api_base)
            +    return httpx.AsyncClient(
            +        base_url=settings.api_base,
            +        timeout=settings.request_timeout,
            +    )
            """
        ).strip(),
        label="grounded",
        subcategory=None,
        notes="Purely grounded edit that only reuses symbols already present in the retrieved code.",
        context_files=[
            _file(
                "src/service/http.py",
                """
                import httpx
                from app.config import Settings

                def create_client(settings: Settings) -> httpx.AsyncClient:
                    return httpx.AsyncClient(base_url=settings.api_base)
                """,
            ),
            _file(
                "src/app/config.py",
                """
                from pydantic import BaseModel

                class Settings(BaseModel):
                    api_base: str
                    request_timeout: float = 5.0
                """,
            ),
        ],
    ),
    CodingCase(
        id="CG2",
        source="handcrafted:typescript_session_headers",
        query="Attach the existing auth header when loading the session.",
        response=dedent(
            """
            diff --git a/src/store/session.ts b/src/store/session.ts
            @@
            -  return await fetchSession(token);
            +  return await fetchSession(token, buildHeaders(token));
            """
        ).strip(),
        label="grounded",
        subcategory=None,
        notes="Grounded TypeScript edit that only composes existing local helpers.",
        context_files=[
            _file(
                "src/store/session.ts",
                """
                import { buildHeaders } from "../http/headers";

                export async function loadSession(storage: Storage) {
                  const token = storage.getItem("session_token");
                  if (!token) return null;
                  return await fetchSession(token);
                }

                async function fetchSession(token: string, headers?: HeadersInit) {
                  return request(`/session/${token}`, { headers });
                }
                """,
            ),
            _file(
                "src/http/headers.ts",
                """
                export function buildHeaders(token: string): HeadersInit {
                  return {
                    Authorization: `Bearer ${token}`,
                  };
                }
                """,
            ),
        ],
    ),
    CodingCase(
        id="CG3",
        source="handcrafted:python_report_footer",
        query="When include_footer is true, append the existing footer helper output.",
        response=dedent(
            """
            diff --git a/src/reporting/formatter.py b/src/reporting/formatter.py
            @@
                 body = "\\n".join(rows)
            -    return body
            +    if include_footer:
            +        body = f"{body}\\n{build_footer(rows)}"
            +    return body
            """
        ).strip(),
        label="grounded",
        subcategory=None,
        notes="Grounded edit that branches on an existing flag and uses an existing helper.",
        context_files=[
            _file(
                "src/reporting/formatter.py",
                """
                def build_footer(rows: list[str]) -> str:
                    return f"total={len(rows)}"

                def format_report(rows: list[str], include_footer: bool = False) -> str:
                    body = "\\n".join(rows)
                    return body
                """,
            ),
        ],
    ),
    CodingCase(
        id="CU1",
        source="handcrafted:python_cache_phantom_api",
        query="Batch load user profiles from cache before falling back to the database.",
        response=dedent(
            """
            diff --git a/src/cache/profile_loader.py b/src/cache/profile_loader.py
            @@
            -    cached = [cache.get(user_id) for user_id in user_ids]
            +    cached = cache.mget(user_ids)
            """
        ).strip(),
        label="ungrounded",
        subcategory="phantom_api",
        notes="High-value coding-agent failure mode: the response invents cache.mget() even though the retrieved code only exposes get/set.",
        context_files=[
            _file(
                "src/cache/profile_loader.py",
                """
                class ProfileCache:
                    def get(self, user_id: str):
                        ...

                    def set(self, user_id: str, payload: dict) -> None:
                        ...

                def load_profiles(cache: ProfileCache, user_ids: list[str]) -> list[dict]:
                    cached = [cache.get(user_id) for user_id in user_ids]
                    return [item for item in cached if item is not None]
                """,
            ),
        ],
    ),
    CodingCase(
        id="CU2",
        source="handcrafted:typescript_router_phantom_api",
        query="Add an admin route prefix for the existing users endpoints.",
        response=dedent(
            """
            diff --git a/src/router.ts b/src/router.ts
            @@
            -router.get("/users", listUsers);
            +router.group("/admin", (admin) => {
            +  admin.get("/users", listUsers);
            +});
            """
        ).strip(),
        label="ungrounded",
        subcategory="phantom_api",
        notes="The response invents router.group() even though the router surface only supports get/post.",
        context_files=[
            _file(
                "src/router.ts",
                """
                type Handler = (req: unknown, res: unknown) => void;

                export class Router {
                  get(path: string, handler: Handler) {}
                  post(path: string, handler: Handler) {}
                }

                const router = new Router();
                router.get("/users", listUsers);
                router.post("/users", createUser);
                """,
            ),
        ],
    ),
    CodingCase(
        id="CU3",
        source="handcrafted:python_entity_swap",
        query="Keep the same email helper but wire the arguments correctly.",
        response=dedent(
            """
            diff --git a/src/notifications/emailer.py b/src/notifications/emailer.py
            @@
            -    send_email(recipient, subject, body)
            +    send_email(subject, recipient, body)
            """
        ).strip(),
        label="ungrounded",
        subcategory="entity_swap",
        notes="The response keeps the grounded helper name but swaps argument roles.",
        context_files=[
            _file(
                "src/notifications/emailer.py",
                """
                def send_email(recipient: str, subject: str, body: str) -> None:
                    ...

                def notify_welcome(recipient: str) -> None:
                    subject = "Welcome"
                    body = "Thanks for signing up."
                    send_email(recipient, subject, body)
                """,
            ),
        ],
    ),
    CodingCase(
        id="CU4",
        source="handcrafted:typescript_entity_swap",
        query="Format prices using the existing helper.",
        response=dedent(
            """
            diff --git a/src/ui/prices.ts b/src/ui/prices.ts
            @@
            -  return formatMoney(amount, currency);
            +  return formatMoney(currency, amount);
            """
        ).strip(),
        label="ungrounded",
        subcategory="entity_swap",
        notes="Correct helper, wrong argument ordering.",
        context_files=[
            _file(
                "src/ui/prices.ts",
                """
                export function formatMoney(amount: number, currency: string): string {
                  return `${currency} ${amount.toFixed(2)}`;
                }

                export function renderPrice(amount: number, currency: string): string {
                  return formatMoney(amount, currency);
                }
                """,
            ),
        ],
    ),
    CodingCase(
        id="CU5",
        source="handcrafted:python_parametric_model_dump",
        query="Serialize the payload model before enqueueing the job.",
        response=dedent(
            """
            diff --git a/src/jobs/queue.py b/src/jobs/queue.py
            @@
            -    body = payload.dict()
            +    body = payload.model_dump()
            """
        ).strip(),
        label="ungrounded",
        subcategory="parametric",
        notes="`model_dump()` is real Pydantic v2 knowledge, but it is not grounded in the retrieved repository context.",
        context_files=[
            _file(
                "src/jobs/queue.py",
                """
                from pydantic import BaseModel

                class JobPayload(BaseModel):
                    name: str
                    retries: int = 0

                def enqueue(payload: JobPayload) -> dict:
                    body = payload.dict()
                    return {"payload": body}
                """,
            ),
        ],
    ),
    CodingCase(
        id="CU6",
        source="handcrafted:react_router_parametric",
        query="Navigate back to the dashboard after save.",
        response=dedent(
            """
            diff --git a/src/components/Editor.tsx b/src/components/Editor.tsx
            @@
            -  const history = useHistory();
            +  const navigate = useNavigate();
            @@
            -      history.push("/dashboard");
            +      navigate("/dashboard");
            """
        ).strip(),
        label="ungrounded",
        subcategory="parametric",
        notes="`useNavigate()` is correct in React Router v6, but the retrieved code is clearly written against the v5 `useHistory()` API.",
        context_files=[
            _file(
                "src/components/Editor.tsx",
                """
                import React from "react";
                import { useHistory } from "react-router-dom";

                export function Editor(): JSX.Element {
                  const history = useHistory();

                  async function handleSave() {
                    await saveDocument();
                    history.push("/dashboard");
                  }

                  return <button onClick={handleSave}>Save</button>;
                }
                """,
            ),
        ],
    ),
    CodingCase(
        id="CA1",
        source="handcrafted:python_partial_groundedness",
        query="Normalize incoming webhook payloads before storing them.",
        response=dedent(
            """
            diff --git a/src/webhooks/ingest.py b/src/webhooks/ingest.py
            @@
            -    payload = normalize_payload(raw_payload)
            +    payload = normalize_payload(raw_payload)
            +    payload = normalize_issue_context(payload)
            """
        ).strip(),
        label="ambiguous",
        subcategory="partial",
        notes="Half the response is grounded, but it adds a plausible unsupported helper.",
        context_files=[
            _file(
                "src/webhooks/ingest.py",
                """
                def normalize_payload(raw_payload: dict) -> dict:
                    return {
                        "id": raw_payload["id"],
                        "title": raw_payload["title"].strip(),
                    }

                def ingest(raw_payload: dict) -> dict:
                    payload = normalize_payload(raw_payload)
                    return store(payload)
                """,
            ),
        ],
    ),
    CodingCase(
        id="CA2",
        source="handcrafted:typescript_partial_groundedness",
        query="Retry the existing request with the standard headers attached.",
        response=dedent(
            """
            diff --git a/src/http/client.ts b/src/http/client.ts
            @@
            -  return request(url, { headers: buildHeaders(token) });
            +  return withRetryPolicy(request(url, { headers: buildHeaders(token) }));
            """
        ).strip(),
        label="ambiguous",
        subcategory="partial",
        notes="Grounded header helper plus an unsupported retry wrapper.",
        context_files=[
            _file(
                "src/http/client.ts",
                """
                export function buildHeaders(token: string): HeadersInit {
                  return { Authorization: `Bearer ${token}` };
                }

                export function callApi(url: string, token: string) {
                  return request(url, { headers: buildHeaders(token) });
                }
                """,
            ),
        ],
    ),
    CodingCase(
        id="CA3",
        source="handcrafted:python_deprecated_flip",
        query="Parse config using the non-deprecated loader.",
        response=dedent(
            """
            diff --git a/src/config/loaders.py b/src/config/loaders.py
            @@
            -    return parse_config(raw)
            +    return legacy_parse(raw)
            """
        ).strip(),
        label="ambiguous",
        subcategory="negation_flip",
        notes="The retrieved code explicitly says the legacy helper is deprecated, but the response flips the recommendation.",
        context_files=[
            _file(
                "src/config/loaders.py",
                """
                def legacy_parse(raw: str) -> dict:
                    return {"mode": "legacy"}

                def parse_config(raw: str) -> dict:
                    return {"mode": "stable"}

                def load_config(raw: str) -> dict:
                    # legacy_parse is deprecated. Use parse_config instead.
                    return parse_config(raw)
                """,
            ),
        ],
    ),
]


def load_handcrafted_cases() -> List[CodingCase]:
    return list(HANDCRAFTED_CASES)
