"""Multi-language AST symbol extractor for the code lane.

Parses response *and* context fenced code blocks with tree-sitter, pulls
out a normalised symbol set (imports, classes, functions, methods,
kwargs, string literals that look like module paths), and derives three
deterministic drift signals:

- ``ast_literal_drift_count``   — response identifiers not present in
  any context unit.
- ``ast_phantom_symbol_count``  — imported *modules* the response
  references that have no prefix match in any context import.
- ``ast_phantom_verdict``       — boolean precision-1.0 flag raised when
  at least one phantom import/class/function was detected.

Design
------
- **Languages** — Python, TypeScript, JavaScript, Go, Rust. Parsers are
  preloaded into a process-wide registry so the first request after
  warmup pays zero cold cost.
- **Graceful degradation** — if tree-sitter is unavailable (optional
  dependency not installed) the module falls back to a conservative
  regex-based extractor that still catches the obvious
  ``import foo.bar`` / ``class Baz`` phantom cases. Every downstream
  consumer checks ``AstGroundingResult.parser_backend`` to know which
  path ran.
- **Thread-safe** — :class:`AstSymbolExtractor` holds a parser per
  language protected by a lock so concurrent requests cannot corrupt
  the parser's internal incremental state.
- **Hot-path friendly** — parsing the response + each unit stays on
  the CPU and runs in a background thread via
  ``asyncio.to_thread`` from the orchestrator.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES: Tuple[str, ...] = (
    "python",
    "typescript",
    "javascript",
    "go",
    "rust",
)

_LANGUAGE_ALIASES: Dict[str, str] = {
    "py": "python",
    "python": "python",
    "python3": "python",
    "ts": "typescript",
    "tsx": "typescript",
    "typescript": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "javascript": "javascript",
    "golang": "go",
    "go": "go",
    "rs": "rust",
    "rust": "rust",
}


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolSet:
    """Normalised bag of identifiers extracted from a code block.

    Every set is frozen and hashable so we can intersect / diff cheaply
    at the drift-detection layer.
    """

    imports: FrozenSet[str] = frozenset()
    classes: FrozenSet[str] = frozenset()
    functions: FrozenSet[str] = frozenset()
    methods: FrozenSet[str] = frozenset()
    kwargs: FrozenSet[str] = frozenset()
    identifiers: FrozenSet[str] = frozenset()

    def union(self, other: "SymbolSet") -> "SymbolSet":
        return SymbolSet(
            imports=self.imports | other.imports,
            classes=self.classes | other.classes,
            functions=self.functions | other.functions,
            methods=self.methods | other.methods,
            kwargs=self.kwargs | other.kwargs,
            identifiers=self.identifiers | other.identifiers,
        )

    def is_empty(self) -> bool:
        return not (
            self.imports
            or self.classes
            or self.functions
            or self.methods
            or self.kwargs
            or self.identifiers
        )


@dataclass
class AstGroundingResult:
    """Per-turn AST drift bundle.

    All counts are derived from the ``response_symbols`` and
    ``context_symbols`` fields so callers can audit the verdict.
    """

    language: Optional[str]
    parser_backend: str  # "tree_sitter" | "regex_fallback" | "disabled"
    ast_literal_drift_count: int
    ast_phantom_symbol_count: int
    ast_phantom_verdict: bool
    drift_symbols: List[str] = field(default_factory=list)
    phantom_symbols: List[str] = field(default_factory=list)
    response_symbols: SymbolSet = field(default_factory=SymbolSet)
    context_symbols: SymbolSet = field(default_factory=SymbolSet)
    latency_ms: float = 0.0


# ----------------------------------------------------------------------
# Tree-sitter loader
# ----------------------------------------------------------------------


@dataclass
class _LanguageBundle:
    """Holds the shared ``Language`` + a per-thread parser pool."""

    language: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    parser: object = None


def _load_language_bundles() -> Dict[str, _LanguageBundle]:
    """Load every supported grammar. Never raises.

    Missing packages degrade to the regex fallback at call time; the
    returned dict only contains languages that loaded successfully so
    callers can gate on membership.
    """
    bundles: Dict[str, _LanguageBundle] = {}
    try:
        from tree_sitter import Language, Parser  # type: ignore
    except Exception as exc:
        logger.info("ast_tree_sitter_unavailable", extra={"error": str(exc)})
        return bundles

    def _maybe_add(name: str, factory) -> None:
        try:
            language_obj = Language(factory())
        except Exception as exc:
            logger.info(
                "ast_language_load_failed",
                extra={"language": name, "error": str(exc)},
            )
            return
        bundle = _LanguageBundle(language=language_obj)
        try:
            bundle.parser = Parser(language_obj)
        except Exception as exc:
            logger.info(
                "ast_parser_build_failed",
                extra={"language": name, "error": str(exc)},
            )
            return
        bundles[name] = bundle

    try:
        import tree_sitter_python  # type: ignore

        _maybe_add("python", tree_sitter_python.language)
    except Exception:
        pass
    try:
        import tree_sitter_typescript  # type: ignore

        _maybe_add("typescript", tree_sitter_typescript.language_typescript)
    except Exception:
        pass
    try:
        import tree_sitter_javascript  # type: ignore

        _maybe_add("javascript", tree_sitter_javascript.language)
    except Exception:
        pass
    try:
        import tree_sitter_go  # type: ignore

        _maybe_add("go", tree_sitter_go.language)
    except Exception:
        pass
    try:
        import tree_sitter_rust  # type: ignore

        _maybe_add("rust", tree_sitter_rust.language)
    except Exception:
        pass
    return bundles


_BUNDLES: Optional[Dict[str, _LanguageBundle]] = None
_BUNDLE_LOCK = threading.Lock()


def _bundles() -> Dict[str, _LanguageBundle]:
    global _BUNDLES
    if _BUNDLES is not None:
        return _BUNDLES
    with _BUNDLE_LOCK:
        if _BUNDLES is None:
            _BUNDLES = _load_language_bundles()
    return _BUNDLES


def preload_grammars() -> List[str]:
    """Force-load every grammar. Called from the handler warmup.

    Returns the list of languages that loaded successfully so
    ``/readyz`` can report which grammars are hot.
    """
    return sorted(_bundles().keys())


# ----------------------------------------------------------------------
# Language detection
# ----------------------------------------------------------------------


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)```", re.DOTALL)


def detect_language(
    text: str,
    *,
    hint: Optional[str] = None,
) -> Optional[str]:
    """Best-effort language detection.

    Priority:

    1. Explicit ``hint`` (normalised through the alias table).
    2. Fenced code block info string (```` ```python ```` → ``python``).
    3. Simple content heuristics (Python keywords, TS ``import``, etc.).

    Returns ``None`` when nothing convincing is found so downstream
    attributors can skip the AST pass rather than run a wrong parser.
    """
    if hint:
        alias = _LANGUAGE_ALIASES.get(hint.strip().lower())
        if alias:
            return alias
    if not text:
        return None
    for match in _FENCE_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        if lang:
            alias = _LANGUAGE_ALIASES.get(lang)
            if alias:
                return alias
    lower = text.lower()
    if "def " in lower and "self" in lower and "import " in lower:
        return "python"
    if " const " in lower and "=>" in lower:
        return "typescript"
    if "package main" in lower and "func " in lower:
        return "go"
    if "fn main" in lower or "let mut " in lower:
        return "rust"
    if "function " in lower and "=>" in lower:
        return "javascript"
    return None


def _extract_fenced_blocks(text: str) -> List[Tuple[Optional[str], str]]:
    """Return ``(language, code)`` pairs from any fenced code blocks.

    When no fenced block is present we return the whole text as a single
    anonymous block so the extractor still picks up paste-style code.
    """
    blocks: List[Tuple[Optional[str], str]] = []
    for match in _FENCE_RE.finditer(text or ""):
        lang = (match.group(1) or "").strip().lower() or None
        blocks.append((lang, match.group(2)))
    if not blocks and text:
        blocks.append((None, text))
    return blocks


# ----------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


# Tree-sitter node-type sets per language.
_NODE_TYPES: Dict[str, Dict[str, Sequence[str]]] = {
    "python": {
        "import": ("import_statement", "import_from_statement"),
        "class": ("class_definition",),
        "function": ("function_definition",),
        "keyword_argument": ("keyword_argument",),
        "call": ("call",),
    },
    "typescript": {
        "import": ("import_statement", "import_require_clause"),
        "class": ("class_declaration",),
        "function": ("function_declaration", "method_definition", "arrow_function"),
        "keyword_argument": ("property_identifier",),
        "call": ("call_expression",),
    },
    "javascript": {
        "import": ("import_statement",),
        "class": ("class_declaration",),
        "function": ("function_declaration", "method_definition", "arrow_function"),
        "keyword_argument": ("property_identifier",),
        "call": ("call_expression",),
    },
    "go": {
        "import": ("import_declaration", "import_spec"),
        "class": ("type_spec",),
        "function": ("function_declaration", "method_declaration"),
        "keyword_argument": ("keyed_element",),
        "call": ("call_expression",),
    },
    "rust": {
        "import": ("use_declaration",),
        "class": ("struct_item", "enum_item"),
        "function": ("function_item",),
        "keyword_argument": ("field_initializer",),
        "call": ("call_expression",),
    },
}


def _node_text(node, source: bytes) -> str:
    try:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _walk(node, source: bytes):
    yield node
    for child in node.children:
        yield from _walk(child, source)


def _extract_with_tree_sitter(language: str, code: str) -> SymbolSet:
    bundle = _bundles().get(language)
    if bundle is None or bundle.parser is None:
        return SymbolSet()
    node_types = _NODE_TYPES.get(language, {})
    source = code.encode("utf-8", errors="ignore")

    imports: Set[str] = set()
    classes: Set[str] = set()
    functions: Set[str] = set()
    methods: Set[str] = set()
    kwargs: Set[str] = set()
    identifiers: Set[str] = set()

    with bundle.lock:
        try:
            tree = bundle.parser.parse(source)
        except Exception as exc:
            logger.info("ast_parse_failed", extra={"language": language, "error": str(exc)})
            return _extract_with_regex(code)
    root = tree.root_node
    for node in _walk(root, source):
        ntype = node.type
        text = _node_text(node, source)
        if ntype == "identifier" or ntype == "type_identifier":
            if text:
                identifiers.add(text)
        if ntype in node_types.get("import", ()):  # type: ignore[operator]
            # Pull every identifier token inside the import statement.
            for ident in _IDENT_RE.finditer(text):
                name = ident.group(0)
                if name:
                    imports.add(name)
        if ntype in node_types.get("class", ()):  # type: ignore[operator]
            name_node = node.child_by_field_name("name") if hasattr(node, "child_by_field_name") else None
            if name_node is not None:
                classes.add(_node_text(name_node, source))
        if ntype in node_types.get("function", ()):  # type: ignore[operator]
            name_node = node.child_by_field_name("name") if hasattr(node, "child_by_field_name") else None
            if name_node is not None:
                fname = _node_text(name_node, source)
                if fname:
                    if ntype.startswith("method"):
                        methods.add(fname)
                    else:
                        functions.add(fname)
        if ntype in node_types.get("keyword_argument", ()):  # type: ignore[operator]
            name_node = node.child_by_field_name("name") if hasattr(node, "child_by_field_name") else None
            if name_node is not None:
                kw = _node_text(name_node, source)
                if kw:
                    kwargs.add(kw)
        if ntype in node_types.get("call", ()):  # type: ignore[operator]
            fn_node = node.child_by_field_name("function") if hasattr(node, "child_by_field_name") else None
            if fn_node is not None:
                call_text = _node_text(fn_node, source)
                if "." in call_text:
                    tail = call_text.rsplit(".", 1)[-1]
                    if tail:
                        methods.add(tail)
                elif call_text:
                    functions.add(call_text)

    return SymbolSet(
        imports=frozenset(imports),
        classes=frozenset(classes),
        functions=frozenset(functions),
        methods=frozenset(methods),
        kwargs=frozenset(kwargs),
        identifiers=frozenset(identifiers),
    )


def _extract_with_regex(code: str) -> SymbolSet:
    """Conservative fallback when tree-sitter is unavailable."""
    imports: Set[str] = set()
    classes: Set[str] = set()
    functions: Set[str] = set()
    methods: Set[str] = set()
    kwargs: Set[str] = set()
    identifiers: Set[str] = set()

    for match in re.finditer(r"(?m)^\s*(?:from|import)\s+([\w\.]+)", code):
        imports.update(match.group(1).split("."))
    for match in re.finditer(r"(?m)^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", code):
        imports.update(match.group(1).split("/"))
    for match in re.finditer(r"(?m)^\s*(?:pub\s+)?class\s+([A-Za-z_][\w]*)", code):
        classes.add(match.group(1))
    for match in re.finditer(r"(?m)^\s*(?:pub\s+)?(?:struct|enum)\s+([A-Za-z_][\w]*)", code):
        classes.add(match.group(1))
    for match in re.finditer(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_][\w]*)", code):
        functions.add(match.group(1))
    for match in re.finditer(r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][\w]*)", code):
        functions.add(match.group(1))
    for match in re.finditer(r"(?m)^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)", code):
        functions.add(match.group(1))
    for match in re.finditer(r"(?m)^\s*func\s+(?:\([^)]*\)\s+)?([A-Za-z_][\w]*)", code):
        functions.add(match.group(1))
    for match in re.finditer(r"([A-Za-z_][\w]*)\s*=\s*[^=]", code):
        kwargs.add(match.group(1))
    for match in _IDENT_RE.finditer(code):
        name = match.group(0)
        if name and name not in {"if", "else", "for", "while", "return", "self", "this"}:
            identifiers.add(name)
    return SymbolSet(
        imports=frozenset(imports),
        classes=frozenset(classes),
        functions=frozenset(functions),
        methods=frozenset(methods),
        kwargs=frozenset(kwargs),
        identifiers=frozenset(identifiers),
    )


# ----------------------------------------------------------------------
# Public extractor
# ----------------------------------------------------------------------


class AstSymbolExtractor:
    """Entry point used by the code-lane orchestrator.

    The class is stateless and thread-safe; all per-language mutable
    state lives in :class:`_LanguageBundle`. Instantiate once in the
    handler singletons and reuse across requests.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        # Pre-load grammars so the first request never pays the cold
        # import cost.
        if self.enabled:
            preload_grammars()

    @staticmethod
    def available_languages() -> List[str]:
        return sorted(_bundles().keys())

    @property
    def parser_backend(self) -> str:
        if not self.enabled:
            return "disabled"
        return "tree_sitter" if _bundles() else "regex_fallback"

    def extract_block(self, language: Optional[str], code: str) -> SymbolSet:
        if not self.enabled or not code:
            return SymbolSet()
        if language and language in _bundles():
            return _extract_with_tree_sitter(language, code)
        return _extract_with_regex(code)

    def extract_from_text(
        self,
        text: str,
        *,
        language_hint: Optional[str] = None,
    ) -> Tuple[Optional[str], SymbolSet]:
        """Extract symbols from free-form text containing code fences.

        If the text has fenced blocks, we parse each block with its own
        language (honouring the per-fence info string first, then the
        global ``language_hint``). The returned ``language`` is the
        most-common block language so downstream consumers know which
        language to render.
        """
        if not self.enabled or not text:
            return None, SymbolSet()
        blocks = _extract_fenced_blocks(text)
        languages_seen: List[str] = []
        combined = SymbolSet()
        for block_lang, code in blocks:
            lang = block_lang or language_hint
            resolved = detect_language(code, hint=lang)
            if resolved:
                languages_seen.append(resolved)
            block_symbols = self.extract_block(resolved, code)
            combined = combined.union(block_symbols)
        # Pick the most-common block language as representative.
        representative: Optional[str] = None
        if languages_seen:
            counts: Dict[str, int] = {}
            for lang in languages_seen:
                counts[lang] = counts.get(lang, 0) + 1
            representative = max(counts.items(), key=lambda kv: kv[1])[0]
        return representative, combined


# ----------------------------------------------------------------------
# Drift / phantom derivation
# ----------------------------------------------------------------------


def _contains_prefix(haystack: Iterable[str], needle: str) -> bool:
    for item in haystack:
        if not item:
            continue
        if item == needle or item.startswith(needle + ".") or needle.startswith(item + "."):
            return True
    return False


def extract_symbols(
    *,
    response_text: str,
    context_texts: Sequence[str],
    extractor: Optional[AstSymbolExtractor] = None,
    language_hint: Optional[str] = None,
) -> AstGroundingResult:
    """Extract symbols from response + context and derive drift signals.

    Parameters
    ----------
    response_text:
        Full ``response_text`` from the request (the orchestrator passes
        the unchunked version so we see every code fence).
    context_texts:
        List of *raw* context texts (one per context file or one per
        support unit, depending on the caller). Order does not matter.
    extractor:
        Optional pre-built :class:`AstSymbolExtractor`. Defaults to a
        process-wide singleton.
    language_hint:
        Optional ``response_language_hint`` from the request.
    """
    import time as _time

    start = _time.perf_counter()
    extractor = extractor or _default_extractor()
    language, response_symbols = extractor.extract_from_text(
        response_text, language_hint=language_hint
    )

    context_symbols = SymbolSet()
    for ctx in context_texts:
        _, syms = extractor.extract_from_text(ctx, language_hint=language)
        context_symbols = context_symbols.union(syms)

    # Drift: any response identifier (class/function/method/import/kwarg)
    # not present in any context unit.
    drift_candidates: Set[str] = set()
    drift_candidates |= response_symbols.classes - context_symbols.classes
    drift_candidates |= response_symbols.functions - context_symbols.functions - context_symbols.methods
    drift_candidates |= response_symbols.methods - context_symbols.methods - context_symbols.functions
    drift_candidates |= response_symbols.kwargs - context_symbols.kwargs
    # Exclude whitelisted identifiers that every codebase ships with
    # (built-in types, common std-lib names). These are not drift.
    builtin_whitelist = {
        "self",
        "cls",
        "None",
        "True",
        "False",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "print",
    }
    drift_symbols = sorted(drift_candidates - builtin_whitelist)

    # Phantom imports: response import roots not present (even as prefix)
    # in context imports.
    phantom_symbols: Set[str] = set()
    for imp in response_symbols.imports:
        if not imp or imp in builtin_whitelist:
            continue
        if not _contains_prefix(context_symbols.imports, imp) and imp not in context_symbols.identifiers:
            phantom_symbols.add(imp)
    # Phantom classes: response classes that do not appear anywhere in
    # the context, in any symbol category.
    context_any = (
        context_symbols.classes
        | context_symbols.functions
        | context_symbols.methods
        | context_symbols.identifiers
        | context_symbols.imports
    )
    for cls in response_symbols.classes:
        if cls and cls not in context_any and cls not in builtin_whitelist:
            phantom_symbols.add(cls)

    verdict = bool(phantom_symbols)

    latency_ms = (_time.perf_counter() - start) * 1000.0
    return AstGroundingResult(
        language=language,
        parser_backend=extractor.parser_backend,
        ast_literal_drift_count=len(drift_symbols),
        ast_phantom_symbol_count=len(phantom_symbols),
        ast_phantom_verdict=verdict,
        drift_symbols=drift_symbols,
        phantom_symbols=sorted(phantom_symbols),
        response_symbols=response_symbols,
        context_symbols=context_symbols,
        latency_ms=latency_ms,
    )


# ----------------------------------------------------------------------
# Default extractor singleton
# ----------------------------------------------------------------------


_DEFAULT_EXTRACTOR: Optional[AstSymbolExtractor] = None
_EXTRACTOR_LOCK = threading.Lock()


def _default_extractor() -> AstSymbolExtractor:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is not None:
        return _DEFAULT_EXTRACTOR
    with _EXTRACTOR_LOCK:
        if _DEFAULT_EXTRACTOR is None:
            _DEFAULT_EXTRACTOR = AstSymbolExtractor(enabled=True)
    return _DEFAULT_EXTRACTOR


def reset_default_extractor() -> None:
    """Drop the cached singleton; used by tests."""
    global _DEFAULT_EXTRACTOR
    with _EXTRACTOR_LOCK:
        _DEFAULT_EXTRACTOR = None


async def extract_symbols_async(
    *,
    response_text: str,
    context_texts: Sequence[str],
    extractor: Optional[AstSymbolExtractor] = None,
    language_hint: Optional[str] = None,
) -> AstGroundingResult:
    """Async wrapper around :func:`extract_symbols`.

    tree-sitter parsing releases the GIL in C, but building the symbol
    sets is pure Python. Running the whole extract pass in a thread
    keeps the event loop free for concurrent requests.
    """
    import asyncio as _asyncio

    return await _asyncio.to_thread(
        extract_symbols,
        response_text=response_text,
        context_texts=list(context_texts),
        extractor=extractor,
        language_hint=language_hint,
    )
