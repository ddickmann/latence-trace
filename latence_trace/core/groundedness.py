"""Shared groundedness helpers for the reference API."""

from __future__ import annotations

import functools
import hashlib
import logging
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from latence_trace.kernels.triton_triangular_maxsim import (
    grounded_coverage,
    naive_reverse_maxsim_qc,
    triangular_maxsim,
    weighted_groundedness,
)
from latence_trace.core.nli import (
    AtomicVerification,
    ClaimVerification,
    NLIProvider,
    PremiseReranker,
    aggregate_nli_score,
    fuse_groundedness_v2,
    is_atomic_enabled,
    is_premise_concat_enabled,
    project_claim_records_to_tokens,
    project_claim_scores_to_tokens,
    verify_claims,
)
from latence_trace.core.semantic_entropy import (
    SemanticEntropyResult,
    compute_semantic_entropy,
    is_semantic_entropy_enabled,
)
from latence_trace.core.thresholds import (
    RiskBandPolicy,
    classify_risk_band,
    get_risk_band_policy,
)
from latence_trace.core.structured import (
    default_penalty_per_mismatch,
    detect_source_format,
    is_structured_enabled,
    resolve_structured_mode,
    verification_to_dict,
    verify_structured_source,
)
from latence_trace.core.attribution.file_attribution import (
    FileAttributionResult,
    attribute_files,
)

logger = logging.getLogger(__name__)

_TOKEN_FALLBACK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.UNICODE)
_SPECIAL_TOKENS = {
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "<s>",
    "</s>",
    "<pad>",
    "<bos>",
    "<eos>",
}
# Stopwords are unioned across English and German so the same content-token
# mask works for monolingual EN, monolingual DE, and mixed-language responses
# without having to detect language first. Adding more languages is purely
# additive: any token whose lowercased form is in this set contributes 0
# weight to the headline reverse-context score.
_STOPWORDS = {
    # English function words. Note that several entries (``a``, ``an``,
    # ``in``, ``is``, ``war``) are also valid German closed-class words; we
    # list them once here and skip duplicates in the German block below.
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    # German function words: articles, common prepositions, copulas,
    # conjunctions, common pronouns. Lower-cased forms only - the
    # _is_content_token path lowercases before lookup.
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einen",
    "einem",
    "einer",
    "eines",
    "und",
    "oder",
    "aber",
    "doch",
    "sondern",
    "weil",
    "dass",
    "wenn",
    "als",
    "wie",
    "von",
    "vom",
    "zu",
    "zum",
    "zur",
    "im",
    # ``in`` and ``an`` already appear above in the English block; they are
    # also valid German closed-class words. Listing them only once keeps
    # the source canonical (the ``set`` would dedupe anyway).
    "am",
    "auf",
    "aus",
    "bei",
    "mit",
    "nach",
    "\u00fcber",
    "unter",
    "vor",
    "hinter",
    "neben",
    "zwischen",
    "f\u00fcr",
    "ohne",
    "gegen",
    "ist",
    "sind",
    "war",
    "waren",
    "wird",
    "werden",
    "wurde",
    "wurden",
    "hat",
    "hatte",
    "haben",
    "hatten",
    "sein",
    "seine",
    "ihre",
    "ihr",
    "ihn",
    "ihm",
    "es",
    "er",
    "sie",
    "wir",
    "uns",
    "euch",
    "mich",
    "dich",
    "mir",
    "dir",
    "nicht",
    "kein",
    "keine",
    "keinen",
    "keinem",
    "keiner",
    "keines",
    "auch",
    "noch",
    "nur",
    "schon",
    "sehr",
    "mehr",
    "denn",
    "daher",
    "dadurch",
    "dabei",
    "damit",
    "darum",
    "hier",
    "dort",
    "so",
}
_MAX_DEBUG_MATRIX_ELEMENTS = 32_768
_DEFAULT_CHUNK_TOKEN_BUDGET = 256
_CONSENSUS_THRESHOLD = 0.85
_CONSENSUS_ALPHA = 20.0
_CONSENSUS_PENALTY_SCALE = 0.03
_CONSENSUS_UNIT_SCALE = 4.0

_CALIBRATION_MIN_STD = 1e-3
_CALIBRATION_TEMPERATURE = 1.0

# Diverse, short, topically unrelated text spans used to build the null
# distribution for per-token calibration. Mixing domains (history, science,
# geography, biology) and languages (English + German) keeps the bank from
# being adversarially close to any single response and gives a stable
# mean/std per response token regardless of input language. Adding more
# languages is purely additive - the calibrated z-score per response token
# only depends on the max similarity across the whole bank.
DEFAULT_NULL_BANK_TEXTS: Tuple[str, ...] = (
    # English null sentences
    "The cat sat on the mat by the window.",
    "In 1492 Christopher Columbus sailed across the Atlantic Ocean.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Eiffel Tower was completed in 1889 on the Champ de Mars in Paris.",
    "Quantum entanglement allows two particles to share a single quantum state.",
    "Water boils at one hundred degrees Celsius at standard sea level pressure.",
    "Shakespeare wrote roughly thirty-seven plays during his lifetime in England.",
    "The Great Wall of China stretches over thirteen thousand miles across Asia.",
    "DNA molecules carry the genetic instructions used by all known organisms.",
    "The first crewed Moon landing took place on the twentieth of July 1969.",
    "Ludwig van Beethoven composed nine symphonies despite progressive deafness.",
    "Light travels at roughly two hundred ninety-nine million meters per second.",
    "Magnesium burns with a bright white flame in the presence of oxygen.",
    "The Pacific Ocean covers more surface area than all of Earth's continents combined.",
    "Penicillin was discovered by Alexander Fleming in 1928 from a stray mould.",
    "The Nile River flows northward through northeastern Africa for over six thousand kilometers.",
    # German null sentences spanning history, geography, science, culture
    "Die Katze sa\u00df am Fenster und beobachtete die Tauben auf dem Dach.",
    "Im Jahr 1492 segelte Christoph Kolumbus \u00fcber den Atlantischen Ozean.",
    "Die Photosynthese wandelt Sonnenlicht in chemische Energie um, die als Glukose gespeichert wird.",
    "Der Eiffelturm wurde 1889 auf dem Marsfeld in Paris fertiggestellt.",
    "Wasser kocht bei einhundert Grad Celsius auf normaler Meeresh\u00f6he.",
    "Wolfgang Amadeus Mozart komponierte mehr als sechshundert Werke in seinem kurzen Leben.",
    "Die Berliner Mauer fiel am neunten November 1989 nach achtundzwanzig Jahren.",
    "Albert Einstein ver\u00f6ffentlichte 1905 die spezielle Relativit\u00e4tstheorie in Bern.",
    "Der Rhein flie\u00dft \u00fcber tausenddreihundert Kilometer von den Alpen bis in die Nordsee.",
    "Die deutsche Wiedervereinigung wurde am dritten Oktober 1990 offiziell vollzogen.",
    "Goethe schrieb den ersten Teil des Faust \u00fcber mehrere Jahrzehnte hinweg.",
    "Magnesium verbrennt mit einer hellen wei\u00dfen Flamme in Gegenwart von Sauerstoff.",
    "Penicillin wurde 1928 von Alexander Fleming durch einen zuf\u00e4lligen Schimmelpilz entdeckt.",
    "Die Zugspitze ist mit zweitausendneunhundertzweiundsechzig Metern der h\u00f6chste Berg Deutschlands.",
    "Bach komponierte das Wohltemperierte Klavier in zwei B\u00e4nden \u00fcber etwa zwanzig Jahre verteilt.",
    "Die Europ\u00e4ische Zentralbank hat ihren Sitz in Frankfurt am Main seit 1998.",
    # French null sentences — history, geography, science, culture
    "Le chat dormait sur le rebord de la fen\u00eatre pendant tout l'apr\u00e8s-midi.",
    "Christophe Colomb a travers\u00e9 l'oc\u00e9an Atlantique en 1492.",
    "La photosynth\u00e8se transforme la lumi\u00e8re du soleil en \u00e9nergie chimique.",
    "La Tour Eiffel a \u00e9t\u00e9 achev\u00e9e en 1889 sur le Champ-de-Mars \u00e0 Paris.",
    "L'eau bout \u00e0 cent degr\u00e9s Celsius au niveau de la mer.",
    "Le Rh\u00f4ne prend sa source en Suisse avant de se jeter en M\u00e9diterran\u00e9e.",
    "Napol\u00e9on a \u00e9t\u00e9 couronn\u00e9 empereur des Fran\u00e7ais en 1804 \u00e0 Notre-Dame.",
    "Marie Curie a re\u00e7u deux prix Nobel, en physique puis en chimie.",
    "La d\u00e9claration des droits de l'homme date de 1789.",
    "Le mont Blanc culmine \u00e0 quatre mille huit cent dix m\u00e8tres d'altitude.",
    "Claude Monet a peint les Nymph\u00e9as \u00e0 Giverny pendant plusieurs d\u00e9cennies.",
    "Le TGV a \u00e9tabli un record mondial \u00e0 cinq cent soixante-quatorze kilom\u00e8tres par heure.",
    # Spanish null sentences
    "El gato dormía en el alféizar de la ventana durante toda la tarde.",
    "Cristóbal Colón cruzó el océano Atlántico en 1492.",
    "La fotosíntesis convierte la luz solar en energía química almacenada en glucosa.",
    "El agua hierve a cien grados Celsius al nivel del mar.",
    "La Sagrada Familia lleva en construcción desde 1882 en Barcelona.",
    "El río Amazonas recorre más de seis mil kilómetros a través de Sudamérica.",
    "Miguel de Cervantes publicó la primera parte del Quijote en 1605.",
    "El Real Madrid fue fundado en 1902 y juega en el estadio Santiago Bernabéu.",
    "La Constitución española de 1978 estableció una monarquía parlamentaria.",
    "La penicilina fue descubierta por Alexander Fleming en 1928.",
    "La expedición de Magallanes-Elcano circunnavegó la Tierra entre 1519 y 1522.",
    "La velocidad de la luz en el vacío es aproximadamente trescientos mil kilómetros por segundo.",
    # Italian null sentences
    "Il gatto dormiva sul davanzale della finestra per tutto il pomeriggio.",
    "Cristoforo Colombo attraversò l'oceano Atlantico nel 1492.",
    "La fotosintesi trasforma la luce solare in energia chimica immagazzinata nel glucosio.",
    "L'acqua bolle a cento gradi Celsius al livello del mare.",
    "Il Colosseo di Roma fu completato intorno all'80 d.C. sotto l'imperatore Tito.",
    "Il fiume Po scorre per circa seicentocinquanta chilometri attraverso l'Italia settentrionale.",
    "Dante Alighieri completò la Divina Commedia poco prima della sua morte nel 1321.",
    "La Repubblica Italiana fu proclamata il due giugno 1946 con un referendum.",
    "Leonardo da Vinci dipinse la Gioconda all'inizio del XVI secolo.",
    "Giuseppe Verdi compose più di venticinque opere liriche durante la sua carriera.",
    "L'Etna è il vulcano attivo più alto d'Europa con oltre tremila metri di altitudine.",
    "L'Unità d'Italia fu proclamata il diciassette marzo 1861.",
)


def default_null_bank_texts() -> List[str]:
    """Return a copy of the default null bank text list used by calibration."""

    return list(DEFAULT_NULL_BANK_TEXTS)


@dataclass
class SupportUnitInput:
    """Normalized support unit used by the groundedness scorer.

    The optional ``source_id``, ``speaker`` and ``timestamp`` fields are the
    K2 structured-premise attribution channel: when set on the input they
    are echoed verbatim into the matching response support unit so callers
    can attribute each surviving claim back to its originating
    speaker/source/turn. ``metadata`` is a free-form dict that flows back
    on the same response unit.
    """

    support_id: str
    text: str
    embeddings: torch.Tensor
    tokens: List[str]
    chunk_id: Optional[Any] = None
    source_mode: str = "chunk_ids"
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    source_id: Optional[str] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), p=2, dim=-1)


@functools.lru_cache(maxsize=131072)
def _strip_marker(token: str) -> str:
    out = token
    for prefix in ("Ġ", "▁"):
        while out.startswith(prefix):
            out = out[len(prefix):]
    while out.startswith("##"):
        out = out[2:]
    return out


@functools.lru_cache(maxsize=131072)
def _is_content_token(token: str) -> bool:
    """Mirror of :func:`token_weights`'s positive cases used for support-side masking.

    A token counts as content-bearing when it is not a special/whitespace marker,
    not pure punctuation, and not in the curated stopword list. Numeric, alphabetic,
    and mixed-content tokens all count as content.

    Cached because support tokens are highly repetitive (typical code corpora see
    <5% unique tokens across 100k+ token banks, per benchmarks on transcripts_v1).
    """

    if token in _SPECIAL_TOKENS:
        return False
    stripped = _strip_marker(token).strip()
    if not stripped:
        return False
    if all(ch in string.punctuation for ch in stripped):
        return False
    if stripped.lower() in _STOPWORDS:
        return False
    return True


def support_content_mask(tokens: Sequence[str]) -> torch.Tensor:
    """Return a boolean mask marking content-bearing support tokens.

    ``True`` means the token is allowed to participate in MaxSim attribution and
    breadth statistics; ``False`` excludes filler such as punctuation, special
    tokens, and curated stopwords from being chosen as the "best supporting"
    token for any response token.
    """

    if not tokens:
        return torch.zeros((0,), dtype=torch.bool)
    return torch.tensor([_is_content_token(token) for token in tokens], dtype=torch.bool)


def token_weights(tokens: Sequence[str]) -> torch.Tensor:
    """Rule-based token weighting that downweights filler and punctuation."""

    weights: List[float] = []
    for token in tokens:
        if token in _SPECIAL_TOKENS:
            weights.append(0.0)
            continue
        stripped = _strip_marker(token).strip()
        if not stripped:
            weights.append(0.0)
            continue
        if all(ch in string.punctuation for ch in stripped):
            weights.append(0.0)
            continue
        if stripped.lower() in _STOPWORDS:
            weights.append(0.0)
            continue
        if any(ch.isdigit() for ch in stripped):
            weights.append(1.5)
            continue
        weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _fallback_tokens(text: str) -> List[str]:
    return _TOKEN_FALLBACK_RE.findall(text)


def _tokens_from_provider_tokenize(output: Any) -> List[str]:
    if isinstance(output, dict):
        return []
    if isinstance(output, str):
        return [output]
    if isinstance(output, (list, tuple)):
        if output and all(isinstance(item, str) for item in output):
            return list(output)
        return []
    try:
        items = list(output)
    except TypeError:
        return []
    if items and all(isinstance(item, str) for item in items):
        return items
    return []


def align_tokens(tokens: Sequence[str], expected_len: int) -> List[str]:
    base = list(tokens)
    if expected_len <= 0:
        return []
    if len(base) > expected_len:
        return base[:expected_len]
    if len(base) < expected_len:
        base.extend(f"tok_{idx}" for idx in range(len(base), expected_len))
    return base


def tokenize_with_offsets(
    provider: Any,
    text: str,
    *,
    expected_len: Optional[int] = None,
    is_query: bool = False,
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """Tokenize ``text`` and return per-token char offsets when available.

    Best-effort: when the provider's tokenizer is a HuggingFace fast tokenizer
    (exposes ``return_offsets_mapping``) the per-token ``(start, end)`` spans
    are returned alongside the token strings. Otherwise the offset list is
    populated with ``None`` for each token. The caller is expected to add a
    base offset (the chunk's start in the original ``response_text``) when
    surfacing offsets for chunked responses.

    Always returns ``len(tokens) == len(offsets) == expected_len`` (after
    alignment when ``expected_len`` is provided), so downstream code can zip
    the two lists without index gymnastics.
    """

    tokens: List[str] = []
    offsets: List[Optional[Tuple[int, int]]] = []

    tokenizer = getattr(provider, "tokenizer", None)
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            offset_mapping = encoded.get("offset_mapping")
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if offset_mapping and isinstance(offset_mapping[0], list) and offset_mapping[0] and isinstance(offset_mapping[0][0], (list, tuple)):
                offset_mapping = offset_mapping[0]
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
            else:
                tokens = [str(item) for item in input_ids]
            if offset_mapping is not None and len(offset_mapping) == len(tokens):
                for span in offset_mapping:
                    if isinstance(span, (list, tuple)) and len(span) == 2:
                        start, end = int(span[0]), int(span[1])
                        if end > start:
                            offsets.append((start, end))
                        else:
                            offsets.append(None)
                    else:
                        offsets.append(None)
        except (TypeError, KeyError):
            tokens = []
            offsets = []
        except Exception:
            tokens = []
            offsets = []

    if not tokens:
        # Fall back to provider.tokenize / fallback path; offsets unavailable.
        tokens = tokenize_text(provider, text, expected_len=None, is_query=is_query)
        offsets = [None] * len(tokens)

    if not offsets or len(offsets) != len(tokens):
        offsets = [None] * len(tokens)

    if expected_len is not None:
        aligned_tokens = align_tokens(tokens, expected_len)
        if len(aligned_tokens) > len(offsets):
            offsets = list(offsets) + [None] * (len(aligned_tokens) - len(offsets))
        elif len(aligned_tokens) < len(offsets):
            offsets = list(offsets[: len(aligned_tokens)])
        tokens = aligned_tokens
    return tokens, offsets


def tokenize_text(
    provider: Any,
    text: str,
    *,
    expected_len: Optional[int] = None,
    is_query: bool = False,
) -> List[str]:
    tokens: List[str] = []

    tokenize_method = getattr(provider, "tokenize", None)
    if callable(tokenize_method):
        try:
            tokens = _tokens_from_provider_tokenize(tokenize_method(text, is_query=is_query))
        except TypeError:
            try:
                tokens = _tokens_from_provider_tokenize(tokenize_method(text))
            except Exception:
                tokens = []
        except Exception:
            tokens = []

    if not tokens and hasattr(provider, "tokenizer"):
        tokenizer = provider.tokenizer
        try:
            encoded = tokenizer(text, add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
            else:
                tokens = [str(item) for item in input_ids]
        except Exception:
            tokens = []

    if not tokens:
        tokens = _fallback_tokens(text)

    if expected_len is not None:
        tokens = align_tokens(tokens, expected_len)
    return tokens


def count_text_tokens(provider: Any, text: str, *, is_query: bool = False) -> int:
    """Count document-side tokens for support-unit packing.

    Prefers a strict ``encoded_token_count(text, is_query=...)`` hook on the
    provider, which lets vLLM-style providers report the exact post-tokenizer
    sequence length (including specials and ``[D]/[Q]`` prefix). Falls back to
    the bare tokenizer when that hook is not available so older providers still
    work.
    """

    stripped = text.strip()
    if not stripped:
        return 0

    encoded_count_method = getattr(provider, "encoded_token_count", None)
    if callable(encoded_count_method):
        try:
            value = encoded_count_method(text, is_query=is_query)
            if isinstance(value, int) and value >= 0:
                return value
        except TypeError:
            try:
                value = encoded_count_method(text)
                if isinstance(value, int) and value >= 0:
                    return value
            except Exception:
                pass
        except Exception:
            pass

    if hasattr(provider, "tokenizer"):
        tokenizer = provider.tokenizer
        try:
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return len(input_ids)
        except Exception:
            pass

    tokenize_method = getattr(provider, "tokenize", None)
    if callable(tokenize_method):
        try:
            tokens = _tokens_from_provider_tokenize(tokenize_method(text, is_query=is_query))
            if tokens:
                return len(tokens)
        except TypeError:
            try:
                tokens = _tokens_from_provider_tokenize(tokenize_method(text))
                if tokens:
                    return len(tokens)
            except Exception:
                pass
        except Exception:
            pass

    return len(_fallback_tokens(text))


def provider_token_limit(provider: Any, *, is_query: bool = False) -> Optional[int]:
    role_attrs = ("query_maxlen", "query_length") if is_query else ("doc_maxlen", "document_length")
    for attr in (*role_attrs, "max_length", "model_max_length"):
        value = getattr(provider, attr, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value

    tokenizer = getattr(provider, "tokenizer", None)
    if tokenizer is None:
        return None
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 1_000_000:
        return value
    return None


def partition_support_units(
    support_units: Sequence[SupportUnitInput],
    *,
    batch_size: int,
) -> List[List[SupportUnitInput]]:
    size = max(1, int(batch_size))
    return [list(support_units[idx : idx + size]) for idx in range(0, len(support_units), size)]


def _to_tensor_list(output: Any, *, expected_items: int) -> List[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        if output.ndim == 1:
            return [output.reshape(1, -1).float()]
        if output.ndim == 2:
            return [output.float()]
        if output.ndim == 3:
            return [output[idx].float() for idx in range(output.shape[0])]
    if isinstance(output, np.ndarray):
        if output.ndim == 1:
            return [torch.from_numpy(output.reshape(1, -1)).float()]
        if output.ndim == 2:
            return [torch.from_numpy(output).float()]
        if output.ndim == 3:
            return [torch.from_numpy(output[idx]).float() for idx in range(output.shape[0])]
    if isinstance(output, list):
        if not output:
            return []
        if len(output) == expected_items and all(
            isinstance(item, (list, np.ndarray, torch.Tensor)) for item in output
        ):
            tensors = []
            for item in output:
                tensor = torch.as_tensor(item, dtype=torch.float32)
                if tensor.ndim == 1:
                    tensor = tensor.reshape(1, -1)
                tensors.append(tensor)
            return tensors
        if expected_items == 1 and output and not isinstance(output[0], list):
            return [torch.as_tensor(output, dtype=torch.float32).reshape(1, -1)]
        if expected_items == 1 and output and isinstance(output[0], list) and (
            not output[0] or not isinstance(output[0][0], list)
        ):
            return [torch.as_tensor(output, dtype=torch.float32)]
        tensors = [torch.as_tensor(item, dtype=torch.float32) for item in output]
        if len(tensors) == expected_items:
            return tensors
        if len(tensors) == 1 and expected_items == 1:
            return tensors
    raise TypeError("Unsupported encoder output shape for groundedness scoring")


def encode_texts(
    provider: Any,
    texts: Sequence[str],
    *,
    is_query: bool,
    prompt_name: Optional[str] = None,
) -> List[torch.Tensor]:
    """Encode texts into multi-vector tensors.

    Prefers batching when supported and falls back to one-by-one encoding.
    """

    items = list(texts)
    if not items:
        return []

    kwargs: Dict[str, Any] = {"is_query": is_query}
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name

    attempts = [dict(kwargs)]
    if "prompt_name" in kwargs:
        attempts.append({"is_query": is_query})
    attempts.append({})

    for candidate_kwargs in attempts:
        try:
            return _to_tensor_list(provider.encode(items, **candidate_kwargs), expected_items=len(items))
        except TypeError:
            continue
        except Exception:
            break

    encoded: List[torch.Tensor] = []
    for text in items:
        tensor: Optional[torch.Tensor] = None
        for candidate_kwargs in attempts:
            try:
                output = provider.encode([text], **candidate_kwargs)
                tensor = _to_tensor_list(output, expected_items=1)[0]
                break
            except TypeError:
                continue
            except Exception:
                tensor = None
                break
        if tensor is None:
            raise TypeError("Provider cannot encode text for groundedness scoring")
        encoded.append(tensor)
    return encoded


def _fallback_segment(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    start = text.find(stripped)
    return [{"text": stripped, "offset_start": start, "offset_end": start + len(stripped)}]


def _trimmed_span(fragment: str, start: int, end: int) -> Optional[Dict[str, Any]]:
    stripped = fragment.strip()
    if not stripped:
        return None
    leading = len(fragment) - len(fragment.lstrip())
    trailing = len(fragment) - len(fragment.rstrip())
    return {
        "text": stripped,
        "offset_start": start + leading,
        "offset_end": end - trailing,
    }


def _paragraph_spans(text: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    cursor = 0
    for part in re.split(r"\n\s*\n", text):
        start = text.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        cursor = end
        span = _trimmed_span(part, start, end)
        if span is not None:
            spans.append(span)
    return spans or _fallback_segment(text)


def _sentence_spans(text: str) -> List[Dict[str, Any]]:
    spans = []
    for match in _SENTENCE_RE.finditer(text):
        span = _trimmed_span(match.group(0), match.start(), match.end())
        if span is not None:
            spans.append(span)
    return spans or _fallback_segment(text)


def _split_oversized_span(
    text: str,
    span: Dict[str, Any],
    *,
    provider: Any,
    chunk_token_budget: int,
) -> List[Dict[str, Any]]:
    """Split a single span whose token count exceeds the budget.

    Long spans (e.g. a sentence-less JSON blob, a paragraph the sentence
    splitter could not break) would otherwise be packed into a single
    chunk that exceeds ``chunk_token_budget``. Downstream encoders (ColBERT
    variants typically cap around 256-512 tokens) would silently truncate
    such chunks and drop the tail tokens from the groundedness signal.

    We greedily cut characters from the front, snap to whitespace, and
    re-measure with the provider tokenizer until each sub-span fits.
    """

    sub_spans: List[Dict[str, Any]] = []
    cursor = int(span["offset_start"])
    end_offset = int(span["offset_end"])
    if end_offset <= cursor:
        return sub_spans
    safety = 0
    while cursor < end_offset and safety < 256:
        safety += 1
        remaining = text[cursor:end_offset]
        remaining_tokens = max(count_text_tokens(provider, remaining, is_query=False), 1)
        if remaining_tokens <= chunk_token_budget:
            sub = _trimmed_span(remaining, cursor, end_offset)
            if sub is not None:
                sub["token_count"] = remaining_tokens
                sub_spans.append(sub)
            break
        ratio = max(1.0, len(remaining) / float(remaining_tokens))
        cut_chars = max(8, int(ratio * chunk_token_budget * 0.9))
        cut_at = min(end_offset, cursor + cut_chars)
        snap = text.rfind(" ", cursor + 1, cut_at + 1)
        if snap > cursor + 8:
            cut_at = snap
        sub_text = text[cursor:cut_at]
        sub_tokens = max(count_text_tokens(provider, sub_text, is_query=False), 1)
        guard = 0
        while sub_tokens > chunk_token_budget and (cut_at - cursor) > 8 and guard < 16:
            guard += 1
            cut_at = cursor + max(8, int((cut_at - cursor) * 0.85))
            sub_text = text[cursor:cut_at]
            sub_tokens = max(count_text_tokens(provider, sub_text, is_query=False), 1)
        sub = _trimmed_span(sub_text, cursor, cut_at)
        if sub is not None:
            sub["token_count"] = sub_tokens
            sub_spans.append(sub)
        if cut_at <= cursor:
            break
        cursor = cut_at
    return sub_spans


def _pack_sentence_spans(
    text: str,
    spans: Sequence[Dict[str, Any]],
    *,
    provider: Any,
    chunk_token_budget: int,
) -> List[Dict[str, Any]]:
    if provider is None:
        raise ValueError("sentence_packed segmentation requires a provider for token counting")
    if chunk_token_budget <= 0:
        raise ValueError("chunk_token_budget must be positive")

    packed: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        start = int(current[0]["offset_start"])
        end = int(current[-1]["offset_end"])
        packed.append(
            {
                "text": text[start:end].strip(),
                "offset_start": start,
                "offset_end": end,
                "token_count": current_tokens,
            }
        )
        current = []
        current_tokens = 0

    for span in spans:
        span_tokens = max(count_text_tokens(provider, span["text"], is_query=False), 1)
        # Hard guard: a single oversized span (e.g. a JSON blob, a long
        # sentence the splitter could not break) must not be packed whole;
        # downstream encoders silently truncate at their model_max_length
        # and drop the tail tokens from the groundedness matrix.
        if span_tokens > chunk_token_budget:
            if current:
                flush()
            for sub in _split_oversized_span(
                text, span, provider=provider, chunk_token_budget=chunk_token_budget
            ):
                sub_tokens = int(sub.get("token_count") or 0) or chunk_token_budget
                packed.append(
                    {
                        "text": sub["text"],
                        "offset_start": int(sub["offset_start"]),
                        "offset_end": int(sub["offset_end"]),
                        "token_count": sub_tokens,
                    }
                )
            continue
        if current and current_tokens + span_tokens > chunk_token_budget:
            flush()
        current.append(span)
        current_tokens += span_tokens
        if current_tokens >= chunk_token_budget:
            flush()

    flush()
    return packed or _fallback_segment(text)


def segment_text(
    text: str,
    mode: str,
    *,
    provider: Any | None = None,
    chunk_token_budget: int = _DEFAULT_CHUNK_TOKEN_BUDGET,
) -> List[Dict[str, Any]]:
    if not text.strip():
        return []

    if mode == "paragraph":
        return _paragraph_spans(text)
    if mode == "sentence_packed":
        return _pack_sentence_spans(
            text,
            _sentence_spans(text),
            provider=provider,
            chunk_token_budget=chunk_token_budget,
        )
    return _sentence_spans(text)


_DEFAULT_COVERAGE_THRESHOLD = 0.5


def compute_unit_coverage(
    reverse_context_unit_values: torch.Tensor,
    *,
    threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> Dict[str, Any]:
    """Per-support-unit coverage scorer (retrieval-efficiency observability).

    Given the per-(response_token, support_unit) similarity matrix
    ``reverse_context_unit_values`` (shape ``(R, U)``) — already computed by
    the scoring kernel as the max similarity of each response token to the
    tokens of each support unit — this helper reduces along the response
    axis to produce one ``coverage_score`` per support unit:

        coverage_score[u] = max over response tokens t of m[t, u]

    A unit is considered "used" when its coverage score crosses
    ``threshold`` (inclusive ``>=`` semantics; default 0.5 — a conservative
    cutoff on raw cosine similarity that separates "weak match" from
    "strong match" for ColBERT-style normalized embeddings). The global
    ``coverage_ratio`` is the fraction of units with ``used == True`` and
    is the headline retrieval-efficiency observability metric: a ratio of
    0.4 means 60% of the retrieved chunks contributed nothing strong to
    the response and the retriever is pulling too much dead weight.

    Robustness guarantees:

    - 0-token support units (whose per-unit maxima are ``-inf`` from the
      scorer) are reported with ``coverage_score = 0.0`` and
      ``used = False`` regardless of the threshold value. This keeps the
      visible score within the ``[0, 1]`` similarity range callers expect
      and prevents a degenerate unit from ever counting as "used".
    - Non-finite values that arise from numerical instability (NaN, Inf)
      are treated identically to 0-token units.
    - Empty matrices (no units, or no response tokens) return a zero-
      filled payload with ``coverage_ratio = 0.0`` instead of raising.

    The function is pure (no I/O, no provider state) and runs in O(R * U)
    elementary ops, the same order as the existing per-unit reductions, so
    it adds no measurable latency to the scoring path.
    """

    if reverse_context_unit_values.numel() == 0 or reverse_context_unit_values.shape[1] == 0:
        return {
            "per_unit_max": [],
            "used_mask": [],
            "used_count": 0,
            "total_count": 0,
            "coverage_ratio": 0.0,
            "threshold": float(threshold),
        }
    per_unit_max = reverse_context_unit_values.max(dim=0).values
    finite_mask = torch.isfinite(per_unit_max)
    # Visible coverage_score is clamped to 0.0 when a unit is degenerate
    # (0 tokens → -inf) or numerically unstable (NaN). Returning a
    # negative number here would leak outside the documented [0, 1]
    # similarity range and confuse callers that derive UI states from
    # coverage_score directly.
    safe_per_unit_max = torch.where(
        finite_mask,
        per_unit_max,
        torch.zeros_like(per_unit_max),
    )
    # `used` requires the value to be both finite AND above threshold so a
    # 0-token unit cannot accidentally flip to used=True when the operator
    # sets ``threshold = 0.0`` (a legal call meaning "report attribution
    # alongside the trivial 'every unit is used' baseline").
    above_threshold = safe_per_unit_max >= float(threshold)
    used_mask = torch.logical_and(finite_mask, above_threshold)
    used_count = int(used_mask.sum().item())
    total_count = int(safe_per_unit_max.shape[0])
    return {
        "per_unit_max": safe_per_unit_max.detach().cpu().tolist(),
        "used_mask": used_mask.detach().cpu().tolist(),
        "used_count": used_count,
        "total_count": total_count,
        "coverage_ratio": float(used_count) / float(total_count) if total_count > 0 else 0.0,
        "threshold": float(threshold),
    }


def _support_unit_maxima(
    similarity: torch.Tensor,
    support_tokens_nested: Sequence[Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_values: List[torch.Tensor] = []
    unit_indices: List[torch.Tensor] = []
    offset = 0
    for tokens in support_tokens_nested:
        token_count = len(tokens)
        if token_count <= 0:
            unit_values.append(torch.full((similarity.shape[0],), float("-inf"), dtype=similarity.dtype, device=similarity.device))
            unit_indices.append(torch.full((similarity.shape[0],), -1, dtype=torch.long, device=similarity.device))
            continue
        unit_slice = similarity[:, offset : offset + token_count]
        values, indices = unit_slice.max(dim=1)
        unit_values.append(values)
        unit_indices.append(indices.to(torch.long))
        offset += token_count

    if not unit_values:
        empty = torch.empty((similarity.shape[0], 0), dtype=similarity.dtype, device=similarity.device)
        empty_idx = torch.empty((similarity.shape[0], 0), dtype=torch.long, device=similarity.device)
        return empty, empty_idx
    return torch.stack(unit_values, dim=1), torch.stack(unit_indices, dim=1)


def _normalize_text_for_signature(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalize_text_for_verbatim_support(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace('"', " ").replace("“", " ").replace("”", " ")
    normalized = normalized.replace("‘", " ").replace("’", " ").replace("'", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


_LEXICAL_TOKEN_RE = re.compile(r"[\w\u00C0-\u017F]{2,}", re.UNICODE)

# Cross-lingual negation cues. Counts are compared between response and support
# so a response that injects a negation where support has none (or strips one
# where support has one) refuses the lexical rescue, even when surface token
# overlap is high.
_NEGATION_TOKENS = {
    # English
    "not", "no", "never", "none", "cannot", "nor", "without",
    # French (conservative subset — 'ne' and 'pas' are most common cues)
    "pas", "non", "aucun", "aucune", "aucuns", "aucunes", "jamais", "sans",
    # German
    "nicht", "kein", "keine", "keinen", "keiner", "keines", "nie", "ohne",
}


def _lexical_token_set(text: str) -> set[str]:
    """Return a lowercase token set used by the near-verbatim Jaccard check."""

    normalized = _normalize_text_for_verbatim_support(text)
    if not normalized:
        return set()
    return {tok for tok in _LEXICAL_TOKEN_RE.findall(normalized)}


def _negation_count(text: str) -> int:
    normalized = _normalize_text_for_verbatim_support(text)
    if not normalized:
        return 0
    return sum(1 for tok in _LEXICAL_TOKEN_RE.findall(normalized) if tok in _NEGATION_TOKENS)


def _verbatim_support_floor(
    response_text: Optional[str],
    support_units: Sequence[Any],
) -> Optional[float]:
    """Return a high-confidence floor when response sentences are copied from support.

    Some multilingual NLI models under-score exact self-entailment for legal
    or enterprise prose. When the response is materially verbatim or
    near-verbatim evidence text we treat it as grounded even if the NLI
    margin is conservative. This floor is narrow:

    * every substantial response sentence is either present in support after
      quote/whitespace normalization (exact match), **or** its lowercase
      token set has Jaccard overlap >= 0.85 with the concatenated support
      tokens (near-verbatim — tolerates minor accent/OCR drift on
      multilingual FR/DE prose).
    """

    response = _normalize_text_for_verbatim_support(response_text or "")
    if not response:
        return None
    support_parts: list[str] = []
    for unit in support_units:
        if isinstance(unit, dict):
            text = unit.get("text") or ""
        else:
            text = getattr(unit, "text", "")
        if text:
            support_parts.append(str(text))
    support_joined = " ".join(support_parts)
    support = _normalize_text_for_verbatim_support(support_joined)
    if not support:
        return None
    support_tokens = _lexical_token_set(support_joined)

    raw_sentences = [
        _normalize_text_for_verbatim_support(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", response_text or "")
    ]
    sentences = [sentence for sentence in raw_sentences if len(sentence) >= 40]
    if not sentences:
        return None

    def _sentence_grounded(sentence: str) -> bool:
        if sentence in support:
            return True
        tokens = {tok for tok in _LEXICAL_TOKEN_RE.findall(sentence)}
        if not tokens or not support_tokens:
            return False
        overlap = len(tokens & support_tokens)
        union = len(tokens | support_tokens)
        if union == 0:
            return False
        jaccard = overlap / float(union)
        # For short sentences use precision against support instead of full
        # Jaccard (support is much larger so union dwarfs the overlap).
        precision = overlap / float(len(tokens)) if tokens else 0.0
        return jaccard >= 0.85 or precision >= 0.92

    if all(_sentence_grounded(sentence) for sentence in sentences):
        return 0.95
    return None


def _lexical_rescue_floor(
    response_text: Optional[str],
    support_units: Sequence[Any],
    *,
    reverse_context_calibrated: Optional[float],
    literal_guarded: Optional[float],
    nli_aggregate: Optional[float],
) -> Optional[float]:
    """Rescue for NLI under-confidence on multilingual / legal prose.

    Fires only when **all** of the following hold:

    1. Two independent lexical channels agree the response is grounded
       (``reverse_context_calibrated >= 0.80`` AND ``literal_guarded >= 0.80``).
    2. NLI is below 0.80 but not strongly contradicting (``>= 0.50``).
       A very low NLI score is treated as a real semantic mismatch signal
       that must NOT be overridden by lexical overlap alone — this blocks
       the classic "response is a copy of support with a negation
       inserted" failure mode.
    3. Response-side token precision against support is at least 0.70,
       so incidental keyword overlap cannot trigger the rescue.
    4. The response and support use negation symmetrically: a response
       that contains more negation tokens than its supporting evidence
       (or fewer) refuses the rescue, even if surface overlap is high.

    Returns the rescued floor (0.85) or ``None`` when any safeguard
    blocks the lift. The caller merges this with
    :func:`_verbatim_support_floor`.
    """

    if reverse_context_calibrated is None or literal_guarded is None:
        return None
    if float(reverse_context_calibrated) < 0.80 or float(literal_guarded) < 0.80:
        return None
    if nli_aggregate is not None:
        nli_value = float(nli_aggregate)
        if nli_value >= 0.80:
            return None
        # Strong NLI contradiction: likely a real semantic mismatch. Do
        # NOT override with lexical overlap.
        if nli_value < 0.50:
            return None
    response_tokens = _lexical_token_set(response_text or "")
    if len(response_tokens) < 6:
        return None
    support_parts: list[str] = []
    for unit in support_units:
        if isinstance(unit, dict):
            text = unit.get("text") or ""
        else:
            text = getattr(unit, "text", "")
        if text:
            support_parts.append(str(text))
    support_joined = " ".join(support_parts)
    support_tokens = _lexical_token_set(support_joined)
    if not support_tokens:
        return None
    overlap = len(response_tokens & support_tokens)
    precision = overlap / float(len(response_tokens))
    if precision < 0.70:
        return None
    # Asymmetric negation: response injects or removes a negation cue
    # that is not balanced by a matching cue on the support side. The
    # check uses a multilingual EN/FR/DE negation stop-list so it is
    # cheap and language-independent.
    response_negations = _negation_count(response_text or "")
    support_negations = _negation_count(support_joined)
    if response_negations != support_negations:
        # If either side has zero negations and the other has >=1, the
        # response changed polarity — refuse the rescue.
        if min(response_negations, support_negations) == 0:
            return None
        # Otherwise allow up to a one-token drift (natural paraphrase).
        if abs(response_negations - support_negations) > 1:
            return None
    return 0.85


# Hedge/uncertainty cues shared by the ambiguous-variant benchmark and the
# scorer-side epistemic-hedge gate. Kept deliberately multilingual because
# Veracier prose arrives in French/German/English.
EPISTEMIC_HEDGE_CUES: Tuple[str, ...] = (
    # English
    "unclear",
    "not established",
    "does not show",
    "not clear",
    "insufficient",
    "missing",
    "no evidence",
    "no confirmation",
    "cannot establish",
    "cannot confirm",
    "cannot be confirmed",
    "cannot be established",
    "unable to confirm",
    "further evidence is required",
    # French
    "ne permet pas",
    "pas clair",
    "n'établit pas",
    "n’etablit pas",
    "n'est pas établi",
    "n'est pas établie",
    "n'est pas etablie",
    "n'est pas etabli",
    "n’est pas établi",
    "n’est pas établie",
    "n’est pas etablie",
    "il n'est pas",
    "reste à confirmer",
    "reste a confirmer",
    "je ne dispose pas",
    "je ne peux pas",
    "insuffisant",
    "incertain",
    "incertitude",
    "manquant",
    "non clarifié",
    "non clarifie",
    "insuffisamment documenté",
    "insuffisamment documente",
    "qui n'est pas fourni",
    "demanderait un recoupement",
    # German
    "nicht klar",
    "nicht belegt",
    "allerdings ist nicht",
    "bleibt unklar",
    "unvollständig",
    "unvollstaendig",
    "unbekannt",
    "keine angabe",
    # Spanish
    "no está claro",
    "no esta claro",
    "no queda claro",
    "no se puede confirmar",
    "no se ha establecido",
    "no consta",
    "falta evidencia",
    "falta informacion",
    "falta información",
    "sin evidencia",
    "insuficiente",
    "incierto",
    "no se establece",
    # Italian
    "non è chiaro",
    "non e chiaro",
    "non risulta",
    "non si può confermare",
    "non si puo confermare",
    "non è stabilito",
    "non e stabilito",
    "insufficiente",
    "incerto",
    "mancano prove",
    "mancano evidenze",
)


def _normalize_hedge_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).lower()


def _epistemic_hedge_gate(
    *,
    groundedness_v2: Optional[float],
    claim_records: Optional[Sequence[Dict[str, Any]]],
    effective_stratum: Optional[str],
    risk_band_policy: "RiskBandPolicy",
    entailment_unsupported_ceiling: float = 0.40,
    entailment_supported_floor: float = 0.60,
    contradiction_ceiling: float = 0.60,
    cap_ratio: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Cap/floor ambiguous prose answers into the amber band.

    The gate fires only on the ``rag_prose`` stratum and only when the answer
    contains a hedge signal **and** at least one other claim is clearly
    supported. It leaves structured/tabular lanes and classic enterprise
    strata untouched.

    Returns ``None`` when the gate cannot fire (wrong stratum, missing claim
    records, or the answer lacks the mixed hedge-and-support signature).
    Otherwise returns a diagnostic dict::

        {
            "fired": bool,
            "applied": bool,           # score actually clamped
            "cap": float,
            "floor": float,
            "claim_indices": list[int],
            "original_score": float,
            "adjusted_score": float,
        }
    """

    if effective_stratum != "rag_prose":
        return None
    if groundedness_v2 is None or not claim_records:
        return None

    hedge_indices: List[int] = []
    supported_indices: List[int] = []
    for record in claim_records:
        if not isinstance(record, dict):
            continue
        if record.get("skipped"):
            continue
        text = _normalize_hedge_text(record.get("text"))
        if not text:
            continue
        entailment = float(record.get("entailment") or 0.0)
        contradiction = float(record.get("contradiction") or 0.0)
        claim_index = int(record.get("index", 0))
        has_cue = any(cue in text for cue in EPISTEMIC_HEDGE_CUES)
        if (
            has_cue
            and entailment < entailment_unsupported_ceiling
            and contradiction < contradiction_ceiling
        ):
            hedge_indices.append(claim_index)
            continue
        if entailment >= entailment_supported_floor:
            supported_indices.append(claim_index)

    if not hedge_indices or not supported_indices:
        return None

    thresholds = risk_band_policy.threshold_for("rag_prose")
    amber_min = float(thresholds["amber_min"])
    green_min = float(thresholds["green_min"])
    if green_min <= amber_min:
        # Degenerate threshold config; refuse to guess.
        return None
    cap = amber_min + (green_min - amber_min) * cap_ratio
    floor = amber_min
    score = float(groundedness_v2)
    adjusted = max(min(score, cap), floor)
    fired = True
    applied = adjusted != score
    return {
        "fired": fired,
        "applied": bool(applied),
        "cap": float(cap),
        "floor": float(floor),
        "claim_indices": hedge_indices,
        "original_score": float(score),
        "adjusted_score": float(adjusted),
    }


def support_unit_signature(unit: SupportUnitInput) -> str:
    """Return a stable signature for de-duplicating near-duplicate support units.

    Preference order: explicit ``chunk_id`` (string), else a SHA1 of the
    normalized support text. Empty texts hash to a stable bucket, so a request
    sending a single empty placeholder twice is still treated as one source.
    """

    if unit.chunk_id is not None:
        return f"chunk:{unit.chunk_id}"
    normalized = _normalize_text_for_signature(unit.text)
    if not normalized:
        return "text:__empty__"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"text:{digest}"


@dataclass(frozen=True)
class UsageClassifierThresholds:
    """Threshold bundle for precision-first support-unit usage classification."""

    strong_coverage_min: float = 0.62
    unused_coverage_max: float = 0.22
    attribution_score_min: float = 0.34
    support_token_ratio_min: float = 0.2
    support_token_score_threshold: float = 0.45
    nli_entailment_min: float = 0.55
    nli_score_min: float = 0.15
    redundancy_overlap_min: float = 0.75
    redundancy_jaccard_min: float = 0.55
    redundancy_centroid_cosine_min: float = 0.92
    redundancy_centroid_overlap_floor: float = 0.4


_DEFAULT_USAGE_THRESHOLDS = UsageClassifierThresholds()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _usage_content_token_set_raw(tokens: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        if not _is_content_token(token):
            continue
        normalized = _strip_marker(str(token)).strip().lower()
        if normalized:
            out.add(normalized)
    return out


def _usage_content_token_set(unit_or_tokens: Any) -> set[str]:
    """Memoized wrapper.

    Accepts either a ``SupportUnitInput`` (preferred, allows per-unit caching)
    or a raw token sequence (legacy path).
    """

    if isinstance(unit_or_tokens, SupportUnitInput):
        cached = getattr(unit_or_tokens, "_cached_token_set", None)
        if cached is not None:
            return cached
        computed = _usage_content_token_set_raw(unit_or_tokens.tokens)
        try:
            object.__setattr__(unit_or_tokens, "_cached_token_set", computed)
        except AttributeError:  # frozen dataclass or slotted fallback
            pass
        return computed
    return _usage_content_token_set_raw(unit_or_tokens)


def _usage_unit_centroid(unit: SupportUnitInput) -> Optional[torch.Tensor]:
    cached = getattr(unit, "_cached_centroid", _CENTROID_SENTINEL)
    if cached is not _CENTROID_SENTINEL:
        return cached
    embeddings = unit.embeddings.float()
    if embeddings.numel() == 0:
        result: Optional[torch.Tensor] = None
    else:
        token_mask = support_content_mask(unit.tokens)
        if token_mask.numel() == int(embeddings.shape[0]) and bool(token_mask.any().item()):
            embeddings = embeddings[token_mask]
        if embeddings.numel() == 0:
            result = None
        else:
            centroid = embeddings.mean(dim=0, keepdim=True)
            if not bool(torch.isfinite(centroid).all().item()):
                result = None
            else:
                result = _normalize(centroid)[0]
    try:
        object.__setattr__(unit, "_cached_centroid", result)
    except AttributeError:  # pragma: no cover - frozen slot fallback
        pass
    return result


_CENTROID_SENTINEL: Any = object()


def _usage_redundancy_similarity(
    *,
    signature_a: str,
    signature_b: str,
    tokens_a: set[str],
    tokens_b: set[str],
    centroid_a: Optional[torch.Tensor],
    centroid_b: Optional[torch.Tensor],
    thresholds: UsageClassifierThresholds,
) -> float:
    if signature_a == signature_b:
        return 1.0
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b)
        overlap_coeff = overlap / float(max(1, min(len(tokens_a), len(tokens_b))))
        jaccard = overlap / float(max(1, len(tokens_a | tokens_b)))
    else:
        overlap_coeff = 0.0
        jaccard = 0.0
    centroid_cosine = 0.0
    if centroid_a is not None and centroid_b is not None:
        centroid_cosine = max(0.0, float(torch.dot(centroid_a, centroid_b).item()))

    if overlap_coeff >= thresholds.redundancy_overlap_min:
        return max(overlap_coeff, jaccard, centroid_cosine)
    if jaccard >= thresholds.redundancy_jaccard_min:
        return max(jaccard, centroid_cosine)
    if (
        centroid_cosine >= thresholds.redundancy_centroid_cosine_min
        and overlap_coeff >= thresholds.redundancy_centroid_overlap_floor
    ):
        return max(centroid_cosine, overlap_coeff)
    return 0.0


def _compute_redundancy_matrix(
    signatures: Sequence[str],
    token_sets: Sequence[set[str]],
    centroids: Sequence[Optional[torch.Tensor]],
    thresholds: UsageClassifierThresholds,
) -> torch.Tensor:
    """Vectorised pairwise redundancy similarity matrix.

    The returned ``(U, U)`` float32 tensor is mathematically identical to the
    per-pair scalar ``_usage_redundancy_similarity``: same gating tiers, same
    priority order, same token-overlap / Jaccard / centroid-cosine semantics,
    and the same signature-equality short circuit. The diagonal is set to
    ``1.0`` (identical-to-self); the consumer is expected to exclude the
    diagonal when taking a per-row max.
    """

    u = len(signatures)
    if u == 0:
        return torch.zeros((0, 0), dtype=torch.float32)

    vocab: Dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    v_size = len(vocab)
    if v_size > 0:
        b_matrix = torch.zeros((u, v_size), dtype=torch.float32)
        for i, tokens in enumerate(token_sets):
            if not tokens:
                continue
            idx = torch.tensor(
                [vocab[token] for token in tokens], dtype=torch.long
            )
            b_matrix[i, idx] = 1.0
        overlap = b_matrix @ b_matrix.T
        sizes = b_matrix.sum(dim=1)
        size_rows = sizes.unsqueeze(1)
        size_cols = sizes.unsqueeze(0)
        both_have_tokens = (size_rows > 0) & (size_cols > 0)
        min_sizes = torch.minimum(size_rows, size_cols).clamp(min=1.0)
        union = (size_rows + size_cols - overlap).clamp(min=1.0)
        zero_uu = torch.zeros_like(overlap)
        overlap_coeff = torch.where(both_have_tokens, overlap / min_sizes, zero_uu)
        jaccard = torch.where(both_have_tokens, overlap / union, zero_uu)
    else:
        overlap_coeff = torch.zeros((u, u), dtype=torch.float32)
        jaccard = torch.zeros((u, u), dtype=torch.float32)

    centroid_cosine = torch.zeros((u, u), dtype=torch.float32)
    centroid_dim: Optional[int] = None
    for centroid in centroids:
        if centroid is not None and centroid.numel() > 0:
            centroid_dim = int(centroid.shape[-1])
            break
    if centroid_dim is not None:
        c_matrix = torch.zeros((u, centroid_dim), dtype=torch.float32)
        centroid_valid = torch.zeros(u, dtype=torch.bool)
        for i, centroid in enumerate(centroids):
            if centroid is None or centroid.numel() == 0:
                continue
            if int(centroid.shape[-1]) != centroid_dim:
                continue
            c_matrix[i] = centroid.detach().to(dtype=torch.float32).cpu().view(-1)
            centroid_valid[i] = True
        if bool(centroid_valid.any().item()):
            raw_cosine = (c_matrix @ c_matrix.T).clamp(min=0.0)
            both_valid = centroid_valid.unsqueeze(1) & centroid_valid.unsqueeze(0)
            centroid_cosine = torch.where(
                both_valid, raw_cosine, torch.zeros_like(raw_cosine)
            )

    pair_matrix = torch.zeros((u, u), dtype=torch.float32)

    tier1_mask = overlap_coeff >= thresholds.redundancy_overlap_min
    tier1_val = torch.maximum(torch.maximum(overlap_coeff, jaccard), centroid_cosine)
    pair_matrix = torch.where(tier1_mask, tier1_val, pair_matrix)

    tier2_mask = (jaccard >= thresholds.redundancy_jaccard_min) & (~tier1_mask)
    tier2_val = torch.maximum(jaccard, centroid_cosine)
    pair_matrix = torch.where(tier2_mask, tier2_val, pair_matrix)

    tier3_mask = (
        (centroid_cosine >= thresholds.redundancy_centroid_cosine_min)
        & (overlap_coeff >= thresholds.redundancy_centroid_overlap_floor)
        & (~tier1_mask)
        & (~tier2_mask)
    )
    tier3_val = torch.maximum(centroid_cosine, overlap_coeff)
    pair_matrix = torch.where(tier3_mask, tier3_val, pair_matrix)

    signature_buckets: Dict[str, List[int]] = {}
    for idx, signature in enumerate(signatures):
        signature_buckets.setdefault(signature, []).append(idx)
    for bucket in signature_buckets.values():
        if len(bucket) < 2:
            if bucket:
                bucket_idx = bucket[0]
                pair_matrix[bucket_idx, bucket_idx] = 1.0
            continue
        group = torch.tensor(bucket, dtype=torch.long)
        pair_matrix[group.unsqueeze(1), group.unsqueeze(0)] = 1.0

    return pair_matrix


def _usage_support_indices_for_record(
    record: Dict[str, Any],
    *,
    support_id_to_index: Dict[str, int],
    support_count: int,
) -> List[int]:
    indices: List[int] = []
    seen: set[int] = set()
    for raw_index in record.get("support_unit_indices") or []:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if not (0 <= index < support_count) or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    if indices:
        return indices
    for support_id in record.get("support_ids") or []:
        index = support_id_to_index.get(str(support_id))
        if index is None or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def _collect_usage_nli_stats(
    claim_records: Optional[Sequence[Dict[str, Any]]],
    *,
    support_id_to_index: Dict[str, int],
    support_count: int,
    thresholds: UsageClassifierThresholds,
) -> List[Dict[str, float | int]]:
    stats: List[Dict[str, float | int]] = [
        {
            "evidence_count": 0,
            "positive_count": 0,
            "entailment_max": 0.0,
            "score_max": 0.0,
        }
        for _ in range(support_count)
    ]
    if not claim_records:
        return stats

    for claim_record in claim_records:
        if not isinstance(claim_record, dict):
            continue
        atom_records = [
            atom
            for atom in (claim_record.get("atoms") or [])
            if isinstance(atom, dict)
        ]
        record_candidates = atom_records if atom_records else [claim_record]
        for record in record_candidates:
            if bool(record.get("skipped")):
                continue
            indices = _usage_support_indices_for_record(
                record,
                support_id_to_index=support_id_to_index,
                support_count=support_count,
            )
            if not indices:
                continue
            entailment = float(record.get("entailment") or 0.0)
            score = float(record.get("score") or 0.0)
            is_positive = (
                entailment >= thresholds.nli_entailment_min
                or score >= thresholds.nli_score_min
            )
            for index in indices:
                stat = stats[index]
                stat["evidence_count"] = int(stat["evidence_count"]) + 1
                stat["entailment_max"] = max(float(stat["entailment_max"]), entailment)
                stat["score_max"] = max(float(stat["score_max"]), score)
                if is_positive:
                    stat["positive_count"] = int(stat["positive_count"]) + 1
    return stats


def apply_support_unit_usage_classification(
    *,
    support_units_payload: List[Dict[str, Any]],
    support_inputs: Sequence[SupportUnitInput],
    coverage_threshold: float,
    claim_records: Optional[Sequence[Dict[str, Any]]] = None,
    thresholds: UsageClassifierThresholds = _DEFAULT_USAGE_THRESHOLDS,
) -> Dict[str, float | int]:
    """Classify support units into precision-first used/unused/uncertain."""

    if not support_units_payload or not support_inputs:
        return {
            "support_units_usage_used": 0,
            "support_units_unused": 0,
            "support_units_uncertain": 0,
            "context_usage_ratio": 0.0,
            "context_unused_ratio": 0.0,
            "context_uncertain_ratio": 0.0,
        }

    support_id_to_index = {
        str(payload.get("support_id")): int(payload.get("index", idx))
        for idx, payload in enumerate(support_units_payload)
    }
    nli_stats = _collect_usage_nli_stats(
        claim_records,
        support_id_to_index=support_id_to_index,
        support_count=len(support_units_payload),
        thresholds=thresholds,
    )

    signatures = [support_unit_signature(unit) for unit in support_inputs]
    token_sets = [_usage_content_token_set(unit) for unit in support_inputs]
    centroids = [_usage_unit_centroid(unit) for unit in support_inputs]

    strong_coverage_min = max(
        thresholds.strong_coverage_min,
        min(0.85, float(coverage_threshold) + 0.08),
    )
    unused_coverage_max = min(
        thresholds.unused_coverage_max,
        max(0.05, float(coverage_threshold) - 0.18),
    )
    support_token_score_threshold = max(
        0.25,
        min(float(coverage_threshold), thresholds.support_token_score_threshold),
    )

    provisional_used: List[bool] = []
    used_confidences: List[float] = []
    usage_signals: List[Dict[str, Any]] = []
    for idx, payload in enumerate(support_units_payload):
        token_count = int(payload.get("token_count") or len(payload.get("tokens") or []))
        token_count = max(token_count, 1)
        token_scores = [float(score) for score in (payload.get("token_scores") or [])]
        support_token_hits = sum(
            1 for score in token_scores if score >= support_token_score_threshold
        )
        support_token_ratio = (
            float(support_token_hits) / float(token_count)
            if token_count > 0
            else 0.0
        )
        coverage_score = float(payload.get("coverage_score") or 0.0)
        attribution_score = float(payload.get("score") or 0.0)
        matched_response_tokens = int(payload.get("matched_response_tokens") or 0)
        nli_stat = nli_stats[idx] if idx < len(nli_stats) else {
            "evidence_count": 0,
            "positive_count": 0,
            "entailment_max": 0.0,
            "score_max": 0.0,
        }
        nli_positive = int(nli_stat["positive_count"]) > 0
        strong_coverage = coverage_score >= strong_coverage_min
        dense_local_use = support_token_ratio >= thresholds.support_token_ratio_min
        direct_attribution = matched_response_tokens > 0 and dense_local_use
        coverage_only_positive = (
            strong_coverage
            and matched_response_tokens == 0
            and not nli_positive
        )
        provisional = bool(
            direct_attribution
            or coverage_only_positive
            or nli_positive
        )
        provisional_used.append(provisional)

        coverage_conf = _clamp01(
            (coverage_score - strong_coverage_min)
            / max(1e-6, 1.0 - strong_coverage_min)
        )
        attribution_conf = _clamp01(
            max(attribution_score, support_token_ratio)
        ) if matched_response_tokens > 0 else 0.0
        nli_conf = _clamp01(
            max(
                float(nli_stat["entailment_max"]),
                0.5 + (0.5 * float(nli_stat["score_max"])),
            )
        ) if nli_positive else 0.0
        used_confidence = max(coverage_conf, attribution_conf, nli_conf)
        used_confidences.append(used_confidence)
        usage_signals.append(
            {
                "coverage_score": coverage_score,
                "attribution_score": attribution_score,
                "matched_response_tokens": matched_response_tokens,
                "coverage_only_positive": coverage_only_positive,
                "dense_local_use": dense_local_use,
                "support_token_ratio": support_token_ratio,
                "nli_positive": nli_positive,
                "nli_positive_count": int(nli_stat["positive_count"]),
                "nli_entailment_max": float(nli_stat["entailment_max"]),
                "nli_score_max": float(nli_stat["score_max"]),
            }
        )

    usage_used_count = 0
    unused_count = 0
    uncertain_count = 0

    total_units = len(support_units_payload)
    if total_units >= 2:
        redundancy_matrix = _compute_redundancy_matrix(
            signatures=signatures,
            token_sets=token_sets,
            centroids=centroids,
            thresholds=thresholds,
        )
        redundancy_matrix = redundancy_matrix.clone()
        if redundancy_matrix.numel() > 0:
            redundancy_matrix.fill_diagonal_(0.0)
        used_mask_tensor = torch.tensor(provisional_used, dtype=torch.bool)
        any_used = bool(used_mask_tensor.any().item())
    else:
        redundancy_matrix = None
        used_mask_tensor = None
        any_used = False

    for idx, payload in enumerate(support_units_payload):
        signals = usage_signals[idx]
        coverage_score = float(signals["coverage_score"])
        matched_response_tokens = int(signals["matched_response_tokens"])
        attribution_score = float(signals["attribution_score"])
        support_token_ratio = float(signals["support_token_ratio"])
        nli_positive = bool(signals["nli_positive"])
        coverage_only_positive = bool(signals["coverage_only_positive"])

        redundancy_similarity = 0.0
        if (
            ((not provisional_used[idx]) or coverage_only_positive)
            and redundancy_matrix is not None
            and used_mask_tensor is not None
            and any_used
        ):
            row = redundancy_matrix[idx]
            row_mask = used_mask_tensor.clone()
            row_mask[idx] = False
            if bool(row_mask.any().item()):
                selected = row[row_mask]
                if selected.numel() > 0:
                    redundancy_similarity = float(selected.max().item())

        coverage_absence = _clamp01(
            (unused_coverage_max - coverage_score)
            / max(1e-6, unused_coverage_max)
        )
        redundancy_clear = _clamp01(1.0 - redundancy_similarity)
        attribution_clear = 1.0 if matched_response_tokens == 0 else 0.0
        nli_clear = 1.0 if not nli_positive else 0.0
        unused_confidence = (
            coverage_absence
            + redundancy_clear
            + attribution_clear
            + nli_clear
        ) / 4.0

        if provisional_used[idx]:
            if coverage_only_positive and redundancy_similarity > 1e-6:
                usage_state = "uncertain"
                usage_confidence = 0.5 * max(
                    used_confidences[idx],
                    _clamp01(redundancy_similarity),
                )
                uncertain_count += 1
            else:
                usage_state = "used"
                usage_confidence = used_confidences[idx]
                usage_used_count += 1
        else:
            is_unused = (
                coverage_score <= unused_coverage_max
                and matched_response_tokens == 0
                and attribution_score <= 1e-6
                and support_token_ratio <= 1e-6
                and not nli_positive
                and redundancy_similarity <= 1e-6
            )
            if is_unused:
                usage_state = "unused"
                usage_confidence = unused_confidence
                unused_count += 1
            else:
                usage_state = "uncertain"
                usage_confidence = 0.5 * max(
                    used_confidences[idx],
                    unused_confidence,
                    _clamp01(redundancy_similarity),
                )
                uncertain_count += 1

        payload["usage_state"] = usage_state
        payload["usage_confidence"] = float(_clamp01(usage_confidence))
        payload["unused_confidence"] = float(_clamp01(unused_confidence))

    total = len(support_units_payload)
    return {
        "support_units_usage_used": int(usage_used_count),
        "support_units_unused": int(unused_count),
        "support_units_uncertain": int(uncertain_count),
        "context_usage_ratio": (
            float(usage_used_count) / float(total) if total > 0 else 0.0
        ),
        "context_unused_ratio": (
            float(unused_count) / float(total) if total > 0 else 0.0
        ),
        "context_uncertain_ratio": (
            float(uncertain_count) / float(total) if total > 0 else 0.0
        ),
    }


def _build_rag_file_attribution(
    *,
    support_units_payload: List[Dict[str, Any]],
    support_inputs: Sequence["SupportUnitInput"],
    n_response_tokens: int,
    n_query_tokens: int,
    coverage_threshold: float,
) -> FileAttributionResult:
    """Project the RAG per-unit stats into a lane-neutral attribution bundle.

    The shared :func:`attribute_files` kernel expects four parallel
    sequences in scorer order:

    - ``per_unit_max``           -> ``payload["coverage_score"]``
      (max cosine of each support unit against any response token).
    - ``per_unit_owner_count``   -> ``payload["matched_response_tokens"]``
      (times the unit won the per-response-token argmax).
    - ``per_unit_query_owner_count`` -> zeros for the RAG lane today.
      Query-side argmax isn't tracked per-unit on the RAG path; we
      still emit the column so the reason-code logic runs uniformly
      (files will land in ``not_query_relevant`` instead of
      ``query_relevant_but_ignored``, which is the correct default).
    - ``usage_states``           -> ``payload["usage_state"]``
      (tri-state from :func:`apply_support_unit_usage_classification`).

    ``resolve_attribution_key`` already knows how to prefer
    ``metadata.path`` / ``metadata.source`` / ``source_id`` over a
    bare ``support_id`` so enterprise stacks that label chunks by
    source document get clean file rollups; plain chunk callers get a
    per-chunk breakdown instead.
    """
    n = len(support_units_payload)
    per_unit_max: List[float] = [
        float(payload.get("coverage_score") or 0.0) for payload in support_units_payload
    ]
    per_unit_owner_count: List[int] = [
        int(payload.get("matched_response_tokens") or 0)
        for payload in support_units_payload
    ]
    per_unit_query_owner_count: List[int] = [0] * n
    usage_states: List[str] = [
        str(payload.get("usage_state") or "uncertain")
        for payload in support_units_payload
    ]
    # ``support_inputs`` may have more entries than ``support_units_payload``
    # in the chunked path (flat_support_units flattens per-chunk lists).
    # We only feed the subset that maps 1:1 with the payloads.
    units_for_attribution = list(support_inputs)[:n]
    return attribute_files(
        units=units_for_attribution,
        per_unit_max=per_unit_max,
        per_unit_owner_count=per_unit_owner_count,
        per_unit_query_owner_count=per_unit_query_owner_count,
        usage_states=usage_states,
        n_response_tokens=int(n_response_tokens),
        n_query_tokens=int(n_query_tokens),
        coverage_threshold=float(coverage_threshold),
    )


def _dedup_unit_maxima(
    unit_maxima: torch.Tensor,
    signatures: Optional[Sequence[str]],
) -> Tuple[torch.Tensor, int]:
    """Cluster columns of a ``(R, U)`` unit-maxima matrix by signature.

    Returns the deduped matrix (one column per unique signature, taking the
    column-wise max within each cluster) and the count of duplicates removed.
    Falls back to the original matrix when signatures are missing or already
    unique, so the path stays a no-op for the common single-source case.
    """

    if unit_maxima.numel() == 0:
        return unit_maxima, 0
    if not signatures or len(signatures) != int(unit_maxima.shape[1]):
        return unit_maxima, 0

    seen: Dict[str, int] = {}
    cluster_columns: List[List[int]] = []
    for column_idx, signature in enumerate(signatures):
        bucket = seen.get(signature)
        if bucket is None:
            seen[signature] = len(cluster_columns)
            cluster_columns.append([column_idx])
        else:
            cluster_columns[bucket].append(column_idx)

    duplicates_removed = int(unit_maxima.shape[1]) - len(cluster_columns)
    if duplicates_removed <= 0:
        return unit_maxima, 0

    deduped = torch.stack(
        [unit_maxima[:, indices].max(dim=1).values for indices in cluster_columns],
        dim=1,
    )
    return deduped, duplicates_removed


@dataclass
class _NullBankPack:
    """Pre-normalized, pre-concatenated null bank built once at cache time.

    Holds:
    - ``concat`` (N, H) - all bank entries' L2-normalized token vectors
      stacked in a single tensor.
    - ``segment_ids`` (N,) - bank-entry index for each row in ``concat``.
    - ``segment_count`` - number of non-empty bank entries.

    Storing the bank in this shape lets ``compute_null_distribution`` run a
    single ``(U, H) @ (N, H).T`` matmul instead of one matmul per bank
    entry, then segment-max over ``segment_ids`` to recover the per-bank
    maxima. This collapses the dominant hot path measured by
    ``scripts/profile_hot_paths.py`` (compute_null_distribution + many
    ``.max()`` calls accounted for ~62% of CPU on the balanced profile)
    into a single batched matmul plus one ``scatter_reduce``.
    """

    concat: torch.Tensor
    segment_ids: torch.Tensor
    segment_count: int


def _build_null_bank_pack(
    null_bank_embeddings: Sequence[torch.Tensor],
) -> Optional[_NullBankPack]:
    """Pre-normalize and concatenate a list of bank embeddings.

    Returns ``None`` for an empty/all-empty bank so the caller can keep
    its existing fallback logic. The result is safe to cache - the
    tensors are detached and never mutated by ``compute_null_distribution``.
    """

    parts: List[torch.Tensor] = []
    segment_ids: List[torch.Tensor] = []
    seg = 0
    for entry in null_bank_embeddings:
        if entry is None or entry.numel() == 0:
            continue
        norm = _normalize(entry.float()).detach()
        parts.append(norm)
        segment_ids.append(
            torch.full((norm.shape[0],), seg, dtype=torch.long)
        )
        seg += 1
    if not parts:
        return None
    return _NullBankPack(
        concat=torch.cat(parts, dim=0).contiguous(),
        segment_ids=torch.cat(segment_ids, dim=0).contiguous(),
        segment_count=seg,
    )


def compute_null_distribution(
    response_embeddings: torch.Tensor,
    null_bank_embeddings: Sequence[torch.Tensor] | _NullBankPack,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Per-response-token null mean and std from a bank of unrelated support units.

    For each null bank entry ``b``, computes ``g_t^(b) = max_j sim(r_t, b_j)``
    over the bank's tokens. Aggregates ``{g_t^(b)}`` across the bank into a per
    response token mean and standard deviation. The third return value is the
    number of bank entries that actually contributed (non-empty embeddings).

    Accepts either a sequence of per-bank-entry tensors (legacy callers) or
    a pre-stacked :class:`_NullBankPack` (cache-friendly fast path used by
    the API service layer). The two paths produce numerically identical
    results modulo floating-point summation order.
    """

    if response_embeddings.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32)
        return empty, empty, 0

    response_token_count = int(response_embeddings.shape[0])
    R_norm = _normalize(response_embeddings.float())

    pack: Optional[_NullBankPack]
    if isinstance(null_bank_embeddings, _NullBankPack):
        pack = null_bank_embeddings
    else:
        pack = _build_null_bank_pack(null_bank_embeddings)

    if pack is None or pack.concat.numel() == 0:
        zeros = torch.zeros((response_token_count,), dtype=torch.float32)
        ones = torch.ones((response_token_count,), dtype=torch.float32)
        return zeros, ones, 0

    # Single batched matmul: (U, H) @ (N, H).T -> (U, N). Fastest path on
    # both CPU and CUDA because the kernel coalesces the per-bank work
    # the original loop dispatched piecemeal.
    bank_concat = pack.concat.to(device=R_norm.device, dtype=R_norm.dtype)
    sim = R_norm @ bank_concat.T  # (U, N)

    segment_ids = pack.segment_ids.to(device=R_norm.device)
    segment_count = pack.segment_count

    # scatter_reduce is the modern (PyTorch >= 1.12) vectorized
    # segment-max. We initialise to a large negative so positions that
    # never get a token write keep -inf and are masked out below.
    per_unit_max = torch.full(
        (response_token_count, segment_count),
        -1.0e30,
        dtype=R_norm.dtype,
        device=R_norm.device,
    )
    seg_index = segment_ids.unsqueeze(0).expand(response_token_count, -1)
    per_unit_max = per_unit_max.scatter_reduce(
        dim=1,
        index=seg_index,
        src=sim,
        reduce="amax",
        include_self=True,
    )

    mean = per_unit_max.mean(dim=1)
    if segment_count > 1:
        std = per_unit_max.std(dim=1, unbiased=False)
    else:
        std = torch.full_like(mean, _CALIBRATION_MIN_STD)
    std = std.clamp(min=_CALIBRATION_MIN_STD)
    return mean.to(torch.float32), std.to(torch.float32), int(segment_count)


def calibrate_per_token_scores(
    rc_values: torch.Tensor,
    null_mean: torch.Tensor,
    null_std: torch.Tensor,
    *,
    temperature: float = _CALIBRATION_TEMPERATURE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standardize raw per-token reverse-context scores using a null distribution.

    Returns ``(z, p_grounded)`` where ``z = (g_t - mu_null_t) / sigma_null_t``
    is the per-token z-score against the null bank and
    ``p_grounded = sigmoid(z / temperature)`` is a probability-style
    aggregation-friendly score in ``(0, 1)``.
    """

    if rc_values.numel() == 0:
        empty = torch.zeros_like(rc_values)
        return empty, empty
    safe_std = null_std.clamp(min=_CALIBRATION_MIN_STD)
    z = (rc_values - null_mean) / safe_std
    prob = torch.sigmoid(z / max(float(temperature), _CALIBRATION_MIN_STD))
    return z, prob


# ----------------------------------------------------------------------
# Phase C: narrow-scope literal extraction and guardrails
# ----------------------------------------------------------------------

# Per-mismatch penalty applied to the literal-guarded secondary score.
# Each unmatched response literal multiplies the score by ``(1 - rate)``
# down to a configurable floor.
_LITERAL_PENALTY_RATE = 0.15
_LITERAL_PENALTY_FLOOR = 0.0

# Identity literals like "GTE-3.5-Turbo" or product codes mix letters and
# digits; URLs must end on a non-punctuation character to avoid trailing
# commas/periods leaking into the literal.
# English month name alternation, kept as a small constant so we can reuse
# it in two patterns without copy/paste drift.
_EN_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)"
)

# German month name alternation. ``M\u00e4rz`` and its abbreviation ``M\u00e4r``
# carry the umlaut explicitly; the regex runs case-insensitively so capitalized
# and lowercased forms both match. Each token appears only once - the
# alternation engine evaluates left-to-right anyway, so duplicates were
# pure dead bytes.
_DE_MONTHS = (
    r"(?:Januar|Februar|M\u00e4rz|April|Mai|Juni|Juli|August|"
    r"September|Oktober|November|Dezember|"
    r"Jan|Feb|M\u00e4r|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Okt|Nov|Dez)"
)

_LITERAL_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    # ISO date YYYY-MM-DD
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    # German numeric date DD.MM.YYYY (e.g. "20.07.1981"). Registered before
    # the English number pattern so the longer date span wins the
    # length-based conflict resolution.
    ("date", re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")),
    # Numeric date DD/MM/YYYY or MM/DD/YYYY
    ("date", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    # English long-form date "20 July 1981" / "20 Jul 1981"
    (
        "date",
        re.compile(
            r"\b\d{1,2}\s+" + _EN_MONTHS + r"\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # English month-leading date "July 20, 1981" / "Jul 20 1981"
    (
        "date",
        re.compile(
            r"\b" + _EN_MONTHS + r"\s+\d{1,2}(?:,)?\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # German long-form date with ordinal dot: "20. Juli 1981", "3. Oktober 1990".
    (
        "date",
        re.compile(
            r"\b\d{1,2}\.\s*" + _DE_MONTHS + r"\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # German month-leading date "Juli 20, 1981" - rare but accepted for symmetry.
    (
        "date",
        re.compile(
            r"\b" + _DE_MONTHS + r"\s+\d{1,2}(?:,)?\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # Bare 4-digit year (filtered later if also captured by another pattern).
    ("year", re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")),
    # Currency amounts with $/\u20ac/\u00a3 prefix and optional decimals/commas.
    ("currency", re.compile(r"(?:\$|\u20ac|\u00a3)\s?\d{1,3}(?:[,\s\.]\d{3})*(?:[\.,]\d+)?")),
    # German-style trailing currency: "5,99 \u20ac" / "1.234,50 EUR".
    ("currency", re.compile(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\s?(?:\u20ac|EUR|CHF)\b")),
    # English percent ("20%", "5.5%") and German percent ("20,5%").
    ("percent", re.compile(r"\b\d{1,3}(?:[\.,]\d+)?\s?%")),
    # Numeric measurements with common English units.
    (
        "measurement",
        re.compile(
            r"\b\d{1,4}(?:\.\d+)?\s?(?:kg|g|mg|km|m|cm|mm|mph|kph|kmh|hours?|"
            r"minutes?|seconds?|days?|years?|months?|weeks?|MB|GB|TB|KB|"
            r"liters?|litres?|ml|gallons?|miles?|feet|inches?|in|ft|lb|lbs|"
            r"oz|\u00b0C|\u00b0F)\b",
            re.IGNORECASE,
        ),
    ),
    # Numeric measurements with common German unit words.
    (
        "measurement",
        re.compile(
            r"\b\d{1,4}(?:[\.,]\d+)?\s?(?:Stunden?|Minuten?|Sekunden?|Tagen?|"
            r"Wochen?|Monaten?|Jahren?|Kilometern?|Metern?|Zentimetern?|"
            r"Millimetern?|Litern?|Milliliter|Pfund|Kilogramm|Gramm|"
            r"Tonnen?|Liter|kg|g|mg|km|m|cm|mm|t)\b",
            re.IGNORECASE,
        ),
    ),
    # German number with thousands separator and decimal comma:
    # "1.234,56", "1.234.567,89", "1.234.567". Registered before the
    # English bare-number pattern so the longer DE-formatted span wins
    # length-based conflict resolution.
    (
        "number",
        re.compile(
            r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b|\b\d{1,3}(?:\.\d{3})+\b"
        ),
    ),
    # Standalone English numbers (integers / decimals / thousands separators).
    ("number", re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")),
    # URLs (HTTP(S)).
    ("url", re.compile(r"https?://[^\s<>\"']+[^\s<>\"'.,;:!?]")),
    # Email addresses.
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # Mixed-case identifiers ("ABC123", "GTE-3.5", "section-7b") - require at
    # least one letter and one digit so it doesn't fire on prose words or
    # bare numbers (already covered above).
    (
        "identifier",
        re.compile(r"\b(?=[A-Za-z0-9._/-]{3,40}\b)[A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*\b"),
    ),
)


def _canonicalize_decimal(text: str) -> str:
    """Normalize a numeric literal (English or German) to a canonical decimal form.

    Heuristics:

    - Mixed ``.`` and ``,``: the rightmost punctuation is the decimal mark
      (``1.234,56`` -> ``1234.56`` German; ``1,234.56`` -> ``1234.56``
      English). Thousands separators on the other side are stripped.
    - Only ``,``: treated as a decimal mark when there's a single comma
      followed by a non-3-digit run (``42,5`` -> ``42.5``); otherwise as
      thousands separators (``1,234`` -> ``1234``).
    - Only ``.``: a single dot stays as the decimal mark (``42.5``); two
      or more dots are German thousands separators and are stripped
      (``1.234.567`` -> ``1234567``).

    Inputs that don't contain ``.`` or ``,`` are returned unchanged.
    """

    if not text:
        return text
    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        if text.rfind(",") > text.rfind("."):
            # German: dot = thousands, comma = decimal.
            return text.replace(".", "").replace(",", ".")
        # English: comma = thousands, dot = decimal.
        return text.replace(",", "")
    if has_comma:
        if text.count(",") == 1:
            tail = text.split(",", 1)[1]
            # A 3-digit tail with no further punctuation could be either a
            # German decimal ("1,234") or an English thousands separator
            # ("1,234"). When ambiguous, treat as English thousands - the
            # German formats this codebase has to handle in practice ship
            # with ``.`` thousands grouping, so the safer default is to
            # strip the comma rather than promote it.
            if len(tail) == 3 and tail.isdigit():
                return text.replace(",", "")
            return text.replace(",", ".")
        # Multiple commas with no dot - English thousands.
        return text.replace(",", "")
    if has_dot and text.count(".") >= 2:
        return text.replace(".", "")
    return text


# Back-compat alias - older internal callers used the German-specific name.
_normalize_german_number = _canonicalize_decimal


def _normalize_literal_value(kind: str, value: str) -> str:
    """Canonicalize a literal so equivalent surface forms collide on a hash."""

    text = value.strip().lower()
    if kind == "currency":
        text = text.replace(" ", "")
        # Map trailing-EUR spellings to the leading-symbol form so
        # "5,99\u20ac" collides with "\u20ac5.99".
        for tail in ("eur", "chf", "\u20ac"):
            if text.endswith(tail):
                text = "\u20ac" + text[: -len(tail)]
                break
        # Apply German -> canonical decimal normalization to any remaining
        # numeric body so 5,99 and 5.99 collide.
        prefix = ""
        body = text
        for sym in ("\u20ac", "$", "\u00a3"):
            if body.startswith(sym):
                prefix = sym
                body = body[len(sym):]
                break
        body = _canonicalize_decimal(body)
        text = prefix + body
    if kind == "percent":
        text = text.replace(" ", "")
        if text.endswith("%"):
            text = _canonicalize_decimal(text[:-1]) + "%"
    if kind == "measurement":
        text = re.sub(r"\s+", " ", text)
        # Split numeric prefix from unit so 1.234,5 km and 1234.5 km collide.
        match = re.match(r"^(\d[\d\.,]*)\s?(.+)$", text)
        if match:
            head, tail = match.group(1), match.group(2)
            text = _canonicalize_decimal(head) + " " + tail.strip()
    if kind == "date":
        text = re.sub(r"[,]", "", text)
        text = re.sub(r"\s+", " ", text)
    if kind == "number":
        text = _canonicalize_decimal(text)
    return text


def _extract_literals_uncached(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    raw_hits: List[Dict[str, Any]] = []
    for kind, pattern in _LITERAL_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            raw_hits.append(
                {
                    "kind": kind,
                    "value": value,
                    "normalized": _normalize_literal_value(kind, value),
                    "start": int(match.start()),
                    "end": int(match.end()),
                }
            )

    if not raw_hits:
        return []

    raw_hits.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"]))
    accepted: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    for hit in raw_hits:
        span = (hit["start"], hit["end"])
        if any(span[0] < end and start < span[1] for start, end in occupied):
            continue
        accepted.append(hit)
        occupied.append(span)
    accepted.sort(key=lambda item: item["start"])
    return accepted


@functools.lru_cache(maxsize=8192)
def _extract_literals_cached(text: str) -> Tuple[Dict[str, Any], ...]:
    return tuple(_extract_literals_uncached(text))


def extract_literals(text: str) -> List[Dict[str, Any]]:
    """Extract narrow-scope literals (dates, numbers, units, identifiers).

    Each literal is represented as ``{"kind": str, "value": str,
    "normalized": str, "start": int, "end": int}``. Overlapping spans are
    resolved greedily by length so the most informative literal wins (e.g. a
    full date beats an embedded year).

    Cached: the same support-unit text is repeatedly re-scanned across
    response chunks, support batches, and scorer configs. Callers iterate
    the returned list; do not mutate.
    """

    if not text:
        return []
    # ``lru_cache`` keys on the string identity / hash; Python interns short
    # strings and hashes longer ones in O(n) once. The hot path (support unit
    # text repeated across batches & encoder configs) hits the cache.
    return list(_extract_literals_cached(text))


def collect_support_literal_set(support_units: Sequence[Any]) -> Dict[str, set]:
    """Build a per-kind set of normalized literals across all support units."""

    bucket: Dict[str, set] = {}
    for unit in support_units:
        text = getattr(unit, "text", "") or ""
        for literal in extract_literals(text):
            bucket.setdefault(literal["kind"], set()).add(literal["normalized"])
    return bucket


def diff_literals(
    response_text: str,
    support_units: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare response literals against the support union.

    Returns ``(response_literals, mismatches, matches)``. A response literal is
    counted as matched when the same kind+normalized value appears anywhere in
    the support set, OR (for ``year`` literals) when any support ``date``
    literal contains the same year. The fall-through helps avoid double-counting
    when the support carries the full long-form date but the response only
    surfaces the year.
    """

    response_literals = extract_literals(response_text or "")
    support_buckets = collect_support_literal_set(support_units)
    support_dates = support_buckets.get("date", set())

    matches: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    for literal in response_literals:
        kind = literal["kind"]
        norm = literal["normalized"]
        bucket = support_buckets.get(kind, set())
        is_match = norm in bucket
        if not is_match and kind == "year":
            is_match = any(norm in support_date for support_date in support_dates)
        if not is_match and kind == "number":
            for measurement in support_buckets.get("measurement", set()):
                if measurement.startswith(norm + " ") or measurement == norm:
                    is_match = True
                    break
        if is_match:
            matches.append(literal)
        else:
            mismatches.append(literal)
    return response_literals, mismatches, matches


def literal_guarded_score(
    base_score: float,
    mismatches: Sequence[Dict[str, Any]],
    *,
    rate: float = _LITERAL_PENALTY_RATE,
    floor: float = _LITERAL_PENALTY_FLOOR,
) -> float:
    """Apply a multiplicative penalty per literal mismatch with a hard floor."""

    if base_score <= 0 or not mismatches:
        return float(base_score)
    penalty = (1.0 - max(0.0, min(1.0, rate))) ** len(mismatches)
    guarded = float(base_score) * float(penalty)
    if floor > 0:
        guarded = max(guarded, float(floor))
    return guarded


def _consensus_statistics(
    unit_maxima: torch.Tensor,
    reverse_context_values: torch.Tensor,
    weights: torch.Tensor,
    *,
    unit_signatures: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor | float | int]:
    if unit_maxima.numel() == 0:
        zeros = torch.zeros_like(reverse_context_values)
        return {
            "hits": zeros.to(torch.long),
            "soft_breadth": zeros,
            "effective_support_units": zeros,
            "consensus_values": reverse_context_values,
            "consensus_score": float(weighted_groundedness(reverse_context_values, weights)),
            "duplicates_removed": 0,
            "effective_unit_count": 0,
        }

    deduped_maxima, duplicates_removed = _dedup_unit_maxima(unit_maxima, unit_signatures)
    positive_mass = torch.clamp(deduped_maxima - _CONSENSUS_THRESHOLD, min=0.0)
    hits = (deduped_maxima >= _CONSENSUS_THRESHOLD).sum(dim=1).to(torch.long)
    soft_breadth = torch.sigmoid((deduped_maxima - _CONSENSUS_THRESHOLD) * _CONSENSUS_ALPHA).sum(dim=1)
    mass_sum = positive_mass.sum(dim=1)
    mass_sq_sum = (positive_mass**2).sum(dim=1)
    effective_support_units = torch.where(
        mass_sq_sum > 0,
        (mass_sum**2) / torch.clamp(mass_sq_sum, min=1e-9),
        torch.zeros_like(mass_sum),
    )
    breadth_factor = 1.0 - torch.exp(
        -torch.clamp(effective_support_units - 1.0, min=0.0) / _CONSENSUS_UNIT_SCALE
    )
    consensus_values = torch.clamp(
        reverse_context_values * (1.0 - (_CONSENSUS_PENALTY_SCALE * (1.0 - breadth_factor))),
        min=0.0,
    )
    return {
        "hits": hits,
        "soft_breadth": soft_breadth,
        "effective_support_units": effective_support_units,
        "consensus_values": consensus_values,
        "consensus_score": float(weighted_groundedness(consensus_values, weights)),
        "duplicates_removed": int(duplicates_removed),
        "effective_unit_count": int(deduped_maxima.shape[1]),
    }


def score_groundedness(
    *,
    support_units: Sequence[SupportUnitInput],
    response_embeddings: torch.Tensor,
    response_tokens: Sequence[str],
    query_embeddings: Optional[torch.Tensor] = None,
    query_tokens: Optional[Sequence[str]] = None,
    evidence_limit: int = 8,
    primary_metric: str = "reverse_context",
    debug_dense_matrices: bool = False,
    null_bank_embeddings: Optional[Sequence[torch.Tensor] | _NullBankPack] = None,
    response_text: Optional[str] = None,
    nli_provider: Optional[NLIProvider] = None,
    nli_max_claims: Optional[int] = None,
    nli_top_k_premises: Optional[int] = None,
    nli_max_batch: Optional[int] = None,
    nli_max_latency_ms: Optional[float] = None,
    nli_reranker: Optional[PremiseReranker] = None,
    nli_concat_premises: Optional[bool] = None,
    nli_premise_concat_word_budget: Optional[int] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    verification_samples: Optional[Sequence[str]] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    risk_band_stratum: Optional[str] = None,
    risk_band_policy: Optional[RiskBandPolicy] = None,
    content_type: Optional[str] = None,
    structured_enabled: Optional[bool] = None,
    structured_verification: Optional[str] = None,
    structured_support_text: Optional[str] = None,
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
    _emit_dedup_warning: bool = True,
) -> Dict[str, Any]:
    """Score response groundedness against support units.

    Returns a plain python payload suitable for Pydantic response models.
    """

    if not support_units:
        raise ValueError("At least one support unit is required")

    support_tensors = [unit.embeddings.float() for unit in support_units]
    support_tokens_nested = [align_tokens(unit.tokens, int(unit.embeddings.shape[0])) for unit in support_units]
    response_tokens_aligned = align_tokens(response_tokens, int(response_embeddings.shape[0]))
    support_signatures = [support_unit_signature(unit) for unit in support_units]

    C = torch.cat(support_tensors, dim=0)
    R = response_embeddings.float()
    C_norm = _normalize(C)
    R_norm = _normalize(R)
    sim_rc_raw = R_norm @ C_norm.T

    flat_support_tokens = [token for tokens in support_tokens_nested for token in tokens]
    support_content_mask_flat = support_content_mask(flat_support_tokens).to(sim_rc_raw.device)
    if support_content_mask_flat.numel() > 0 and bool(support_content_mask_flat.any()):
        masked_sim_rc = sim_rc_raw.masked_fill(~support_content_mask_flat.unsqueeze(0), -1e4)
    else:
        masked_sim_rc = sim_rc_raw
    sim_rc = masked_sim_rc
    rc_values, rc_indices = sim_rc.max(dim=1)

    weights = token_weights(response_tokens_aligned).to(rc_values.device, dtype=rc_values.dtype)
    reverse_context_score = weighted_groundedness(rc_values, weights)
    reverse_context_unit_values, _reverse_context_unit_token_indices = _support_unit_maxima(
        sim_rc,
        support_tokens_nested,
    )
    consensus = _consensus_statistics(
        reverse_context_unit_values,
        rc_values,
        weights,
        unit_signatures=support_signatures,
    )
    consensus_values = consensus["consensus_values"]
    consensus_hardened_score = float(consensus["consensus_score"])
    support_unit_hits = consensus["hits"]
    support_unit_soft_breadth = consensus["soft_breadth"]
    effective_support_units = consensus["effective_support_units"]
    consensus_duplicates_removed = int(consensus.get("duplicates_removed", 0))
    consensus_effective_unit_count = int(consensus.get("effective_unit_count", reverse_context_unit_values.shape[1]))

    if isinstance(null_bank_embeddings, _NullBankPack):
        _null_bank_arg = null_bank_embeddings
    elif null_bank_embeddings:
        _null_bank_arg = list(null_bank_embeddings)
    else:
        _null_bank_arg = []
    null_mean, null_std, null_bank_size = compute_null_distribution(
        response_embeddings,
        _null_bank_arg,
    )
    null_mean = null_mean.to(rc_values.device, dtype=rc_values.dtype)
    null_std = null_std.to(rc_values.device, dtype=rc_values.dtype)
    rc_z_values, p_grounded_values = calibrate_per_token_scores(rc_values, null_mean, null_std)
    if null_bank_size > 0:
        reverse_context_calibrated_score = weighted_groundedness(p_grounded_values, weights)
    else:
        reverse_context_calibrated_score = float(reverse_context_score)

    reverse_query_context_score: Optional[float] = None
    reverse_query_context_values: Optional[torch.Tensor] = None
    triangular_score: Optional[float] = None
    triangular_values: Optional[torch.Tensor] = None
    triangular_indices: Optional[torch.Tensor] = None
    echo_values: Optional[torch.Tensor] = None
    echo_mean: Optional[float] = None
    grounded_coverage_score: Optional[float] = None
    query_tokens_aligned: Optional[List[str]] = None
    query_token_rows: Optional[List[Dict[str, Any]]] = None
    sim_rq: Optional[torch.Tensor] = None
    sim_tri: Optional[torch.Tensor] = None

    if query_embeddings is not None:
        Q = query_embeddings.float()
        Q_norm = _normalize(Q)
        sim_rq = R_norm @ Q_norm.T
        query_tokens_aligned = align_tokens(query_tokens or [], int(Q.shape[0]))
        reverse_query_context_score, reverse_query_context_values = naive_reverse_maxsim_qc(
            Q_norm,
            C_norm,
            R_norm,
            weights=weights,
            normalize=False,
        )
        tri = triangular_maxsim(Q_norm, C_norm, R_norm, normalize=False)
        triangular_score = weighted_groundedness(tri.g, weights)
        triangular_values = tri.g
        triangular_indices = tri.jstar.to(torch.long)
        echo_values = tri.e
        echo_mean = weighted_groundedness(tri.e, weights)
        grounded_coverage_score = grounded_coverage(tri.u)
        query_token_rows = [
            {
                "index": idx,
                "token": query_tokens_aligned[idx],
                "coverage": float(tri.u[idx].item()),
            }
            for idx in range(len(query_tokens_aligned))
        ]
        if debug_dense_matrices:
            sim_tri = torch.minimum(sim_rc_raw, tri.a[None, :])

    metric_name = primary_metric
    metric_values = rc_values
    metric_indices = rc_indices
    if primary_metric == "triangular":
        if triangular_values is None or triangular_indices is None:
            raise ValueError("triangular primary metric requires query embeddings")
        metric_values = triangular_values
        metric_indices = triangular_indices

    support_token_unit: List[int] = []
    support_token_local_idx: List[int] = []
    support_tokens_flat: List[str] = []
    support_units_payload: List[Dict[str, Any]] = []
    support_token_scores: List[List[float]] = []
    for unit_idx, unit in enumerate(support_units):
        tokens = support_tokens_nested[unit_idx]
        support_units_payload.append(
            {
                "index": unit_idx,
                "support_id": unit.support_id,
                "chunk_id": unit.chunk_id,
                "source_mode": unit.source_mode,
                "text": unit.text,
                "offset_start": unit.offset_start,
                "offset_end": unit.offset_end,
                "token_count": len(tokens),
                "tokens": tokens,
                "token_scores": [0.0] * len(tokens),
                "score": 0.0,
                "matched_response_tokens": 0,
                "source_id": unit.source_id,
                "speaker": unit.speaker,
                "timestamp": unit.timestamp,
                "metadata": dict(unit.metadata) if unit.metadata else None,
            }
        )
        support_token_scores.append([0.0] * len(tokens))
        for token_idx, token in enumerate(tokens):
            support_token_unit.append(unit_idx)
            support_token_local_idx.append(token_idx)
            support_tokens_flat.append(token)

    support_score_numerators = [0.0 for _ in support_units]
    support_score_denominators = [0.0 for _ in support_units]

    response_token_rows: List[Dict[str, Any]] = []
    evidence_candidates: List[Dict[str, Any]] = []
    for idx, token in enumerate(response_tokens_aligned):
        support_flat_idx = int(metric_indices[idx].item()) if metric_indices.numel() else -1
        support_unit_idx = None
        support_token_idx = None
        support_token = None
        chunk_id = None
        if 0 <= support_flat_idx < len(support_token_unit):
            support_unit_idx = support_token_unit[support_flat_idx]
            support_token_idx = support_token_local_idx[support_flat_idx]
            support_token = support_tokens_flat[support_flat_idx]
            chunk_id = support_units_payload[support_unit_idx]["chunk_id"]
            score_value = float(metric_values[idx].item())
            support_token_scores[support_unit_idx][support_token_idx] = max(
                support_token_scores[support_unit_idx][support_token_idx],
                score_value,
            )
            support_score_numerators[support_unit_idx] += float(weights[idx].item()) * score_value
            support_score_denominators[support_unit_idx] += float(weights[idx].item())
            support_units_payload[support_unit_idx]["matched_response_tokens"] += 1
            if float(weights[idx].item()) > 0:
                evidence_candidates.append(
                    {
                        "response_token_index": idx,
                        "response_token": token,
                        "support_unit_index": support_unit_idx,
                        "support_token_index": support_token_idx,
                        "support_token": support_token,
                        "chunk_id": chunk_id,
                        "metric": metric_name,
                        "score": score_value,
                        "_rank": float(weights[idx].item()) * score_value,
                    }
                )
        response_token_rows.append(
            {
                "index": idx,
                "token": token,
                "weight": float(weights[idx].item()),
                "reverse_context": float(rc_values[idx].item()),
                "reverse_context_calibrated": float(p_grounded_values[idx].item()) if null_bank_size > 0 else None,
                "reverse_context_z": float(rc_z_values[idx].item()) if null_bank_size > 0 else None,
                "null_mean": float(null_mean[idx].item()) if null_bank_size > 0 else None,
                "null_std": float(null_std[idx].item()) if null_bank_size > 0 else None,
                "consensus_hardened": float(consensus_values[idx].item()),
                "support_unit_hits_above_threshold": int(support_unit_hits[idx].item()),
                "support_unit_soft_breadth": float(support_unit_soft_breadth[idx].item()),
                "effective_support_units": float(effective_support_units[idx].item()),
                "reverse_query_context": (
                    float(reverse_query_context_values[idx].item())
                    if reverse_query_context_values is not None
                    else None
                ),
                "triangular": float(triangular_values[idx].item()) if triangular_values is not None else None,
                "echo": float(echo_values[idx].item()) if echo_values is not None else None,
                "support_unit_index": support_unit_idx,
                "support_token_index": support_token_idx,
                "support_token": support_token,
                "chunk_id": chunk_id,
                "heatmap_score": float(metric_values[idx].item()),
            }
        )

    for unit_idx, payload in enumerate(support_units_payload):
        payload["token_scores"] = support_token_scores[unit_idx]
        denom = max(support_score_denominators[unit_idx], 1e-9)
        payload["score"] = float(support_score_numerators[unit_idx] / denom) if support_score_denominators[unit_idx] else 0.0

    # Per-unit coverage: max-over-response-tokens similarity to each unit.
    # This is the retrieval-efficiency observability signal — units with
    # coverage below threshold contributed nothing strong to any response
    # token and are dead weight from the retriever.
    coverage = compute_unit_coverage(
        reverse_context_unit_values,
        threshold=coverage_threshold,
    )
    coverage_per_unit = coverage["per_unit_max"]
    coverage_used_mask = coverage["used_mask"]
    attribution_used_count = sum(
        1 for payload in support_units_payload if int(payload["matched_response_tokens"]) > 0
    )
    for unit_idx, payload in enumerate(support_units_payload):
        if unit_idx < len(coverage_per_unit):
            payload["coverage_score"] = float(coverage_per_unit[unit_idx])
            payload["used"] = bool(coverage_used_mask[unit_idx])
        else:
            payload["coverage_score"] = 0.0
            payload["used"] = False

    evidence_candidates.sort(key=lambda item: item["_rank"], reverse=True)
    top_evidence = [
        {key: value for key, value in evidence.items() if key != "_rank"}
        for evidence in evidence_candidates[: max(1, evidence_limit)]
    ]

    debug_payload: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    if _emit_dedup_warning and consensus_duplicates_removed > 0:
        warnings.append(
            "support_unit_dedup: {removed} duplicate support unit(s) collapsed into {kept} unique source(s) for breadth statistics".format(
                removed=consensus_duplicates_removed,
                kept=consensus_effective_unit_count,
            )
        )
    if null_bank_size == 0:
        warnings.append(
            "calibration_disabled: no null bank embeddings supplied; reverse_context_calibrated falls back to raw reverse_context"
        )
    if debug_dense_matrices:
        debug_payload = {}
        rc_elements = int(sim_rc_raw.shape[0] * sim_rc_raw.shape[1])
        if rc_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
            debug_payload["response_to_support"] = sim_rc_raw.detach().cpu().tolist()
        else:
            warnings.append("response_to_support matrix omitted because it exceeds the debug size limit")
        if sim_rq is not None:
            rq_elements = int(sim_rq.shape[0] * sim_rq.shape[1])
            if rq_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["response_to_query"] = sim_rq.detach().cpu().tolist()
            else:
                warnings.append("response_to_query matrix omitted because it exceeds the debug size limit")
        if sim_tri is not None:
            tri_elements = int(sim_tri.shape[0] * sim_tri.shape[1])
            if tri_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["triangular_gated"] = sim_tri.detach().cpu().tolist()
            else:
                warnings.append("triangular_gated matrix omitted because it exceeds the debug size limit")
        if not debug_payload:
            debug_payload = None

    response_literals, literal_mismatches, literal_matches = diff_literals(
        response_text or "", support_units
    )
    base_for_guard = (
        reverse_context_calibrated_score if null_bank_size > 0 else reverse_context_score
    )
    literal_guarded_value = literal_guarded_score(float(base_for_guard), literal_mismatches)
    if literal_mismatches:
        warnings.append(
            "literal_mismatch: {n} response literal(s) not present in support: {sample}".format(
                n=len(literal_mismatches),
                sample=", ".join(
                    "{kind}={value}".format(kind=item["kind"], value=item["value"])
                    for item in literal_mismatches[:5]
                ),
            )
        )

    response_token_total = len(response_tokens_aligned)
    nli_payload = _maybe_run_nli(
        response_text=response_text,
        response_tokens=response_tokens_aligned,
        support_units=support_units,
        nli_provider=nli_provider,
        nli_max_claims=nli_max_claims,
        nli_top_k_premises=nli_top_k_premises,
        nli_max_batch=nli_max_batch,
        nli_max_latency_ms=nli_max_latency_ms,
        nli_reranker=nli_reranker,
        nli_concat_premises=nli_concat_premises,
        nli_premise_concat_word_budget=nli_premise_concat_word_budget,
        nli_use_atomic_claims=nli_use_atomic_claims,
    )
    if nli_payload is not None:
        warnings.extend(nli_payload.pop("warnings", []))
    nli_aggregate = nli_payload["aggregate_score"] if nli_payload else None
    verbatim_floor = _verbatim_support_floor(response_text, support_units_payload)
    if verbatim_floor is not None:
        nli_aggregate = max(float(nli_aggregate or 0.0), verbatim_floor)
    lexical_rescue = _lexical_rescue_floor(
        response_text,
        support_units_payload,
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else None
        ),
        literal_guarded=float(literal_guarded_value),
        nli_aggregate=nli_aggregate,
    )
    if lexical_rescue is not None:
        nli_aggregate = max(float(nli_aggregate or 0.0), lexical_rescue)
        warnings.append("lexical_rescue_applied")
    nli_per_token = nli_payload["per_token"] if nli_payload else [None] * response_token_total

    semantic_entropy_payload = _maybe_run_semantic_entropy(
        verification_samples=verification_samples,
        nli_provider=nli_provider,
        enabled=semantic_entropy_enabled,
        warnings=warnings,
    )
    semantic_entropy_aggregate = (
        semantic_entropy_payload["aggregate"] if semantic_entropy_payload else None
    )

    structured_payload = _maybe_run_structured(
        response_text=response_text,
        support_units=support_units_payload,
        structured_support_text=structured_support_text,
        content_type=content_type,
        enabled=structured_enabled,
        structured_verification=structured_verification,
        warnings=warnings,
    )
    structured_aggregate = (
        structured_payload["guarded_score"] if structured_payload else None
    )
    typed_structured_score = (
        structured_payload.get("typed_score") if structured_payload else None
    )
    structured_source_format = (
        structured_payload.get("source_format") if structured_payload else None
    )
    typed_claims_matched = (
        int(structured_payload.get("typed_claim_aligned", 0))
        if structured_payload
        else 0
    )
    structured_gate_applied = bool(
        typed_structured_score is not None
        and (
            str(structured_source_format or "").strip().lower()
            in {"json", "markdown_table"}
            or typed_claims_matched >= 2
        )
    )
    usage_aggregates = apply_support_unit_usage_classification(
        support_units_payload=support_units_payload,
        support_inputs=support_units,
        coverage_threshold=coverage_threshold,
        claim_records=(
            nli_payload.get("claim_records") if nli_payload is not None else None
        ),
    )

    groundedness_v2 = fuse_groundedness_v2(
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else float(reverse_context_score)
        ),
        literal_guarded=float(literal_guarded_value),
        nli_aggregate=nli_aggregate,
        semantic_entropy=semantic_entropy_aggregate,
        structured_source_guarded=structured_aggregate,
        typed_structured=typed_structured_score,
        source_format=structured_source_format,
        typed_claims_matched=typed_claims_matched,
        weights=fusion_weights,
    )
    for token_idx, row in enumerate(response_token_rows):
        row["nli_score"] = nli_per_token[token_idx] if token_idx < len(nli_per_token) else None

    effective_stratum = _resolve_effective_stratum(
        risk_band_stratum=risk_band_stratum,
        structured_source_format=structured_source_format,
        structured_verification=structured_verification,
    )
    epistemic_hedge = _epistemic_hedge_gate(
        groundedness_v2=groundedness_v2,
        claim_records=(
            nli_payload.get("claim_records") if nli_payload is not None else None
        ),
        effective_stratum=effective_stratum,
        risk_band_policy=risk_band_policy or get_risk_band_policy(),
    )
    if epistemic_hedge is not None and epistemic_hedge.get("applied"):
        groundedness_v2 = epistemic_hedge["adjusted_score"]
        warnings.append("epistemic_hedge_gate_applied")

    headline_for_band = _resolve_headline_for_risk_band(
        groundedness_v2=groundedness_v2,
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else None
        ),
        reverse_context=float(reverse_context_score),
    )
    risk_band = classify_risk_band(
        headline_for_band,
        stratum=effective_stratum,
        policy=risk_band_policy,
    )

    scores = {
        "primary_name": metric_name,
        "primary_score": float(triangular_score if metric_name == "triangular" else reverse_context_score),
        "reverse_context": float(reverse_context_score),
        "reverse_context_calibrated": float(reverse_context_calibrated_score) if null_bank_size > 0 else None,
        "literal_guarded": float(literal_guarded_value),
        "literal_mismatch_count": int(len(literal_mismatches)),
        "literal_match_count": int(len(literal_matches)),
        "literal_total_count": int(len(response_literals)),
        "nli_aggregate": float(nli_aggregate) if nli_aggregate is not None else None,
        "verbatim_support_floor": (
            float(verbatim_floor) if verbatim_floor is not None else None
        ),
        "lexical_rescue_floor": (
            float(lexical_rescue) if lexical_rescue is not None else None
        ),
        "epistemic_hedge_gate": epistemic_hedge,
        "nli_claim_count": (
            int(nli_payload["claim_count"]) if nli_payload is not None else 0
        ),
        "nli_skipped_count": (
            int(nli_payload["skipped_count"]) if nli_payload is not None else 0
        ),
        "groundedness_v2": float(groundedness_v2) if groundedness_v2 is not None else None,
        "consensus_hardened": consensus_hardened_score,
        "reverse_query_context": (
            float(reverse_query_context_score) if reverse_query_context_score is not None else None
        ),
        "triangular": float(triangular_score) if triangular_score is not None else None,
        "echo_mean": float(echo_mean) if echo_mean is not None else None,
        "grounded_coverage": float(grounded_coverage_score) if grounded_coverage_score is not None else None,
        "null_bank_size": null_bank_size,
        "semantic_entropy_aggregate": (
            float(semantic_entropy_aggregate)
            if semantic_entropy_aggregate is not None
            else None
        ),
        "semantic_entropy_raw": (
            float(semantic_entropy_payload["entropy_raw"])
            if semantic_entropy_payload and semantic_entropy_payload.get("entropy_raw") is not None
            else None
        ),
        "semantic_entropy_sample_count": (
            int(semantic_entropy_payload["sample_count"])
            if semantic_entropy_payload
            else 0
        ),
        "structured_source_guarded": (
            float(structured_aggregate) if structured_aggregate is not None else None
        ),
        "structured_source_detected": (
            bool(structured_payload["detected"]) if structured_payload else None
        ),
        "structured_source": (
            float(typed_structured_score) if typed_structured_score is not None else None
        ),
        "structured_source_typed_aligned": (
            int(structured_payload.get("typed_claim_aligned", 0)) if structured_payload else 0
        ),
        "structured_source_typed_count": (
            int(structured_payload.get("typed_claim_count", 0)) if structured_payload else 0
        ),
        "structured_verification": (
            str(structured_payload.get("structured_verification"))
            if structured_payload and structured_payload.get("structured_verification")
            else resolve_structured_mode(structured_verification)
        ),
        "structured_gate_applied": bool(structured_gate_applied),
        "risk_band": risk_band,
        "context_coverage_ratio": float(coverage["coverage_ratio"]),
        "context_coverage_threshold": float(coverage["threshold"]),
        "support_units_used": int(coverage["used_count"]),
        "support_units_total": int(coverage["total_count"]),
        "context_attribution_ratio": (
            float(attribution_used_count) / float(coverage["total_count"])
            if coverage["total_count"] > 0 else 0.0
        ),
        "context_attribution_used_count": int(attribution_used_count),
        "support_units_usage_used": int(usage_aggregates["support_units_usage_used"]),
        "support_units_unused": int(usage_aggregates["support_units_unused"]),
        "support_units_uncertain": int(usage_aggregates["support_units_uncertain"]),
        "context_usage_ratio": float(usage_aggregates["context_usage_ratio"]),
        "context_unused_ratio": float(usage_aggregates["context_unused_ratio"]),
        "context_uncertain_ratio": float(usage_aggregates["context_uncertain_ratio"]),
    }

    file_attribution = _build_rag_file_attribution(
        support_units_payload=support_units_payload,
        support_inputs=support_units,
        n_response_tokens=int(len(response_tokens_aligned)),
        n_query_tokens=int(len(query_tokens_aligned) if query_tokens_aligned is not None else 0),
        coverage_threshold=float(coverage_threshold),
    )
    scores["dead_weight_ratio"] = float(file_attribution.dead_weight_ratio)
    scores["dead_weight_file_count"] = int(len(file_attribution.dead_weight_files))

    return {
        "scores": scores,
        "response_tokens": response_token_rows,
        "support_units": support_units_payload,
        "top_evidence": top_evidence,
        "query_tokens": query_token_rows,
        "debug": debug_payload,
        "warnings": warnings,
        "file_attribution": file_attribution,
        "literal_diagnostics": {
            "response_literals": response_literals,
            "matches": literal_matches,
            "mismatches": literal_mismatches,
        },
        "nli_diagnostics": (
            None if nli_payload is None else {
                "claims": nli_payload["claim_records"],
                "aggregate_score": nli_payload["aggregate_score"],
            }
        ),
        "semantic_entropy_diagnostics": semantic_entropy_payload,
        "structured_diagnostics": structured_payload,
        "_internals": {
            "reverse_context_unit_values": reverse_context_unit_values,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_bank_size": null_bank_size,
        },
    }


def _resolve_effective_stratum(
    *,
    risk_band_stratum: Optional[str],
    structured_source_format: Optional[str],
    structured_verification: Optional[str],
) -> Optional[str]:
    """Pick the risk-band stratum, auto-routing prose RAG to ``rag_prose``.

    When the caller supplied an explicit ``risk_band_stratum`` we honour
    it. Otherwise, when the structured-verification lane stayed silent
    (detector returned no strong structured format and the caller did
    not force it on), we route to the ``rag_prose`` stratum. This keeps
    tabular/financial content strict (``default``) while giving
    multilingual legal/cyber prose a honest band threshold that matches
    the narrative-only fusion.
    """

    if risk_band_stratum:
        return risk_band_stratum
    mode = resolve_structured_mode(structured_verification)
    if mode == "on":
        return None
    if not structured_source_format:
        return "rag_prose"
    if str(structured_source_format).strip().lower() in {"json", "markdown_table"}:
        return None
    # ``prose_table`` / ``numeric_fact`` auto-detected: keep the default
    # stratum so the strict thresholds still apply to tabular content.
    return None


def _resolve_headline_for_risk_band(
    *,
    groundedness_v2: Optional[float],
    reverse_context_calibrated: Optional[float],
    reverse_context: Optional[float],
) -> Optional[float]:
    """Pick the most-fused available score for the risk-band classifier.

    Preference mirrors the evaluation harness (``_resolve_headline``):
    the fused ``groundedness_v2`` trumps the calibrated score, which in
    turn trumps the raw reverse-context MaxSim. We never fall back to
    non-calibrated values silently when the calibrated channel is
    available, because the thresholds are calibrated against the
    calibrated score.
    """

    if groundedness_v2 is not None:
        return float(groundedness_v2)
    if reverse_context_calibrated is not None:
        return float(reverse_context_calibrated)
    if reverse_context is not None:
        return float(reverse_context)
    return None


def _maybe_run_semantic_entropy(
    *,
    verification_samples: Optional[Sequence[str]],
    nli_provider: Optional[NLIProvider],
    enabled: Optional[bool],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """Compute semantic-entropy if enabled; return a serializable payload or None.

    Returns ``None`` when the peer is fully disabled or inapplicable. Emits a
    warning into ``warnings`` when the caller asked for semantic entropy but
    the prerequisites (NLI provider, samples) are missing so the response
    still reflects the fallback behaviour.
    """

    if verification_samples is None or not list(verification_samples):
        return None
    if enabled is False:
        return None
    resolved_enabled = enabled if enabled is not None else is_semantic_entropy_enabled()
    if not resolved_enabled:
        return None

    result = compute_semantic_entropy(verification_samples, nli_provider)
    if result.skipped_reason:
        warnings.append(
            "semantic_entropy_skipped: reason={reason}, samples={n}".format(
                reason=result.skipped_reason, n=result.sample_count
            )
        )
    return _semantic_entropy_to_dict(result)


def _maybe_run_structured(
    *,
    response_text: Optional[str],
    support_units: Sequence[Dict[str, Any]],
    structured_support_text: Optional[str],
    content_type: Optional[str],
    enabled: Optional[bool],
    structured_verification: Optional[str] = None,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """Run structured-source verification when a structured source is detected.

    Returns a serializable diagnostic payload. When ``guarded_score`` is
    ``None`` the fusion layer treats the channel as inactive.

    We try (in order): the caller-supplied ``structured_support_text``
    (typically the original ``raw_context``), then the concatenation of
    the support-unit texts we received. The caller-supplied text is
    preferred because raw_context chunk splitting breaks JSON and
    markdown tables into pieces that can no longer be parsed.

    ``structured_verification`` accepts ``"auto"`` (default detector),
    ``"on"`` (force detection regardless of prose-safety guard) or
    ``"off"`` (skip the lane entirely). ``VOYAGER_GROUNDEDNESS_STRUCTURED_MODE``
    overrides the default when no caller value is supplied.
    """

    if enabled is False:
        return None
    resolved_enabled = enabled if enabled is not None else is_structured_enabled()
    if not resolved_enabled:
        return None
    if not (response_text or "").strip():
        return None

    mode = resolve_structured_mode(structured_verification)
    if mode == "off":
        return None

    candidate_text = (structured_support_text or "").strip()
    if not candidate_text:
        combined = "\n".join(
            str(unit.get("text") or "") for unit in support_units if unit.get("text")
        )
        candidate_text = combined.strip()
    if not candidate_text:
        return None

    # When mode == "on" we pretend the caller sent a structured content
    # type so the detector bypasses the prose-safety guard. ``"auto"``
    # keeps the normal behaviour.
    effective_content_type = content_type
    if mode == "on" and not effective_content_type:
        effective_content_type = "application/json+schema"

    try:
        result = verify_structured_source(
            support_text=candidate_text,
            response_text=response_text or "",
            content_type=effective_content_type,
            penalty_per_mismatch=default_penalty_per_mismatch(),
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("structured_verification_failed", extra={"error": str(exc)})
        warnings.append("structured_verification_failed")
        return None

    payload = verification_to_dict(result)
    payload["detected"] = bool(result.detected)
    payload["structured_verification"] = mode
    payload["guarded_score"] = (
        float(result.guarded_score) if result.guarded_score is not None else None
    )
    if result.detected and result.mismatches:
        warnings.append(
            "structured_source_mismatch: {n} mismatch(es) across {fmt} source".format(
                n=len(result.mismatches), fmt=result.source_format or "structured"
            )
        )
    return payload


def _semantic_entropy_to_dict(result: SemanticEntropyResult) -> Dict[str, Any]:
    return {
        "aggregate": (
            float(result.aggregate) if result.aggregate is not None else None
        ),
        "entropy_raw": (
            float(result.entropy_raw) if result.entropy_raw is not None else None
        ),
        "cluster_count": int(result.cluster_count),
        "sample_count": int(result.sample_count),
        "latency_ms": float(result.latency_ms),
        "skipped_reason": result.skipped_reason,
        "clusters": [
            {
                "cluster_id": int(cluster.cluster_id),
                "members": list(cluster.members),
                "representative": cluster.representative,
            }
            for cluster in result.clusters
        ],
    }


def _maybe_run_nli(
    *,
    response_text: Optional[str],
    response_tokens: Sequence[str],
    support_units: Sequence[Any],
    nli_provider: Optional[NLIProvider],
    nli_max_claims: Optional[int],
    nli_top_k_premises: Optional[int],
    nli_max_batch: Optional[int],
    nli_max_latency_ms: Optional[float],
    nli_reranker: Optional[PremiseReranker] = None,
    nli_concat_premises: Optional[bool] = None,
    nli_premise_concat_word_budget: Optional[int] = None,
    nli_use_atomic_claims: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Run claim-level NLI and project to tokens; return ``None`` when disabled."""

    if nli_provider is None:
        return None
    text = response_text or ""
    if not text.strip() or not list(support_units):
        return {
            "claim_records": [],
            "claim_count": 0,
            "skipped_count": 0,
            "aggregate_score": None,
            "per_token": [None] * len(response_tokens),
            "warnings": [],
        }
    concat_premises = nli_concat_premises if nli_concat_premises is not None else is_premise_concat_enabled()
    use_atomic_claims = nli_use_atomic_claims if nli_use_atomic_claims is not None else is_atomic_enabled()
    verifications, warnings = verify_claims(
        response_text=text,
        support_units=support_units,
        nli_provider=nli_provider,
        max_claims=nli_max_claims if nli_max_claims is not None else 16,
        top_k_premises=nli_top_k_premises if nli_top_k_premises is not None else 3,
        max_batch=nli_max_batch if nli_max_batch is not None else 16,
        max_latency_ms=nli_max_latency_ms if nli_max_latency_ms is not None else 2000.0,
        reranker=nli_reranker,
        concat_premises=concat_premises,
        premise_concat_word_budget=nli_premise_concat_word_budget if nli_premise_concat_word_budget is not None else 384,
        use_atomic_claims=use_atomic_claims,
    )
    aggregate = aggregate_nli_score(verifications)
    per_token = project_claim_scores_to_tokens(response_tokens, text, verifications)
    claim_records = [_claim_to_dict(v) for v in verifications]
    skipped = sum(1 for v in verifications if v.skipped)
    return {
        "claim_records": claim_records,
        "claim_count": len(verifications),
        "skipped_count": int(skipped),
        "aggregate_score": aggregate,
        "per_token": per_token,
        "warnings": warnings,
    }


def _atom_to_dict(av: AtomicVerification) -> Dict[str, Any]:
    return {
        "atom_index": int(av.atom.atom_index),
        "text": av.atom.text,
        "char_start": int(av.atom.char_start),
        "char_end": int(av.atom.char_end),
        "entailment": float(av.entailment),
        "neutral": float(av.neutral),
        "contradiction": float(av.contradiction),
        "score": float(av.score),
        "skipped": bool(av.skipped),
        "skip_reason": av.skip_reason,
        "premise_count": int(len(av.premises)),
        "support_ids": list(av.support_ids),
        "support_unit_indices": [int(index) for index in av.support_unit_indices],
    }


def _claim_to_dict(verification: ClaimVerification) -> Dict[str, Any]:
    return {
        "index": int(verification.claim.index),
        "text": verification.claim.text,
        "char_start": int(verification.claim.char_start),
        "char_end": int(verification.claim.char_end),
        "entailment": float(verification.entailment),
        "neutral": float(verification.neutral),
        "contradiction": float(verification.contradiction),
        "score": float(verification.score),
        "skipped": bool(verification.skipped),
        "skip_reason": verification.skip_reason,
        "premise_count": int(len(verification.premises)),
        "support_ids": list(verification.support_ids),
        "support_unit_indices": [int(index) for index in verification.support_unit_indices],
        "atoms": [_atom_to_dict(a) for a in verification.atoms] if verification.atoms else [],
    }


def score_groundedness_chunked(
    *,
    support_batches: Sequence[Sequence[SupportUnitInput]],
    response_embeddings: torch.Tensor,
    response_tokens: Sequence[str],
    query_embeddings: Optional[torch.Tensor] = None,
    query_tokens: Optional[Sequence[str]] = None,
    evidence_limit: int = 8,
    primary_metric: str = "reverse_context",
    debug_dense_matrices: bool = False,
    null_bank_embeddings: Optional[Sequence[torch.Tensor] | _NullBankPack] = None,
    response_text: Optional[str] = None,
    nli_provider: Optional[NLIProvider] = None,
    nli_max_claims: Optional[int] = None,
    nli_top_k_premises: Optional[int] = None,
    nli_max_batch: Optional[int] = None,
    nli_max_latency_ms: Optional[float] = None,
    nli_reranker: Optional[PremiseReranker] = None,
    nli_concat_premises: Optional[bool] = None,
    nli_premise_concat_word_budget: Optional[int] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    verification_samples: Optional[Sequence[str]] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    risk_band_stratum: Optional[str] = None,
    risk_band_policy: Optional[RiskBandPolicy] = None,
    content_type: Optional[str] = None,
    structured_enabled: Optional[bool] = None,
    structured_verification: Optional[str] = None,
    structured_support_text: Optional[str] = None,
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> Dict[str, Any]:
    """Score chunked support windows and merge them by per-token maxima.

    This is mathematically exact for the shipped naive reverse-context score
    because the support-token partition only changes how the maxima are computed,
    not the underlying support-token union.
    """

    batches = [list(batch) for batch in support_batches if batch]
    if not batches:
        raise ValueError("At least one non-empty support batch is required")
    if len(batches) == 1:
        return score_groundedness(
            support_units=batches[0],
            response_embeddings=response_embeddings,
            response_tokens=response_tokens,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=evidence_limit,
            primary_metric=primary_metric,
            debug_dense_matrices=debug_dense_matrices,
            null_bank_embeddings=null_bank_embeddings,
            response_text=response_text,
            nli_provider=nli_provider,
            nli_max_claims=nli_max_claims,
            nli_top_k_premises=nli_top_k_premises,
            nli_max_batch=nli_max_batch,
            nli_max_latency_ms=nli_max_latency_ms,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_premise_concat_word_budget=nli_premise_concat_word_budget,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=verification_samples,
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
            risk_band_stratum=risk_band_stratum,
            risk_band_policy=risk_band_policy,
            content_type=content_type,
            structured_enabled=structured_enabled,
            structured_verification=structured_verification,
            structured_support_text=structured_support_text,
            coverage_threshold=coverage_threshold,
        )

    flat_support_units = [unit for batch in batches for unit in batch]
    response_tokens_aligned = align_tokens(response_tokens, int(response_embeddings.shape[0]))
    weights = token_weights(response_tokens_aligned).to(dtype=torch.float32)

    support_units_payload: List[Dict[str, Any]] = []
    support_token_scores: List[List[float]] = []
    for unit_idx, unit in enumerate(flat_support_units):
        tokens = align_tokens(unit.tokens, int(unit.embeddings.shape[0]))
        support_units_payload.append(
            {
                "index": unit_idx,
                "support_id": unit.support_id,
                "chunk_id": unit.chunk_id,
                "source_mode": unit.source_mode,
                "text": unit.text,
                "offset_start": unit.offset_start,
                "offset_end": unit.offset_end,
                "token_count": len(tokens),
                "tokens": tokens,
                "token_scores": [0.0] * len(tokens),
                "score": 0.0,
                "matched_response_tokens": 0,
                "source_id": unit.source_id,
                "speaker": unit.speaker,
                "timestamp": unit.timestamp,
                "metadata": dict(unit.metadata) if unit.metadata else None,
            }
        )
        support_token_scores.append([0.0] * len(tokens))

    batch_offsets: List[int] = []
    batch_results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    response_to_support_parts: List[np.ndarray] = []
    triangular_gated_parts: List[np.ndarray] = []
    response_to_query_matrix: Optional[List[List[float]]] = None
    reverse_context_unit_value_parts: List[torch.Tensor] = []

    support_offset = 0
    for batch_idx, batch in enumerate(batches):
        batch_offsets.append(support_offset)
        support_offset += len(batch)
        batch_result = score_groundedness(
            support_units=batch,
            response_embeddings=response_embeddings,
            response_tokens=response_tokens_aligned,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=evidence_limit,
            primary_metric=primary_metric,
            debug_dense_matrices=debug_dense_matrices,
            null_bank_embeddings=null_bank_embeddings if batch_idx == 0 else None,
            response_text=None,
            _emit_dedup_warning=False,
        )
        batch_results.append(batch_result)
        for warning_msg in batch_result.get("warnings", []):
            if warning_msg.startswith("calibration_disabled"):
                continue
            if warning_msg.startswith("literal_mismatch"):
                continue
            warnings.append(warning_msg)
        batch_internals = batch_result.get("_internals") or {}
        if batch_internals.get("reverse_context_unit_values") is not None:
            reverse_context_unit_value_parts.append(batch_internals["reverse_context_unit_values"])
        if debug_dense_matrices:
            debug_payload = batch_result.get("debug") or {}
            if debug_payload.get("response_to_support") is not None:
                response_to_support_parts.append(np.asarray(debug_payload["response_to_support"], dtype=np.float32))
            if debug_payload.get("triangular_gated") is not None:
                triangular_gated_parts.append(np.asarray(debug_payload["triangular_gated"], dtype=np.float32))
            if response_to_query_matrix is None and debug_payload.get("response_to_query") is not None:
                response_to_query_matrix = debug_payload["response_to_query"]

    token_count = len(response_tokens_aligned)
    merged_reverse_context = [float("-inf")] * token_count
    merged_reverse_query_context: Optional[List[float]] = None
    merged_triangular: Optional[List[float]] = None
    merged_echo: Optional[List[float]] = None
    merged_query_coverages: Optional[List[float]] = None
    merged_query_tokens: Optional[List[str]] = None

    winner_scores = [float("-inf")] * token_count
    winner_support_units: List[Optional[int]] = [None] * token_count
    winner_support_tokens: List[Optional[int]] = [None] * token_count
    winner_support_token_text: List[Optional[str]] = [None] * token_count
    winner_chunk_ids: List[Optional[Any]] = [None] * token_count

    for batch_idx, batch_result in enumerate(batch_results):
        support_unit_offset = batch_offsets[batch_idx]
        for row in batch_result["response_tokens"]:
            token_idx = int(row["index"])
            reverse_context_value = float(row["reverse_context"])
            if reverse_context_value > merged_reverse_context[token_idx]:
                merged_reverse_context[token_idx] = reverse_context_value

            reverse_query_context_value = row.get("reverse_query_context")
            if reverse_query_context_value is not None:
                if merged_reverse_query_context is None:
                    merged_reverse_query_context = [float("-inf")] * token_count
                reverse_query_context_value = float(reverse_query_context_value)
                if reverse_query_context_value > merged_reverse_query_context[token_idx]:
                    merged_reverse_query_context[token_idx] = reverse_query_context_value

            triangular_value = row.get("triangular")
            if triangular_value is not None:
                if merged_triangular is None:
                    merged_triangular = [float("-inf")] * token_count
                triangular_value = float(triangular_value)
                if triangular_value > merged_triangular[token_idx]:
                    merged_triangular[token_idx] = triangular_value

            echo_value = row.get("echo")
            if echo_value is not None:
                if merged_echo is None:
                    merged_echo = [float("-inf")] * token_count
                echo_value = float(echo_value)
                if echo_value > merged_echo[token_idx]:
                    merged_echo[token_idx] = echo_value

            metric_value = (
                float(row["triangular"])
                if primary_metric == "triangular" and row.get("triangular") is not None
                else reverse_context_value
            )
            if metric_value > winner_scores[token_idx]:
                winner_scores[token_idx] = metric_value
                local_support_unit = row.get("support_unit_index")
                winner_support_units[token_idx] = (
                    support_unit_offset + int(local_support_unit) if local_support_unit is not None else None
                )
                winner_support_tokens[token_idx] = (
                    int(row["support_token_index"]) if row.get("support_token_index") is not None else None
                )
                winner_support_token_text[token_idx] = row.get("support_token")
                winner_chunk_ids[token_idx] = row.get("chunk_id")

        query_rows = batch_result.get("query_tokens") or []
        if query_rows:
            if merged_query_coverages is None:
                merged_query_coverages = [float("-inf")] * len(query_rows)
                merged_query_tokens = [str(row["token"]) for row in query_rows]
            for row in query_rows:
                query_idx = int(row["index"])
                coverage_value = float(row["coverage"])
                if coverage_value > merged_query_coverages[query_idx]:
                    merged_query_coverages[query_idx] = coverage_value

    reverse_context_values = torch.tensor(merged_reverse_context, dtype=torch.float32)
    reverse_context_score = weighted_groundedness(reverse_context_values, weights)
    reverse_context_unit_values = (
        torch.cat(reverse_context_unit_value_parts, dim=1)
        if reverse_context_unit_value_parts
        else torch.empty((token_count, 0), dtype=torch.float32)
    )

    first_internals = batch_results[0].get("_internals") or {} if batch_results else {}
    null_mean_tensor = first_internals.get("null_mean")
    null_std_tensor = first_internals.get("null_std")
    null_bank_size = int(first_internals.get("null_bank_size", 0) or 0)
    if null_bank_size > 0 and null_mean_tensor is not None and null_std_tensor is not None:
        null_mean_tensor = null_mean_tensor.detach().to(dtype=torch.float32)
        null_std_tensor = null_std_tensor.detach().to(dtype=torch.float32)
        rc_z_values, p_grounded_values = calibrate_per_token_scores(
            reverse_context_values,
            null_mean_tensor,
            null_std_tensor,
            temperature=_CALIBRATION_TEMPERATURE,
        )
        reverse_context_calibrated_score = weighted_groundedness(p_grounded_values, weights)
    else:
        rc_z_values = torch.zeros_like(reverse_context_values)
        p_grounded_values = reverse_context_values.clone()
        reverse_context_calibrated_score = float(reverse_context_score)
        null_mean_tensor = None
        null_std_tensor = None
    if null_bank_size == 0:
        warnings.append(
            "calibration_disabled: no null bank embeddings supplied; reverse_context_calibrated falls back to raw reverse_context"
        )
    flat_unit_signatures = [support_unit_signature(unit) for unit in flat_support_units]
    consensus = _consensus_statistics(
        reverse_context_unit_values,
        reverse_context_values,
        weights,
        unit_signatures=flat_unit_signatures,
    )
    consensus_values = consensus["consensus_values"]
    consensus_hardened_score = float(consensus["consensus_score"])
    support_unit_hits = consensus["hits"]
    support_unit_soft_breadth = consensus["soft_breadth"]
    effective_support_units = consensus["effective_support_units"]
    consensus_duplicates_removed = int(consensus.get("duplicates_removed", 0))
    consensus_effective_unit_count = int(consensus.get("effective_unit_count", reverse_context_unit_values.shape[1]))
    if consensus_duplicates_removed > 0:
        warnings.append(
            "support_unit_dedup: {removed} duplicate support unit(s) collapsed into {kept} unique source(s) for breadth statistics".format(
                removed=consensus_duplicates_removed,
                kept=consensus_effective_unit_count,
            )
        )

    reverse_query_context_score: Optional[float] = None
    reverse_query_context_values: Optional[torch.Tensor] = None
    if merged_reverse_query_context is not None:
        reverse_query_context_values = torch.tensor(merged_reverse_query_context, dtype=torch.float32)
        reverse_query_context_score = weighted_groundedness(reverse_query_context_values, weights)

    triangular_score: Optional[float] = None
    triangular_values: Optional[torch.Tensor] = None
    if merged_triangular is not None:
        triangular_values = torch.tensor(merged_triangular, dtype=torch.float32)
        triangular_score = weighted_groundedness(triangular_values, weights)

    echo_mean: Optional[float] = None
    echo_values: Optional[torch.Tensor] = None
    if merged_echo is not None:
        echo_values = torch.tensor(merged_echo, dtype=torch.float32)
        echo_mean = weighted_groundedness(echo_values, weights)

    grounded_coverage_score: Optional[float] = None
    query_token_rows: Optional[List[Dict[str, Any]]] = None
    if merged_query_coverages is not None and merged_query_tokens is not None:
        query_coverage_tensor = torch.tensor(merged_query_coverages, dtype=torch.float32)
        grounded_coverage_score = grounded_coverage(query_coverage_tensor)
        query_token_rows = [
            {
                "index": idx,
                "token": merged_query_tokens[idx],
                "coverage": float(query_coverage_tensor[idx].item()),
            }
            for idx in range(len(merged_query_tokens))
        ]

    metric_name = primary_metric
    metric_values = triangular_values if metric_name == "triangular" else reverse_context_values
    if metric_name == "triangular" and metric_values is None:
        raise ValueError("triangular primary metric requires query embeddings")

    support_score_numerators = [0.0 for _ in flat_support_units]
    support_score_denominators = [0.0 for _ in flat_support_units]
    response_token_rows: List[Dict[str, Any]] = []
    evidence_candidates: List[Dict[str, Any]] = []
    for token_idx, token in enumerate(response_tokens_aligned):
        support_unit_idx = winner_support_units[token_idx]
        support_token_idx = winner_support_tokens[token_idx]
        support_token = winner_support_token_text[token_idx]
        chunk_id = winner_chunk_ids[token_idx]
        score_value = float(metric_values[token_idx].item())
        if support_unit_idx is not None and support_token_idx is not None:
            support_token_scores[support_unit_idx][support_token_idx] = max(
                support_token_scores[support_unit_idx][support_token_idx],
                score_value,
            )
            support_score_numerators[support_unit_idx] += float(weights[token_idx].item()) * score_value
            support_score_denominators[support_unit_idx] += float(weights[token_idx].item())
            support_units_payload[support_unit_idx]["matched_response_tokens"] += 1
            if float(weights[token_idx].item()) > 0:
                evidence_candidates.append(
                    {
                        "response_token_index": token_idx,
                        "response_token": token,
                        "support_unit_index": support_unit_idx,
                        "support_token_index": support_token_idx,
                        "support_token": support_token,
                        "chunk_id": chunk_id,
                        "metric": metric_name,
                        "score": score_value,
                        "_rank": float(weights[token_idx].item()) * score_value,
                    }
                )
        response_token_rows.append(
            {
                "index": token_idx,
                "token": token,
                "weight": float(weights[token_idx].item()),
                "reverse_context": float(reverse_context_values[token_idx].item()),
                "reverse_context_calibrated": (
                    float(p_grounded_values[token_idx].item()) if null_bank_size > 0 else None
                ),
                "reverse_context_z": (
                    float(rc_z_values[token_idx].item()) if null_bank_size > 0 else None
                ),
                "null_mean": (
                    float(null_mean_tensor[token_idx].item())
                    if null_mean_tensor is not None
                    else None
                ),
                "null_std": (
                    float(null_std_tensor[token_idx].item())
                    if null_std_tensor is not None
                    else None
                ),
                "consensus_hardened": float(consensus_values[token_idx].item()),
                "support_unit_hits_above_threshold": int(support_unit_hits[token_idx].item()),
                "support_unit_soft_breadth": float(support_unit_soft_breadth[token_idx].item()),
                "effective_support_units": float(effective_support_units[token_idx].item()),
                "reverse_query_context": (
                    float(reverse_query_context_values[token_idx].item())
                    if reverse_query_context_values is not None
                    else None
                ),
                "triangular": float(triangular_values[token_idx].item()) if triangular_values is not None else None,
                "echo": float(echo_values[token_idx].item()) if echo_values is not None else None,
                "support_unit_index": support_unit_idx,
                "support_token_index": support_token_idx,
                "support_token": support_token,
                "chunk_id": chunk_id,
                "heatmap_score": score_value,
            }
        )

    for unit_idx, payload in enumerate(support_units_payload):
        payload["token_scores"] = support_token_scores[unit_idx]
        denom = max(support_score_denominators[unit_idx], 1e-9)
        payload["score"] = float(support_score_numerators[unit_idx] / denom) if support_score_denominators[unit_idx] else 0.0

    coverage = compute_unit_coverage(
        reverse_context_unit_values,
        threshold=coverage_threshold,
    )
    coverage_per_unit = coverage["per_unit_max"]
    coverage_used_mask = coverage["used_mask"]
    attribution_used_count = sum(
        1 for payload in support_units_payload if int(payload["matched_response_tokens"]) > 0
    )
    for unit_idx, payload in enumerate(support_units_payload):
        if unit_idx < len(coverage_per_unit):
            payload["coverage_score"] = float(coverage_per_unit[unit_idx])
            payload["used"] = bool(coverage_used_mask[unit_idx])
        else:
            payload["coverage_score"] = 0.0
            payload["used"] = False

    evidence_candidates.sort(key=lambda item: item["_rank"], reverse=True)
    top_evidence = [
        {key: value for key, value in evidence.items() if key != "_rank"}
        for evidence in evidence_candidates[: max(1, evidence_limit)]
    ]

    debug_payload: Optional[Dict[str, Any]] = None
    if debug_dense_matrices:
        debug_payload = {}
        if len(response_to_support_parts) == len(batches) and response_to_support_parts:
            merged_response_to_support = np.concatenate(response_to_support_parts, axis=1)
            if merged_response_to_support.size <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["response_to_support"] = merged_response_to_support.tolist()
            else:
                warnings.append("response_to_support matrix omitted because the merged debug matrix exceeds the size limit")
        if response_to_query_matrix is not None:
            debug_payload["response_to_query"] = response_to_query_matrix
        if len(triangular_gated_parts) == len(batches) and triangular_gated_parts:
            merged_triangular_gated = np.concatenate(triangular_gated_parts, axis=1)
            if merged_triangular_gated.size <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["triangular_gated"] = merged_triangular_gated.tolist()
            else:
                warnings.append("triangular_gated matrix omitted because the merged debug matrix exceeds the size limit")
        if not debug_payload:
            debug_payload = None

    response_literals, literal_mismatches, literal_matches = diff_literals(
        response_text or "", flat_support_units
    )
    base_for_guard = (
        reverse_context_calibrated_score if null_bank_size > 0 else reverse_context_score
    )
    literal_guarded_value = literal_guarded_score(float(base_for_guard), literal_mismatches)
    if literal_mismatches:
        warnings.append(
            "literal_mismatch: {n} response literal(s) not present in support: {sample}".format(
                n=len(literal_mismatches),
                sample=", ".join(
                    "{kind}={value}".format(kind=item["kind"], value=item["value"])
                    for item in literal_mismatches[:5]
                ),
            )
        )

    nli_payload = _maybe_run_nli(
        response_text=response_text,
        response_tokens=response_tokens_aligned,
        support_units=flat_support_units,
        nli_provider=nli_provider,
        nli_max_claims=nli_max_claims,
        nli_top_k_premises=nli_top_k_premises,
        nli_max_batch=nli_max_batch,
        nli_max_latency_ms=nli_max_latency_ms,
        nli_reranker=nli_reranker,
        nli_concat_premises=nli_concat_premises,
        nli_premise_concat_word_budget=nli_premise_concat_word_budget,
        nli_use_atomic_claims=nli_use_atomic_claims,
    )
    if nli_payload is not None:
        warnings.extend(nli_payload.pop("warnings", []))
    nli_aggregate = nli_payload["aggregate_score"] if nli_payload else None
    verbatim_floor = _verbatim_support_floor(response_text, support_units_payload)
    if verbatim_floor is not None:
        nli_aggregate = max(float(nli_aggregate or 0.0), verbatim_floor)
    lexical_rescue = _lexical_rescue_floor(
        response_text,
        support_units_payload,
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else None
        ),
        literal_guarded=float(literal_guarded_value),
        nli_aggregate=nli_aggregate,
    )
    if lexical_rescue is not None:
        nli_aggregate = max(float(nli_aggregate or 0.0), lexical_rescue)
        warnings.append("lexical_rescue_applied")
    nli_per_token_chunked = nli_payload["per_token"] if nli_payload else [None] * token_count

    usage_aggregates = apply_support_unit_usage_classification(
        support_units_payload=support_units_payload,
        support_inputs=flat_support_units,
        coverage_threshold=coverage_threshold,
        claim_records=(
            nli_payload.get("claim_records") if nli_payload is not None else None
        ),
    )

    semantic_entropy_payload = _maybe_run_semantic_entropy(
        verification_samples=verification_samples,
        nli_provider=nli_provider,
        enabled=semantic_entropy_enabled,
        warnings=warnings,
    )
    semantic_entropy_aggregate = (
        semantic_entropy_payload["aggregate"] if semantic_entropy_payload else None
    )

    structured_payload = _maybe_run_structured(
        response_text=response_text,
        support_units=support_units_payload,
        structured_support_text=structured_support_text,
        content_type=content_type,
        enabled=structured_enabled,
        structured_verification=structured_verification,
        warnings=warnings,
    )
    structured_aggregate = (
        structured_payload["guarded_score"] if structured_payload else None
    )
    typed_structured_score = (
        structured_payload.get("typed_score") if structured_payload else None
    )
    structured_source_format = (
        structured_payload.get("source_format") if structured_payload else None
    )
    typed_claims_matched = (
        int(structured_payload.get("typed_claim_aligned", 0))
        if structured_payload
        else 0
    )
    structured_gate_applied = bool(
        typed_structured_score is not None
        and (
            str(structured_source_format or "").strip().lower()
            in {"json", "markdown_table"}
            or typed_claims_matched >= 2
        )
    )

    groundedness_v2 = fuse_groundedness_v2(
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else float(reverse_context_score)
        ),
        literal_guarded=float(literal_guarded_value),
        nli_aggregate=nli_aggregate,
        semantic_entropy=semantic_entropy_aggregate,
        structured_source_guarded=structured_aggregate,
        typed_structured=typed_structured_score,
        source_format=structured_source_format,
        typed_claims_matched=typed_claims_matched,
        weights=fusion_weights,
    )
    for token_idx, row in enumerate(response_token_rows):
        row["nli_score"] = (
            nli_per_token_chunked[token_idx]
            if token_idx < len(nli_per_token_chunked)
            else None
        )

    effective_stratum = _resolve_effective_stratum(
        risk_band_stratum=risk_band_stratum,
        structured_source_format=structured_source_format,
        structured_verification=structured_verification,
    )
    epistemic_hedge = _epistemic_hedge_gate(
        groundedness_v2=groundedness_v2,
        claim_records=(
            nli_payload.get("claim_records") if nli_payload is not None else None
        ),
        effective_stratum=effective_stratum,
        risk_band_policy=risk_band_policy or get_risk_band_policy(),
    )
    if epistemic_hedge is not None and epistemic_hedge.get("applied"):
        groundedness_v2 = epistemic_hedge["adjusted_score"]
        warnings.append("epistemic_hedge_gate_applied")

    headline_for_band = _resolve_headline_for_risk_band(
        groundedness_v2=groundedness_v2,
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else None
        ),
        reverse_context=float(reverse_context_score),
    )
    risk_band = classify_risk_band(
        headline_for_band,
        stratum=effective_stratum,
        policy=risk_band_policy,
    )

    scores = {
        "primary_name": metric_name,
        "primary_score": float(triangular_score if metric_name == "triangular" else reverse_context_score),
        "reverse_context": float(reverse_context_score),
        "reverse_context_calibrated": (
            float(reverse_context_calibrated_score) if null_bank_size > 0 else None
        ),
        "literal_guarded": float(literal_guarded_value),
        "literal_mismatch_count": int(len(literal_mismatches)),
        "literal_match_count": int(len(literal_matches)),
        "literal_total_count": int(len(response_literals)),
        "nli_aggregate": float(nli_aggregate) if nli_aggregate is not None else None,
        "verbatim_support_floor": (
            float(verbatim_floor) if verbatim_floor is not None else None
        ),
        "lexical_rescue_floor": (
            float(lexical_rescue) if lexical_rescue is not None else None
        ),
        "epistemic_hedge_gate": epistemic_hedge,
        "nli_claim_count": (
            int(nli_payload["claim_count"]) if nli_payload is not None else 0
        ),
        "nli_skipped_count": (
            int(nli_payload["skipped_count"]) if nli_payload is not None else 0
        ),
        "groundedness_v2": float(groundedness_v2) if groundedness_v2 is not None else None,
        "consensus_hardened": consensus_hardened_score,
        "reverse_query_context": (
            float(reverse_query_context_score) if reverse_query_context_score is not None else None
        ),
        "triangular": float(triangular_score) if triangular_score is not None else None,
        "echo_mean": float(echo_mean) if echo_mean is not None else None,
        "grounded_coverage": float(grounded_coverage_score) if grounded_coverage_score is not None else None,
        "null_bank_size": null_bank_size,
        "semantic_entropy_aggregate": (
            float(semantic_entropy_aggregate)
            if semantic_entropy_aggregate is not None
            else None
        ),
        "semantic_entropy_raw": (
            float(semantic_entropy_payload["entropy_raw"])
            if semantic_entropy_payload and semantic_entropy_payload.get("entropy_raw") is not None
            else None
        ),
        "semantic_entropy_sample_count": (
            int(semantic_entropy_payload["sample_count"])
            if semantic_entropy_payload
            else 0
        ),
        "structured_source_guarded": (
            float(structured_aggregate) if structured_aggregate is not None else None
        ),
        "structured_source_detected": (
            bool(structured_payload["detected"]) if structured_payload else None
        ),
        "structured_source": (
            float(typed_structured_score) if typed_structured_score is not None else None
        ),
        "structured_source_typed_aligned": (
            int(structured_payload.get("typed_claim_aligned", 0)) if structured_payload else 0
        ),
        "structured_source_typed_count": (
            int(structured_payload.get("typed_claim_count", 0)) if structured_payload else 0
        ),
        "structured_verification": (
            str(structured_payload.get("structured_verification"))
            if structured_payload and structured_payload.get("structured_verification")
            else resolve_structured_mode(structured_verification)
        ),
        "structured_gate_applied": bool(structured_gate_applied),
        "risk_band": risk_band,
        "context_coverage_ratio": float(coverage["coverage_ratio"]),
        "context_coverage_threshold": float(coverage["threshold"]),
        "support_units_used": int(coverage["used_count"]),
        "support_units_total": int(coverage["total_count"]),
        "context_attribution_ratio": (
            float(attribution_used_count) / float(coverage["total_count"])
            if coverage["total_count"] > 0 else 0.0
        ),
        "context_attribution_used_count": int(attribution_used_count),
        "support_units_usage_used": int(usage_aggregates["support_units_usage_used"]),
        "support_units_unused": int(usage_aggregates["support_units_unused"]),
        "support_units_uncertain": int(usage_aggregates["support_units_uncertain"]),
        "context_usage_ratio": float(usage_aggregates["context_usage_ratio"]),
        "context_unused_ratio": float(usage_aggregates["context_unused_ratio"]),
        "context_uncertain_ratio": float(usage_aggregates["context_uncertain_ratio"]),
    }

    file_attribution = _build_rag_file_attribution(
        support_units_payload=support_units_payload,
        support_inputs=flat_support_units,
        n_response_tokens=int(len(response_tokens_aligned)),
        n_query_tokens=int(len(query_token_rows) if query_token_rows else 0),
        coverage_threshold=float(coverage_threshold),
    )
    scores["dead_weight_ratio"] = float(file_attribution.dead_weight_ratio)
    scores["dead_weight_file_count"] = int(len(file_attribution.dead_weight_files))

    return {
        "scores": scores,
        "response_tokens": response_token_rows,
        "support_units": support_units_payload,
        "top_evidence": top_evidence,
        "query_tokens": query_token_rows,
        "debug": debug_payload,
        "warnings": list(dict.fromkeys(warnings)),
        "file_attribution": file_attribution,
        "literal_diagnostics": {
            "response_literals": response_literals,
            "matches": literal_matches,
            "mismatches": literal_mismatches,
        },
        "nli_diagnostics": (
            None if nli_payload is None else {
                "claims": nli_payload["claim_records"],
                "aggregate_score": nli_payload["aggregate_score"],
            }
        ),
        "semantic_entropy_diagnostics": semantic_entropy_payload,
        "structured_diagnostics": structured_payload,
        "_internals": {
            "reverse_context_unit_values": reverse_context_unit_values,
            "null_mean": null_mean_tensor,
            "null_std": null_std_tensor,
            "null_bank_size": null_bank_size,
        },
    }


@dataclass(frozen=True)
class ResponseChunkInput:
    """Encoded response window for the chunked-response scoring path.

    A ``ResponseChunkInput`` represents one packed window of the original
    ``response_text``: its embeddings, the aligned token strings, optional
    per-token char offsets within the original ``response_text`` (best-effort
    when the encoder's tokenizer exposes ``offset_mapping``), and the chunk's
    ``offset_start`` / ``offset_end`` within the original ``response_text``.

    The orchestrator stitches these chunks together so per-token scores can
    be re-keyed to their global positions for client-side heatmaps without
    UI-side re-tokenization.
    """

    text: str
    embeddings: torch.Tensor
    tokens: List[str]
    offset_start: int
    offset_end: int
    token_char_spans: Optional[List[Optional[Tuple[int, int]]]] = None


def _build_response_chunks(
    response_text: str,
    *,
    provider: Any,
    chunk_token_budget: int,
    encode_fn: Any,
    document_prompt_name: Optional[str] = None,
) -> List[ResponseChunkInput]:
    """Segment ``response_text`` into packed windows and encode each one.

    The chunker reuses ``segment_text(mode="sentence_packed")`` so the response
    is split on sentence boundaries with the same token-budget semantics as the
    raw_context path. All chunks are sent through a single batched
    ``encode_fn`` call so the provider can dispatch them as one GPU batch
    (PA4-aligned). Returns an empty list when ``response_text`` is empty.
    """

    if not response_text or not response_text.strip():
        return []

    if chunk_token_budget <= 0:
        raise ValueError("chunk_token_budget must be positive")

    spans = segment_text(
        response_text,
        mode="sentence_packed",
        provider=provider,
        chunk_token_budget=chunk_token_budget,
    )
    if not spans:
        return []

    chunk_texts = [str(span["text"]) for span in spans]
    embeddings_list = encode_fn(
        provider,
        chunk_texts,
        is_query=False,
        prompt_name=document_prompt_name,
    )

    chunks: List[ResponseChunkInput] = []
    for span, chunk_text, chunk_emb in zip(spans, chunk_texts, embeddings_list):
        expected_len = int(chunk_emb.shape[0])
        # Defensive: pathological inputs (e.g. all-whitespace chunks) can
        # produce an empty embedding row. Skip them rather than feed an
        # empty tensor into the scorer, which would either error out on
        # tokens-vs-embeddings alignment or silently produce 0/NaN scores.
        if expected_len <= 0:
            continue
        tokens, offsets = tokenize_with_offsets(
            provider,
            chunk_text,
            expected_len=expected_len,
            is_query=False,
        )
        chunks.append(
            ResponseChunkInput(
                text=chunk_text,
                embeddings=chunk_emb,
                tokens=tokens,
                offset_start=int(span["offset_start"]),
                offset_end=int(span["offset_end"]),
                token_char_spans=offsets,
            )
        )
    return chunks


def _shifted_token_char_spans(
    chunk: ResponseChunkInput,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Translate per-token char spans from chunk-local to ``response_text``-global."""

    if not chunk.token_char_spans:
        return [(None, None)] * len(chunk.tokens)
    base = int(chunk.offset_start)
    out: List[Tuple[Optional[int], Optional[int]]] = []
    for span in chunk.token_char_spans:
        if span is None:
            out.append((None, None))
            continue
        start, end = span
        out.append((base + int(start), base + int(end)))
    return out


def score_groundedness_response_chunked(
    *,
    response_chunks: Sequence[ResponseChunkInput],
    support_batches: Sequence[Sequence[SupportUnitInput]],
    response_text: str,
    query_embeddings: Optional[torch.Tensor] = None,
    query_tokens: Optional[Sequence[str]] = None,
    evidence_limit: int = 8,
    primary_metric: str = "reverse_context",
    debug_dense_matrices: bool = False,
    null_bank_embeddings: Optional[Sequence[torch.Tensor] | _NullBankPack] = None,
    nli_provider: Optional[NLIProvider] = None,
    nli_max_claims: Optional[int] = None,
    nli_top_k_premises: Optional[int] = None,
    nli_max_batch: Optional[int] = None,
    nli_max_latency_ms: Optional[float] = None,
    nli_reranker: Optional[PremiseReranker] = None,
    nli_concat_premises: Optional[bool] = None,
    nli_premise_concat_word_budget: Optional[int] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    verification_samples: Optional[Sequence[str]] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    risk_band_stratum: Optional[str] = None,
    risk_band_policy: Optional[RiskBandPolicy] = None,
    content_type: Optional[str] = None,
    structured_enabled: Optional[bool] = None,
    structured_verification: Optional[str] = None,
    structured_support_text: Optional[str] = None,
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> Dict[str, Any]:
    """Score response groundedness across one or more response chunks.

    Parity guarantee: when ``len(response_chunks) == 1`` this is a thin
    wrapper around :func:`score_groundedness_chunked` plus per-token char
    offset stamping. Per-token reverse-context, calibrated, triangular,
    echo, support attribution, and headline weighted aggregates are
    bitwise-identical (modulo float reduction order) to the unchunked path
    on the same input — see ``tests/test_response_chunking_parity.py``.

    For ``len(response_chunks) > 1`` the orchestrator scores each chunk
    against the full ``support_batches``, stitches per-token rows by
    re-mapping chunk-local indices to global positions, and recomputes
    weighted aggregates and risk bands on the concatenated global vectors.
    The math is exact because ``g_t = max_u m_{t,u}`` is row-independent and
    the global ``Σ w_t g_t / Σ w_t`` is computed once on the merged vectors,
    not as a per-chunk weighted average. NLI / literal / semantic-entropy /
    structured-source channels run exactly once on the full
    ``response_text`` (chunk 0 invocation) because they are text-level, not
    embedding-level. NLI's per-token projection is re-anchored on the
    *global* token list afterwards so per-token NLI heatmaps stay correct
    on every chunk, not just the first window.
    """

    if not response_chunks:
        raise ValueError("At least one response chunk is required")

    flat_response_tokens: List[str] = []
    chunk_token_offsets: List[int] = []
    chunk_token_spans: List[List[Tuple[Optional[int], Optional[int]]]] = []
    for chunk in response_chunks:
        chunk_token_offsets.append(len(flat_response_tokens))
        aligned = align_tokens(chunk.tokens, int(chunk.embeddings.shape[0]))
        flat_response_tokens.extend(aligned)
        chunk_token_spans.append(_shifted_token_char_spans(chunk))

    if not flat_response_tokens:
        raise ValueError("Response chunks produced an empty token sequence")

    # Single-chunk fast path — by construction parity with the unchunked
    # path. We still stamp char offsets on the returned tokens.
    if len(response_chunks) == 1:
        chunk = response_chunks[0]
        result = score_groundedness_chunked(
            support_batches=support_batches,
            response_embeddings=chunk.embeddings,
            response_tokens=chunk.tokens,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=evidence_limit,
            primary_metric=primary_metric,
            debug_dense_matrices=debug_dense_matrices,
            null_bank_embeddings=null_bank_embeddings,
            response_text=response_text,
            nli_provider=nli_provider,
            nli_max_claims=nli_max_claims,
            nli_top_k_premises=nli_top_k_premises,
            nli_max_batch=nli_max_batch,
            nli_max_latency_ms=nli_max_latency_ms,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_premise_concat_word_budget=nli_premise_concat_word_budget,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=verification_samples,
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
            risk_band_stratum=risk_band_stratum,
            risk_band_policy=risk_band_policy,
            content_type=content_type,
            structured_enabled=structured_enabled,
            structured_verification=structured_verification,
            structured_support_text=structured_support_text,
            coverage_threshold=coverage_threshold,
        )
        spans = chunk_token_spans[0]
        for row in result["response_tokens"]:
            idx = int(row["index"])
            if 0 <= idx < len(spans):
                start, end = spans[idx]
                row["char_start"] = start
                row["char_end"] = end
                row["response_chunk_index"] = None
        return result

    # Multi-chunk path. Score each response chunk against the full support
    # set; then stitch per-token rows and recompute global aggregates.
    chunk_results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for chunk_idx, chunk in enumerate(response_chunks):
        is_first = chunk_idx == 0
        chunk_result = score_groundedness_chunked(
            support_batches=support_batches,
            response_embeddings=chunk.embeddings,
            response_tokens=chunk.tokens,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=evidence_limit,
            primary_metric=primary_metric,
            debug_dense_matrices=debug_dense_matrices,
            null_bank_embeddings=null_bank_embeddings,
            # Text-level channels (NLI / literal / SE / structured) only on
            # the first chunk so the global response_text is scored exactly
            # once and not re-scored per chunk.
            response_text=response_text if is_first else None,
            nli_provider=nli_provider if is_first else None,
            nli_max_claims=nli_max_claims if is_first else None,
            nli_top_k_premises=nli_top_k_premises if is_first else None,
            nli_max_batch=nli_max_batch if is_first else None,
            nli_max_latency_ms=nli_max_latency_ms if is_first else None,
            nli_reranker=nli_reranker if is_first else None,
            nli_concat_premises=nli_concat_premises if is_first else None,
            nli_premise_concat_word_budget=(
                nli_premise_concat_word_budget if is_first else None
            ),
            nli_use_atomic_claims=nli_use_atomic_claims if is_first else None,
            verification_samples=verification_samples if is_first else None,
            semantic_entropy_enabled=semantic_entropy_enabled if is_first else None,
            fusion_weights=fusion_weights,
            risk_band_stratum=risk_band_stratum,
            risk_band_policy=risk_band_policy,
            content_type=content_type,
            structured_enabled=structured_enabled if is_first else None,
            structured_verification=structured_verification if is_first else None,
            structured_support_text=structured_support_text if is_first else None,
            coverage_threshold=coverage_threshold,
        )
        chunk_results.append(chunk_result)
        for warning_msg in chunk_result.get("warnings", []):
            warnings.append(warning_msg)

    base = chunk_results[0]
    flat_support_units = [unit for batch in support_batches for unit in batch]
    weights = token_weights(flat_response_tokens).to(dtype=torch.float32)
    token_count = len(flat_response_tokens)

    # Stitch per-token rows from each chunk into a single global list,
    # re-mapping chunk-local index -> global index.
    global_response_rows: List[Optional[Dict[str, Any]]] = [None] * token_count
    for chunk_idx, chunk_result in enumerate(chunk_results):
        offset = chunk_token_offsets[chunk_idx]
        spans = chunk_token_spans[chunk_idx]
        for row in chunk_result["response_tokens"]:
            local_idx = int(row["index"])
            global_idx = offset + local_idx
            if global_idx >= token_count:
                continue
            row = dict(row)
            row["index"] = global_idx
            row["weight"] = float(weights[global_idx].item())
            row["response_chunk_index"] = chunk_idx
            if 0 <= local_idx < len(spans):
                start, end = spans[local_idx]
                row["char_start"] = start
                row["char_end"] = end
            else:
                row["char_start"] = None
                row["char_end"] = None
            global_response_rows[global_idx] = row

    # Replace any positions that never received a row (defensive — should
    # not happen because chunks partition the response by construction).
    for idx in range(token_count):
        if global_response_rows[idx] is None:
            global_response_rows[idx] = {
                "index": idx,
                "token": flat_response_tokens[idx],
                "weight": float(weights[idx].item()),
                "reverse_context": 0.0,
                "reverse_context_calibrated": None,
                "reverse_context_z": None,
                "null_mean": None,
                "null_std": None,
                "consensus_hardened": 0.0,
                "support_unit_hits_above_threshold": 0,
                "support_unit_soft_breadth": 0.0,
                "effective_support_units": 0.0,
                "reverse_query_context": None,
                "triangular": None,
                "echo": None,
                "support_unit_index": None,
                "support_token_index": None,
                "support_token": None,
                "chunk_id": None,
                "heatmap_score": 0.0,
                "char_start": None,
                "char_end": None,
                "response_chunk_index": None,
                "nli_score": None,
            }

    response_token_rows: List[Dict[str, Any]] = [row for row in global_response_rows if row is not None]

    def _gather(field_name: str) -> List[Optional[float]]:
        return [
            (row.get(field_name) if row is not None else None)
            for row in response_token_rows
        ]

    rc_global = torch.tensor(
        [float(row["reverse_context"]) for row in response_token_rows],
        dtype=torch.float32,
    )
    reverse_context_score = weighted_groundedness(rc_global, weights)

    null_mean_global_values = _gather("null_mean")
    p_grounded_values_list = _gather("reverse_context_calibrated")
    have_calibration = all(value is not None for value in null_mean_global_values) and bool(null_mean_global_values)
    if have_calibration and all(value is not None for value in p_grounded_values_list):
        p_grounded_global = torch.tensor(
            [float(value) for value in p_grounded_values_list],
            dtype=torch.float32,
        )
        reverse_context_calibrated_score: Optional[float] = float(
            weighted_groundedness(p_grounded_global, weights)
        )
        null_bank_size = int(base["scores"].get("null_bank_size", 0) or 0)
    else:
        reverse_context_calibrated_score = None
        null_bank_size = 0

    triangular_global = None
    triangular_global_score: Optional[float] = None
    triangular_present = all(row.get("triangular") is not None for row in response_token_rows)
    if triangular_present:
        triangular_global = torch.tensor(
            [float(row["triangular"]) for row in response_token_rows],
            dtype=torch.float32,
        )
        triangular_global_score = float(weighted_groundedness(triangular_global, weights))

    echo_global = None
    echo_mean_global: Optional[float] = None
    echo_present = all(row.get("echo") is not None for row in response_token_rows)
    if echo_present:
        echo_global = torch.tensor(
            [float(row["echo"]) for row in response_token_rows],
            dtype=torch.float32,
        )
        echo_mean_global = float(weighted_groundedness(echo_global, weights))

    reverse_query_context_global_score: Optional[float] = None
    rqc_present = all(row.get("reverse_query_context") is not None for row in response_token_rows)
    if rqc_present:
        rqc_global = torch.tensor(
            [float(row["reverse_query_context"]) for row in response_token_rows],
            dtype=torch.float32,
        )
        reverse_query_context_global_score = float(weighted_groundedness(rqc_global, weights))

    consensus_global = torch.tensor(
        [float(row["consensus_hardened"]) for row in response_token_rows],
        dtype=torch.float32,
    )
    consensus_hardened_score = float(weighted_groundedness(consensus_global, weights))

    # Per-token NLI re-projection (heatmap-correctness fix for multi-chunk
    # responses). NLI runs once on chunk 0 with ``response_text`` = the full
    # response, so ``claim_records`` already covers every claim across the
    # whole response. The per-token projection inside chunk 0 is anchored on
    # chunk-0's tokens though, which means tokens beyond the first window
    # would otherwise carry ``nli_score = None`` even when their claim was
    # successfully verified. Re-project the claim records onto the *global*
    # token list so the heatmap shows every supported / refuted token, not
    # just those in the first response chunk.
    nli_diag = base.get("nli_diagnostics")
    if nli_diag and nli_diag.get("claims"):
        global_token_strings = [str(row["token"]) for row in response_token_rows]
        global_nli_per_token = project_claim_records_to_tokens(
            global_token_strings,
            response_text,
            nli_diag["claims"],
        )
        for row, value in zip(response_token_rows, global_nli_per_token):
            row["nli_score"] = value

    base_scores = base["scores"]
    base_groundedness_v2 = base_scores.get("groundedness_v2")
    base_literal_guarded = base_scores.get("literal_guarded")
    nli_aggregate = base_scores.get("nli_aggregate")
    semantic_entropy_aggregate = base_scores.get("semantic_entropy_aggregate")
    structured_aggregate = base_scores.get("structured_source_guarded")
    typed_structured_score = base_scores.get("structured_source")

    # Refuse the v2 fusion on the global vectors so the headline reflects
    # the full response, not just the first chunk's support coverage.
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score)
            if reverse_context_calibrated_score is not None
            else float(reverse_context_score)
        ),
        literal_guarded=float(base_literal_guarded) if base_literal_guarded is not None else float(reverse_context_score),
        nli_aggregate=nli_aggregate,
        semantic_entropy=semantic_entropy_aggregate,
        structured_source_guarded=structured_aggregate,
        typed_structured=typed_structured_score,
        weights=fusion_weights,
    )
    groundedness_v2 = fused if fused is not None else base_groundedness_v2

    # The merger reuses the base chunk's structured detection to decide
    # whether the rag_prose stratum is the right band for the gate.
    _merger_structured_format: Optional[str] = None
    if base_scores.get("structured_source_detected"):
        _merger_structured_format = "markdown_table"
    merger_effective_stratum = _resolve_effective_stratum(
        risk_band_stratum=risk_band_stratum,
        structured_source_format=_merger_structured_format,
        structured_verification=structured_verification,
    )
    merger_claim_records = (
        nli_diag.get("claims") if isinstance(nli_diag, dict) else None
    )
    epistemic_hedge = _epistemic_hedge_gate(
        groundedness_v2=groundedness_v2,
        claim_records=merger_claim_records,
        effective_stratum=merger_effective_stratum,
        risk_band_policy=risk_band_policy or get_risk_band_policy(),
    )
    if epistemic_hedge is not None and epistemic_hedge.get("applied"):
        groundedness_v2 = epistemic_hedge["adjusted_score"]
        warnings.append("epistemic_hedge_gate_applied")

    headline_for_band = _resolve_headline_for_risk_band(
        groundedness_v2=groundedness_v2,
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score)
            if reverse_context_calibrated_score is not None
            else None
        ),
        reverse_context=float(reverse_context_score),
    )
    risk_band = classify_risk_band(
        headline_for_band,
        stratum=merger_effective_stratum or risk_band_stratum,
        policy=risk_band_policy,
    )

    # Re-derive support_units payload (token_scores, score, matched counts)
    # from the global per-token rows so support-side heatmaps reflect the
    # full response, not just chunk-0.
    support_units_payload: List[Dict[str, Any]] = []
    support_token_scores: List[List[float]] = []
    for unit_idx, unit in enumerate(flat_support_units):
        tokens = align_tokens(unit.tokens, int(unit.embeddings.shape[0]))
        support_units_payload.append(
            {
                "index": unit_idx,
                "support_id": unit.support_id,
                "chunk_id": unit.chunk_id,
                "source_mode": unit.source_mode,
                "text": unit.text,
                "offset_start": unit.offset_start,
                "offset_end": unit.offset_end,
                "token_count": len(tokens),
                "tokens": tokens,
                "token_scores": [0.0] * len(tokens),
                "score": 0.0,
                "matched_response_tokens": 0,
                "source_id": unit.source_id,
                "speaker": unit.speaker,
                "timestamp": unit.timestamp,
                "metadata": dict(unit.metadata) if unit.metadata else None,
            }
        )
        support_token_scores.append([0.0] * len(tokens))

    support_score_numerators = [0.0 for _ in flat_support_units]
    support_score_denominators = [0.0 for _ in flat_support_units]
    evidence_candidates: List[Dict[str, Any]] = []
    metric_name = primary_metric
    metric_values = (
        triangular_global if (metric_name == "triangular" and triangular_global is not None) else rc_global
    )

    for token_idx, row in enumerate(response_token_rows):
        support_unit_idx = row.get("support_unit_index")
        support_token_idx = row.get("support_token_index")
        support_token = row.get("support_token")
        chunk_id = row.get("chunk_id")
        score_value = float(metric_values[token_idx].item())
        weight_value = float(weights[token_idx].item())
        if support_unit_idx is not None and 0 <= int(support_unit_idx) < len(flat_support_units):
            unit_idx = int(support_unit_idx)
            if support_token_idx is not None and 0 <= int(support_token_idx) < len(support_token_scores[unit_idx]):
                tok_idx = int(support_token_idx)
                support_token_scores[unit_idx][tok_idx] = max(
                    support_token_scores[unit_idx][tok_idx],
                    score_value,
                )
            support_score_numerators[unit_idx] += weight_value * score_value
            support_score_denominators[unit_idx] += weight_value
            support_units_payload[unit_idx]["matched_response_tokens"] += 1
            if weight_value > 0:
                evidence_candidates.append(
                    {
                        "response_token_index": token_idx,
                        "response_token": row["token"],
                        "support_unit_index": unit_idx,
                        "support_token_index": int(support_token_idx) if support_token_idx is not None else None,
                        "support_token": support_token,
                        "chunk_id": chunk_id,
                        "metric": metric_name,
                        "score": score_value,
                        "_rank": weight_value * score_value,
                    }
                )
        # heatmap_score already set per chunk; refresh to use the (possibly
        # primary-metric overridden) score so heatmaps stay consistent.
        row["heatmap_score"] = score_value

    for unit_idx, payload in enumerate(support_units_payload):
        payload["token_scores"] = support_token_scores[unit_idx]
        denom = max(support_score_denominators[unit_idx], 1e-9)
        payload["score"] = (
            float(support_score_numerators[unit_idx] / denom)
            if support_score_denominators[unit_idx] else 0.0
        )

    # Per-unit coverage across response chunks: each chunk already computed
    # coverage_score against the FULL support set (because each chunk is
    # scored against support_batches in full). The global coverage for unit
    # u is therefore the max over chunks of chunk_k.support_units[u].
    # coverage_score. This is mathematically exact — partitioning the
    # response axis cannot change the maximum, only how it is computed.
    #
    # We use sentinel ``None`` (instead of -inf) for "no chunk reported a
    # value" so the downstream "used" derivation can short-circuit
    # degenerate units (0 tokens) to ``used=False`` consistently with the
    # ``compute_unit_coverage`` helper. This matches the semantics of the
    # single-chunk and chunked paths bit-for-bit so chunked / unchunked
    # parity holds for ``coverage_score`` and ``used`` as well as for the
    # headline scalar.
    unit_count = len(flat_support_units)
    coverage_per_unit_global: List[Optional[float]] = [None] * unit_count
    for chunk_result in chunk_results:
        for chunk_unit in chunk_result.get("support_units", []):
            unit_idx = int(chunk_unit.get("index", -1))
            if not (0 <= unit_idx < unit_count):
                continue
            chunk_cov = chunk_unit.get("coverage_score")
            if chunk_cov is None:
                continue
            chunk_cov_value = float(chunk_cov)
            current = coverage_per_unit_global[unit_idx]
            if current is None or chunk_cov_value > current:
                coverage_per_unit_global[unit_idx] = chunk_cov_value
    threshold_value = float(coverage_threshold)
    for unit_idx, payload in enumerate(support_units_payload):
        cov = coverage_per_unit_global[unit_idx]
        if cov is None:
            payload["coverage_score"] = 0.0
            payload["used"] = False
            continue
        cov_value = float(cov)
        payload["coverage_score"] = cov_value
        payload["used"] = bool(cov_value >= threshold_value)

    coverage_used_count = sum(1 for payload in support_units_payload if payload["used"])
    coverage_attribution_used_count = sum(
        1 for payload in support_units_payload if int(payload["matched_response_tokens"]) > 0
    )
    coverage_total = unit_count
    usage_aggregates = apply_support_unit_usage_classification(
        support_units_payload=support_units_payload,
        support_inputs=flat_support_units,
        coverage_threshold=coverage_threshold,
        claim_records=(
            nli_diag.get("claims")
            if isinstance(nli_diag, dict)
            else None
        ),
    )

    evidence_candidates.sort(key=lambda item: item["_rank"], reverse=True)
    top_evidence = [
        {key: value for key, value in evidence.items() if key != "_rank"}
        for evidence in evidence_candidates[: max(1, evidence_limit)]
    ]

    # Aggregate query-side coverage by per-query-token max across chunks.
    query_token_rows: Optional[List[Dict[str, Any]]] = None
    base_query_rows = base.get("query_tokens")
    if base_query_rows:
        query_count = len(base_query_rows)
        query_tokens_strs = [str(row["token"]) for row in base_query_rows]
        merged_coverage = [float("-inf")] * query_count
        for chunk_result in chunk_results:
            chunk_query_rows = chunk_result.get("query_tokens") or []
            for q_row in chunk_query_rows:
                q_idx = int(q_row["index"])
                if q_idx >= query_count:
                    continue
                value = float(q_row["coverage"])
                if value > merged_coverage[q_idx]:
                    merged_coverage[q_idx] = value
        query_token_rows = [
            {
                "index": idx,
                "token": query_tokens_strs[idx],
                "coverage": merged_coverage[idx] if merged_coverage[idx] != float("-inf") else 0.0,
            }
            for idx in range(query_count)
        ]
        coverage_tensor = torch.tensor(
            [row["coverage"] for row in query_token_rows], dtype=torch.float32
        )
        grounded_coverage_score: Optional[float] = float(grounded_coverage(coverage_tensor))
    else:
        grounded_coverage_score = None

    debug_payload: Optional[Dict[str, Any]] = None
    if debug_dense_matrices:
        # Stitch dense matrices vertically across response chunks so the
        # client gets a single (R_total × C_total) matrix matching the
        # global per-token order. Skip when any chunk is missing a debug
        # payload (defensive — happens when debug requests exceed the
        # internal size guard).
        rs_parts: List[np.ndarray] = []
        tg_parts: List[np.ndarray] = []
        rq_first: Optional[Any] = None
        ok_rs = True
        ok_tg = True
        for chunk_result in chunk_results:
            chunk_debug = chunk_result.get("debug") or {}
            chunk_rs = chunk_debug.get("response_to_support")
            if chunk_rs is None:
                ok_rs = False
            elif ok_rs:
                rs_parts.append(np.asarray(chunk_rs, dtype=np.float32))
            chunk_tg = chunk_debug.get("triangular_gated")
            if chunk_tg is None:
                ok_tg = False
            elif ok_tg:
                tg_parts.append(np.asarray(chunk_tg, dtype=np.float32))
            if rq_first is None and chunk_debug.get("response_to_query") is not None:
                rq_first = chunk_debug["response_to_query"]
        debug_payload = {}
        if ok_rs and rs_parts:
            merged_rs = np.concatenate(rs_parts, axis=0)
            if merged_rs.size <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["response_to_support"] = merged_rs.tolist()
            else:
                warnings.append(
                    "response_to_support matrix omitted because the merged debug matrix exceeds the size limit"
                )
        if ok_tg and tg_parts:
            merged_tg = np.concatenate(tg_parts, axis=0)
            if merged_tg.size <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["triangular_gated"] = merged_tg.tolist()
            else:
                warnings.append(
                    "triangular_gated matrix omitted because the merged debug matrix exceeds the size limit"
                )
        if rq_first is not None:
            debug_payload["response_to_query"] = rq_first
        if not debug_payload:
            debug_payload = None

    scores = dict(base_scores)
    scores["primary_name"] = metric_name
    scores["primary_score"] = float(
        triangular_global_score
        if (metric_name == "triangular" and triangular_global_score is not None)
        else reverse_context_score
    )
    scores["reverse_context"] = float(reverse_context_score)
    scores["reverse_context_calibrated"] = (
        float(reverse_context_calibrated_score)
        if reverse_context_calibrated_score is not None
        else None
    )
    scores["consensus_hardened"] = consensus_hardened_score
    scores["reverse_query_context"] = reverse_query_context_global_score
    scores["triangular"] = triangular_global_score
    scores["echo_mean"] = echo_mean_global
    scores["grounded_coverage"] = grounded_coverage_score
    scores["null_bank_size"] = null_bank_size
    scores["groundedness_v2"] = float(groundedness_v2) if groundedness_v2 is not None else None
    scores["epistemic_hedge_gate"] = epistemic_hedge
    scores["risk_band"] = risk_band
    scores["context_coverage_ratio"] = (
        float(coverage_used_count) / float(coverage_total) if coverage_total > 0 else 0.0
    )
    scores["context_coverage_threshold"] = float(coverage_threshold)
    scores["support_units_used"] = int(coverage_used_count)
    scores["support_units_total"] = int(coverage_total)
    scores["context_attribution_ratio"] = (
        float(coverage_attribution_used_count) / float(coverage_total)
        if coverage_total > 0 else 0.0
    )
    scores["context_attribution_used_count"] = int(coverage_attribution_used_count)
    scores["support_units_usage_used"] = int(usage_aggregates["support_units_usage_used"])
    scores["support_units_unused"] = int(usage_aggregates["support_units_unused"])
    scores["support_units_uncertain"] = int(usage_aggregates["support_units_uncertain"])
    scores["context_usage_ratio"] = float(usage_aggregates["context_usage_ratio"])
    scores["context_unused_ratio"] = float(usage_aggregates["context_unused_ratio"])
    scores["context_uncertain_ratio"] = float(usage_aggregates["context_uncertain_ratio"])

    return {
        "scores": scores,
        "response_tokens": response_token_rows,
        "support_units": support_units_payload,
        "top_evidence": top_evidence,
        "query_tokens": query_token_rows,
        "debug": debug_payload,
        "warnings": list(dict.fromkeys(warnings)),
        "literal_diagnostics": base.get("literal_diagnostics"),
        "nli_diagnostics": base.get("nli_diagnostics"),
        "semantic_entropy_diagnostics": base.get("semantic_entropy_diagnostics"),
        "structured_diagnostics": base.get("structured_diagnostics"),
        "_internals": base.get("_internals"),
        "_response_chunk_count": len(response_chunks),
    }

