import re
import unicodedata

import polars as pl

try:
    from indic_transliteration import sanscript as _sanscript
    from indic_transliteration.sanscript import transliterate as _transliterate
    HAS_INDIC = True
except Exception:
    HAS_INDIC = False

# Unicode ranges for Latin Extended-A/B (common French/European accented chars)
_LATIN_EXTENDED_RANGES = [
    (0x00C0, 0x00FF),  # Latin-1 Supplement (À-ÿ)
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0180, 0x024F),  # Latin Extended-B
    (0x1E00, 0x1EFF),  # Latin Extended Additional
]

def _is_latin_extended(text: str) -> bool:
    """Check if text contains Latin Extended characters (French/European accents)."""
    return any(any(lo <= ord(c) <= hi for lo, hi in _LATIN_EXTENDED_RANGES) for c in text)

def _fold_latin_accents(text: str) -> str:
    """Fold Latin accented characters to base ASCII (é→e, ç→c, etc.)."""
    # NFD decomposes accented chars into base + combining mark
    # Then filter out combining marks (Mn category)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

STOPWORDS = {"the", "and", "of", "for"}

# Legal / business entity suffixes stripped (token-exact) from the END of a
# business name, so words that merely contain a suffix substring like "coffee"
# are never altered.
BUSINESS_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "group", "plc", "llp", "pvt", "private", "pty",
    "trust", "trustees", "sarl", "sas", "sa", "eurl", "sarll", "gmbh", "ag",
    "kg", "kft", "srl", "spzoo", "holding",
}

# Token-level address abbreviation expansion (US + India + basic France).
ADDRESS_EXPANSION = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "hwy": "highway", "ct": "court", "dr": "drive",
    "ln": "lane", "pkwy": "parkway", "ter": "terrace", "tpke": "turnpike",
    "pl": "place", "cir": "circle", "sq": "square",
    "n": "north", "e": "east", "s": "south", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nr": "near", "no": "number", "nos": "numbers", "po": "post office",
    "opp": "opposite", "c/o": "care",
    "r": "rue", "b": "bis",
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

_ASCII_PUNCT_RE = re.compile(r"[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+")
_SPACE_RE = re.compile(r"\s+")


def _norm_tokens(text: str) -> list:
    # ASCII punctuation only — Unicode letter/combining marks are preserved so
    # Devanagari/Telugu/etc. words are never split apart.
    text = text.replace("'", "").replace("\u2019", "").replace("\u2018", "")
    text = _ASCII_PUNCT_RE.sub(" ", text.lower())
    return _SPACE_RE.sub(" ", text).strip().split()


def _strip_suffixes(tokens: list) -> list:
    while tokens and tokens[-1] in BUSINESS_SUFFIXES:
        tokens.pop()
    return tokens


# --- Transliteration ---------------------------------------------------------
_SCRIPT_BLOCKS = {
    _sanscript.DEVANAGARI: (0x0900, 0x097F),
    _sanscript.TELUGU: (0x0C00, 0x0C7F),
    _sanscript.KANNADA: (0x0C80, 0x0CFF),
    _sanscript.TAMIL: (0x0B80, 0x0BFF),
    _sanscript.MALAYALAM: (0x0D00, 0x0D7F),
    _sanscript.GUJARATI: (0x0A80, 0x0AFF),
    _sanscript.BENGALI: (0x0980, 0x09FF),
    _sanscript.GURMUKHI: (0x0A00, 0x0A7F),
    _sanscript.ORIYA: (0x0B00, 0x0B7F),
}


def _detect_scheme(text: str):
    for scheme, (lo, hi) in _SCRIPT_BLOCKS.items():
        if any(lo <= ord(c) <= hi for c in text):
            return scheme
    return _sanscript.DEVANAGARI


_LEFT_OVER_PUNCT_RE = re.compile(r"[^\x00-\x7f]+")


def transliterate(name: str) -> str:
    """Romanise non-Latin script names to lowercased ASCII (best-effort).
    
    For Latin-script with accents (French, etc.), fold accents to base Latin.
    For Indic scripts, use indic-transliteration library.
    """
    if name is None or not name:
        return ""
    if name.isascii():
        return name.lower()
    
    # Latin Extended (French, European accents) -> fold accents properly
    if _is_latin_extended(name):
        folded = _fold_latin_accents(name)
        return _SPACE_RE.sub(" ", folded).strip().lower()
    
    # Indic scripts -> use transliteration library
    if HAS_INDIC:
        try:
            roman = _transliterate(name, _detect_scheme(name), _sanscript.ITRANS)
        except Exception:
            roman = name
    else:
        roman = name
    # Drop any non-ASCII the library left behind.
    return _SPACE_RE.sub(" ", "".join(
        c if c.isascii() else " " for c in roman
    )).strip().lower()


def normalize_name(name: str, script_fold: bool = True):
    """Return (normalized_name_string, significant_name_tokens).

    With script_fold=True the name is first romanised, so a Devanagari name and
    its Latin spelling share tokens and blocking keys.
    """
    if not name:
        return "", []
    raw = name.replace("&", " and ")
    if script_fold and not raw.isascii():
        raw = transliterate(raw)
    tokens = _norm_tokens(raw)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = _strip_suffixes(tokens)
    return " ".join(tokens), tokens


def _expand_in_slot(tokens: list, pos: int) -> None:
    if 0 <= pos < len(tokens) and len(tokens[pos]) == 2 and tokens[pos] in US_STATES:
        tokens[pos] = US_STATES[tokens[pos]]


def normalize_address(address: str) -> str:
    if not address:
        return ""
    addr = address.replace("&", " and ")
    regions = [_norm_tokens(r) for r in addr.split(",")]

    if len(regions) == 1:
        _expand_in_slot(regions[0], 0)
    else:
        if len(regions[0]) == 1:
            _expand_in_slot(regions[0], 0)
        tail = regions[-1]
        for pos in (len(tail) - 1, len(tail) - 2):
            if 0 <= pos < len(tail):
                if tail[pos].isdigit():
                    continue
                _expand_in_slot(tail, pos)

    flat = [ADDRESS_EXPANSION.get(t, t) for t in sum(regions, [])]
    return " ".join(flat)


NORM_COLS = ["entity_id", "name_norm", "name_tokens", "addr_norm", "country_norm"]


def normalize_business(df: pl.DataFrame) -> pl.DataFrame:
    """Augment a source dataframe with normalized columns:

      name_norm    : normalized name (transliterated, suffixes + stopwords rm)
      name_tokens  : list[str] of significant name tokens
      addr_norm    : normalized address ("" when the address is null)
      country_norm : lowercased country label
    """
    return df.with_columns(
        pl.col("business_name").map_elements(
            lambda s: normalize_name(s)[0], return_dtype=pl.Utf8
        ).alias("name_norm"),
        pl.col("business_name").map_elements(
            lambda s: normalize_name(s)[1], return_dtype=pl.List(pl.Utf8)
        ).alias("name_tokens"),
        pl.col("business_address").map_elements(
            normalize_address, return_dtype=pl.Utf8
        ).alias("addr_norm"),
        pl.col("country").str.strip_chars().str.to_lowercase().alias("country_norm"),
    ).select(["entity_id", "name_norm", "name_tokens", "addr_norm", "country_norm"])


if __name__ == "__main__":
    sample = pl.DataFrame({
        "entity_id": ["X1", "X2", "X3", "X4", "X5"],
        "business_name": [
            "Acme Corp Inc",
            "भारत हेरिटेज प्राइवेट लिमिटेड",
            "Smart Healthcare Private Limited",
            "McDonald's Corp.",
            "డ్రీమ్ కన్\u200cస్ట్రక్షన్ లిమిటెడ్",
        ],
        "business_address": [
            "123 Main St, Springfield, CA",
            "No. 10 MG Road, Bengaluru, Karnataka",
            "303, 3Rd Floor Sakar 5 B/H Natraj Cinema Ashram Road, Ahmedabad, GJ",
            "4 Willow Ct, Austin, TX 78701",
            None,
        ],
        "country": ["US", "India", "India", "US", "India"],
    })
    out = normalize_business(sample)
    print(out.to_dicts())