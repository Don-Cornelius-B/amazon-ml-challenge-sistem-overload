"""High-Speed Text Preprocessing & Entity Normalization Module

Amazon ML Challenge 2026 — Team SISTem Overload.

Features:
- Unicode NFKD normalization (French accents é->e, ç->c, etc.)
- Single-pass compiled regex substitution for legal suffixes and address tokens
- Discrete postal/PIN and numeric token extraction for verification
"""

import re
import unicodedata
from typing import Set, Tuple, List, Dict
import polars as pl

# Basic regex cleaners
RE_AMPERSAND = re.compile(r"&", re.IGNORECASE)
RE_CONTROL_CHARS = re.compile(r"[\r\n\t\x00-\x1f\x7f-\x9f]")
RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d{1,8}\b")

# Legal Suffix Dictionary
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    # Indian / Commonwealth
    "private limited": " pvtltd ",
    "pvt ltd": " pvtltd ",
    "pvt. ltd.": " pvtltd ",
    "pvt limited": " pvtltd ",
    "pvt. limited": " pvtltd ",
    "private ltd": " pvtltd ",
    "limited liability partnership": " llp ",
    "l.l.p.": " llp ",
    "llp": " llp ",
    "limited": " ltd ",
    "ltd": " ltd ",
    "ltd.": " ltd ",
    # US / UK
    "limited liability company": " llc ",
    "l.l.c.": " llc ",
    "llc": " llc ",
    "incorporated": " inc ",
    "inc": " inc ",
    "inc.": " inc ",
    "corporation": " corp ",
    "corp": " corp ",
    "corp.": " corp ",
    "company": " company ",
    "co.": " company ",
    "co": " company ",
    "enterprises": " enterprise ",
    "enterprise": " enterprise ",
    "ent.": " enterprise ",
    # French (Critical for France test set)
    "societe a responsabilite limitee": " sarl ",
    "s.a.r.l.": " sarl ",
    "sarl": " sarl ",
    "societe par actions simplifiee": " sas ",
    "s.a.s.": " sas ",
    "sas": " sas ",
    "societe anonyme": " sa ",
    "s.a.": " sa ",
    "sa": " sa ",
    "entreprise unipersonnelle a responsabilite limitee": " eurl ",
    "e.u.r.l.": " eurl ",
    "eurl": " eurl ",
    "societe civile immobiliere": " sci ",
    "s.c.i.": " sci ",
    "sci": " sci ",
    # European
    "gesellschaft mit beschrankter haftung": " gmbh ",
    "gmbh": " gmbh ",
}

# Compile single-pass pattern sorted by length descending
SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Address Abbreviations Dictionary
ADDRESS_ABBR_MAP: Dict[str, str] = {
    "road": "road",
    "rd.": "road",
    "rd": "road",
    "street": "street",
    "st.": "street",
    "st": "street",
    "avenue": "avenue",
    "ave.": "avenue",
    "av.": "avenue",
    "ave": "avenue",
    "boulevard": "boulevard",
    "blvd.": "boulevard",
    "bd.": "boulevard",
    "bvd.": "boulevard",
    "blvd": "boulevard",
    "drive": "drive",
    "dr.": "drive",
    "dr": "drive",
    "lane": "lane",
    "ln.": "lane",
    "ln": "lane",
    "highway": "highway",
    "hwy.": "highway",
    "apartment": "apartment",
    "apt.": "apartment",
    "apt": "apartment",
    "suite": "suite",
    "ste.": "suite",
    "ste": "suite",
    "floor": "floor",
    "fl.": "floor",
    "building": "building",
    "bldg.": "building",
    "bldg": "building",
    "opposite": "opposite",
    "opp.": "opposite",
    "opp": "opposite",
    "near": "near",
    "nr.": "near",
    "nr": "near",
    "route": "route",
    "rte.": "route",
    "rte": "route",
}

ADDRESS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ADDRESS_ABBR_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def strip_accents_and_normalize(text: str) -> str:
    """Normalize Unicode characters using NFKD decomposition."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ASCII", "ignore").decode("utf-8").lower()


def clean_business_name(name: str) -> str:
    """Standardizes a business name string with single-pass regex."""
    if not name:
        return ""
    s = strip_accents_and_normalize(name)
    s = RE_CONTROL_CHARS.sub(" ", s)
    s = RE_AMPERSAND.sub(" and ", s)
    s = SUFFIX_PATTERN.sub(lambda m: LEGAL_SUFFIX_MAP.get(m.group(0).lower(), " "), s)
    s = RE_NON_ALPHANUM.sub(" ", s)
    return RE_WHITESPACE.sub(" ", s).strip()


def clean_business_address(address: str) -> str:
    """Standardizes a business address string with single-pass regex."""
    if not address:
        return ""
    s = strip_accents_and_normalize(address)
    s = RE_CONTROL_CHARS.sub(" ", s)
    s = RE_AMPERSAND.sub(" and ", s)
    s = ADDRESS_PATTERN.sub(lambda m: f" {ADDRESS_ABBR_MAP.get(m.group(0).lower(), m.group(0))} ", s)
    s = RE_NON_ALPHANUM.sub(" ", s)
    return RE_WHITESPACE.sub(" ", s).strip()


def extract_numeric_tokens(address: str) -> Set[str]:
    """Extracts numeric digit tokens from an address string."""
    if not address:
        return set()
    return set(RE_DIGITS.findall(address))


def extract_primary_token(name: str) -> str:
    """Extracts the first significant token of the business name."""
    if not name:
        return ""
    tokens = name.split()
    return tokens[0] if tokens else ""


def build_blocking_text(clean_name: str, clean_address: str) -> str:
    """Combines normalized name and address into a composite string for blocking."""
    return f"{clean_name} {clean_address}".strip()


def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Applies high-speed batch normalization across business_name and business_address."""
    names = df["business_name"].fill_null("").to_list()
    addresses = df["business_address"].fill_null("").to_list()

    clean_names = [clean_business_name(n) for n in names]
    clean_addrs = [clean_business_address(a) for a in addresses]
    blocking_texts = [
        build_blocking_text(n, a) for n, a in zip(clean_names, clean_addrs)
    ]

    return df.with_columns([
        pl.Series("clean_name", clean_names),
        pl.Series("clean_address", clean_addrs),
        pl.Series("blocking_text", blocking_texts),
    ])
