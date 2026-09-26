"""Ground Truth & Error Distribution Audit Script

Amazon ML Challenge 2026 — Team SISTem Overload.
Location: experiments/audit_ground_truth.py

Performs deep audit of the training ground truth and source datasets to quantify:
1. True singleton percentage (unmatched S1 entities).
2. Multi-match distribution (P(S1 has matches in both S2 and S3), S2 only, S3 only).
3. Exact normalized key resolution potential (Phone, Domain, Exact Name + Postal Code).
4. Precision and coverage of deterministic pre-pass linkages.
"""

import os
import sys
import gc
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import Counter
import polars as pl

# Ensure correct package import path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.config import (
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GT_PATH,
)
from src.preprocess import clean_business_name, clean_business_address, extract_numeric_tokens

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GroundTruthAudit")

DOMAIN_BLACKLIST = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "linkedin.com",
    "google.com",
    "apple.com",
    "microsoft.com",
    "amazon.com",
}

# Precompiled regexes for deterministic key extraction
RE_PHONE_CANDIDATE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,5}"
)
RE_PHONE_PREFIXED = re.compile(
    r"(?:ph|phone|tel|telephone|mobile|cell|mob|fax|contact|mo)[:.\s]*([0-9\s,\-\.\(\)\+]{7,}\d)",
    re.IGNORECASE,
)
RE_PHONE_STANDALONE = re.compile(r"\b\d{7,13}\b")
RE_DOMAIN = re.compile(
    r"(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)*\.(?:com|org|net|in|fr|co|gov|edu|biz|info|us|io|ai|ca|de|uk|eu|co\.in|co\.uk))\b",
    re.IGNORECASE,
)
RE_POSTAL_5_6 = re.compile(r"\b\d{5,6}\b")


def clean_phone_digits(text: str) -> Set[str]:
    """Extracts and normalizes phone digit strings between 7 and 13 digits."""
    if not text:
        return set()
    phones: Set[str] = set()
    patterns = [RE_PHONE_CANDIDATE, RE_PHONE_PREFIXED, RE_PHONE_STANDALONE]
    for pat in patterns:
        for m in pat.findall(text):
            raw = m if isinstance(m, str) else m[0]
            d = re.sub(r"\D", "", raw)
            if 7 <= len(d) <= 13:
                if len(d) > 10 and d.startswith("91"):
                    d = d[2:]
                elif len(d) > 10 and d.startswith("0"):
                    d = d[1:]
                elif len(d) > 10:
                    d = d[-10:]
                if 7 <= len(d) <= 10:
                    phones.add(d)
    return phones


def extract_base_domains(text: str) -> Set[str]:
    """Extracts base domains excluding subpaths, ports, and generic provider domains."""
    if not text:
        return set()
    domains: Set[str] = set()
    for match in RE_DOMAIN.findall(text):
        domain = match.lower().strip()
        domain = domain.split("/")[0].split(":")[0].strip()
        if "." in domain and len(domain) >= 4 and domain not in DOMAIN_BLACKLIST:
            domains.add(domain)
    return domains


def extract_postal_codes(address: str) -> Set[str]:
    """Extracts 5-digit and 6-digit postal/PIN tokens from address."""
    if not address:
        return set()
    return set(RE_POSTAL_5_6.findall(address))


def audit_ground_truth(sample_size: int = 50000, output_report_path: Path = None):
    """Audits ground truth file and source files."""
    t_start = time.time()
    logger.info("================================================================")
    logger.info("PHASE 1: GROUND TRUTH AUDIT & DISTRIBUTION ANALYSIS")
    logger.info("================================================================")

    logger.info(f"Loading Ground Truth from: {TRAIN_GT_PATH}")
    gt_df = pl.read_csv(str(TRAIN_GT_PATH), separator="\t", has_header=True)
    total_gt_entities = len(gt_df)

    # Classify Singletons vs Non-singletons
    gt_dict: Dict[str, Set[str]] = {}
    singleton_count = 0
    matched_entity_count = 0

    s2_only_count = 0
    s3_only_count = 0
    both_sources_count = 0

    match_count_distribution: Counter = Counter()
    s2_count_distribution: Counter = Counter()
    s3_count_distribution: Counter = Counter()

    for row in gt_df.iter_rows():
        s1_id = row[0]
        raw_matches = row[1] if len(row) > 1 and row[1] is not None else ""
        raw_matches = raw_matches.strip()
        if raw_matches:
            target_ids = {m.strip() for m in raw_matches.split(",") if m.strip()}
            gt_dict[s1_id] = target_ids
            matched_entity_count += 1

            has_s2 = any(t.startswith("S2-") for t in target_ids)
            has_s3 = any(t.startswith("S3-") for t in target_ids)

            if has_s2 and has_s3:
                both_sources_count += 1
            elif has_s2:
                s2_only_count += 1
            elif has_s3:
                s3_only_count += 1

            s2_c = sum(1 for t in target_ids if t.startswith("S2-"))
            s3_c = sum(1 for t in target_ids if t.startswith("S3-"))
            match_count_distribution[len(target_ids)] += 1
            s2_count_distribution[s2_c] += 1
            s3_count_distribution[s3_c] += 1
        else:
            gt_dict[s1_id] = set()
            singleton_count += 1

    singleton_pct = (singleton_count / total_gt_entities) * 100
    matched_pct = (matched_entity_count / total_gt_entities) * 100

    p_both = (both_sources_count / matched_entity_count * 100) if matched_entity_count else 0
    p_s2_only = (s2_only_count / matched_entity_count * 100) if matched_entity_count else 0
    p_s3_only = (s3_only_count / matched_entity_count * 100) if matched_entity_count else 0

    total_target_matches = sum(len(m) for m in gt_dict.values())
    avg_matches = total_target_matches / max(matched_entity_count, 1)

    logger.info(f"\n1. GROUND TRUTH SCALE & SINGLETON DISTRIBUTION:")
    logger.info(f"   Total S1 Entities           : {total_gt_entities:,}")
    logger.info(f"   True Singletons (Unmatched) : {singleton_count:,} ({singleton_pct:.2f}%)")
    logger.info(f"   Matched Entities            : {matched_entity_count:,} ({matched_pct:.2f}%)")
    logger.info(f"   Total Target Matches        : {total_target_matches:,}")
    logger.info(f"   Average Matches per Matched : {avg_matches:.2f}")

    logger.info(f"\n2. MULTI-MATCH SOURCE DISTRIBUTION (Across {matched_entity_count:,} matched entities):")
    logger.info(f"   S1 with matches in BOTH S2 & S3: {both_sources_count:,} ({p_both:.2f}%)")
    logger.info(f"   S1 with matches in S2 ONLY    : {s2_only_count:,} ({p_s2_only:.2f}%)")
    logger.info(f"   S1 with matches in S3 ONLY    : {s3_only_count:,} ({p_s3_only:.2f}%)")

    logger.info(f"\n   Match Cardinality Distribution:")
    for k in sorted(match_count_distribution.keys())[:10]:
        cnt = match_count_distribution[k]
        pct = (cnt / matched_entity_count) * 100
        logger.info(f"     {k} match(es) : {cnt:,} entities ({pct:.2f}%)")

    # PHASE 2: EXACT KEY LINKAGE RESOLUTION AUDIT
    logger.info("\n================================================================")
    logger.info("PHASE 2: DETERMINISTIC EXACT KEY RESOLUTION AUDIT")
    logger.info("================================================================")

    # Sample S1 entities for exact key audit
    if sample_size > 0 and sample_size < total_gt_entities:
        logger.info(f"Sampling {sample_size:,} S1 entities for exact key resolution analysis...")
        eval_s1_ids = list(gt_dict.keys())[:sample_size]
    else:
        logger.info(f"Evaluating ALL {total_gt_entities:,} S1 entities...")
        eval_s1_ids = list(gt_dict.keys())

    eval_s1_set = set(eval_s1_ids)
    eval_matched_s1_set = {sid for sid in eval_s1_ids if len(gt_dict[sid]) > 0}
    eval_gt_dict = {sid: gt_dict[sid] for sid in eval_s1_ids}

    # Collect needed target IDs for this S1 sample
    needed_target_ids: Set[str] = set()
    for sid in eval_s1_ids:
        needed_target_ids.update(gt_dict[sid])

    logger.info(f"S1 sample: {len(eval_s1_ids):,} entities ({len(eval_matched_s1_set):,} non-singletons).")
    logger.info(f"Target pool contains {len(needed_target_ids):,} ground truth positive target IDs.")

    # Load S1 records
    logger.info("Loading and normalizing S1 sample...")
    s1_df = (
        pl.scan_csv(str(TRAIN_S1_PATH), separator="\t")
        .filter(pl.col("entity_id").is_in(eval_s1_ids))
        .collect()
    )
    s1_data: Dict[str, Dict] = {}
    for r in s1_df.iter_rows(named=True):
        sid = r["entity_id"]
        cname = clean_business_name(r["business_name"] or "")
        caddr = clean_business_address(r["business_address"] or "")
        raw_text = f"{r['business_name'] or ''} {r['business_address'] or ''}"
        s1_data[sid] = {
            "country": r["country"],
            "clean_name": cname,
            "clean_addr": caddr,
            "phones": clean_phone_digits(raw_text),
            "domains": extract_base_domains(raw_text),
            "postals": extract_postal_codes(r["business_address"] or ""),
        }
    del s1_df
    gc.collect()

    # Load Target records (S2 + S3)
    logger.info("Loading and normalizing Target records (S2 + S3)...")
    s2_scan = pl.scan_csv(str(TRAIN_S2_PATH), separator="\t")
    s3_scan = pl.scan_csv(str(TRAIN_S3_PATH), separator="\t")

    # Load targets that are known ground truth matches + random background sample
    s2_pos = s2_scan.filter(pl.col("entity_id").is_in(list(needed_target_ids))).collect()
    s2_bg = s2_scan.limit(sample_size).collect()
    s2_targets = pl.concat([s2_pos, s2_bg]).unique(subset=["entity_id"])

    s3_pos = s3_scan.filter(pl.col("entity_id").is_in(list(needed_target_ids))).collect()
    s3_bg = s3_scan.limit(sample_size).collect()
    s3_targets = pl.concat([s3_pos, s3_bg]).unique(subset=["entity_id"])

    logger.info(f"Target sample assembled: S2={len(s2_targets):,}, S3={len(s3_targets):,}")

    # Build Exact Lookup Indices
    logger.info("Building Deterministic Exact Key Lookup Indices on Targets...")
    phone_index: Dict[Tuple[str, str], List[str]] = {}
    domain_index: Dict[Tuple[str, str], List[str]] = {}
    name_postal_index: Dict[Tuple[str, str, str], List[str]] = {}

    target_dfs = [("S2", s2_targets), ("S3", s3_targets)]
    for src_label, df in target_dfs:
        for r in df.iter_rows(named=True):
            tid = r["entity_id"]
            cntry = r["country"]
            cname = clean_business_name(r["business_name"] or "")
            raw_text = f"{r['business_name'] or ''} {r['business_address'] or ''}"

            phones = clean_phone_digits(raw_text)
            domains = extract_base_domains(raw_text)
            postals = extract_postal_codes(r["business_address"] or "")

            for ph in phones:
                phone_index.setdefault((cntry, ph), []).append(tid)

            for d in domains:
                domain_index.setdefault((cntry, d), []).append(tid)

            if cname and len(cname) >= 4:
                for p in postals:
                    name_postal_index.setdefault((cntry, cname, p), []).append(tid)

    del s2_targets, s3_targets
    gc.collect()

    logger.info(f"Unique Phone Keys Indexed       : {len(phone_index):,}")
    logger.info(f"Unique Domain Keys Indexed      : {len(domain_index):,}")
    logger.info(f"Unique Name+Postal Keys Indexed : {len(name_postal_index):,}")

    # Evaluate Resolution Potential and Precision
    logger.info("\nEvaluating resolution coverage and precision on S1 sample...")

    phone_tp, phone_fp = 0, 0
    domain_tp, domain_fp = 0, 0
    np_tp, np_fp = 0, 0
    union_tp, union_fp = 0, 0

    s1_resolved_by_phone: Set[str] = set()
    s1_resolved_by_domain: Set[str] = set()
    s1_resolved_by_np: Set[str] = set()
    s1_resolved_by_any: Set[str] = set()

    for sid in eval_s1_ids:
        if sid not in s1_data:
            continue
        info = s1_data[sid]
        cntry = info["country"]
        true_targets = eval_gt_dict.get(sid, set())

        # Phone matching
        p_matches: Set[str] = set()
        for ph in info["phones"]:
            for tid in phone_index.get((cntry, ph), []):
                p_matches.add(tid)
        if p_matches:
            s1_resolved_by_phone.add(sid)
            for tid in p_matches:
                if tid in true_targets:
                    phone_tp += 1
                else:
                    phone_fp += 1

        # Domain matching
        d_matches: Set[str] = set()
        for d in info["domains"]:
            for tid in domain_index.get((cntry, d), []):
                d_matches.add(tid)
        if d_matches:
            s1_resolved_by_domain.add(sid)
            for tid in d_matches:
                if tid in true_targets:
                    domain_tp += 1
                else:
                    domain_fp += 1

        # Exact Name + Postal matching
        np_matches: Set[str] = set()
        cname = info["clean_name"]
        if cname and len(cname) >= 4:
            for p in info["postals"]:
                for tid in name_postal_index.get((cntry, cname, p), []):
                    np_matches.add(tid)
        if np_matches:
            s1_resolved_by_np.add(sid)
            for tid in np_matches:
                if tid in true_targets:
                    np_tp += 1
                else:
                    np_fp += 1

        # Combined Union
        all_exact_matches = p_matches | d_matches | np_matches
        if all_exact_matches:
            s1_resolved_by_any.add(sid)
            for tid in all_exact_matches:
                if tid in true_targets:
                    union_tp += 1
                else:
                    union_fp += 1

    n_matched_eval = len(eval_matched_s1_set)
    logger.info(f"\n3. EXACT KEY LINKAGE RESOLUTION SUMMARY (Sample S1={len(eval_s1_ids):,}):")

    def calc_prec(tp, fp):
        return (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0

    p_prec = calc_prec(phone_tp, phone_fp)
    d_prec = calc_prec(domain_tp, domain_fp)
    np_prec = calc_prec(np_tp, np_fp)
    u_prec = calc_prec(union_tp, union_fp)

    logger.info(f"   [Key 1] Normalized Phone:")
    logger.info(f"     Resolved S1 Entities : {len(s1_resolved_by_phone):,} ({len(s1_resolved_by_phone)/n_matched_eval*100:.2f}% of matched)")
    logger.info(f"     Pairs (TP / FP)      : {phone_tp:,} / {phone_fp:,} -> Precision: {p_prec:.2f}%")

    logger.info(f"   [Key 2] Clean Domain/URL:")
    logger.info(f"     Resolved S1 Entities : {len(s1_resolved_by_domain):,} ({len(s1_resolved_by_domain)/n_matched_eval*100:.2f}% of matched)")
    logger.info(f"     Pairs (TP / FP)      : {domain_tp:,} / {domain_fp:,} -> Precision: {d_prec:.2f}%")

    logger.info(f"   [Key 3] Exact Name + Postal Code:")
    logger.info(f"     Resolved S1 Entities : {len(s1_resolved_by_np):,} ({len(s1_resolved_by_np)/n_matched_eval*100:.2f}% of matched)")
    logger.info(f"     Pairs (TP / FP)      : {np_tp:,} / {np_fp:,} -> Precision: {np_prec:.2f}%")

    logger.info(f"\n   [DETERMINISTIC UNION LAYER]:")
    logger.info(f"     Total S1 Resolved    : {len(s1_resolved_by_any):,} ({len(s1_resolved_by_any)/n_matched_eval*100:.2f}% of matched)")
    logger.info(f"     Pairs (TP / FP)      : {union_tp:,} / {union_fp:,} -> Precision: {u_prec:.2f}%")
    logger.info(f"     Residual S1 for ML   : {len(eval_s1_ids) - len(s1_resolved_by_any):,} ({(len(eval_s1_ids) - len(s1_resolved_by_any))/len(eval_s1_ids)*100:.2f}%)")

    report_data = {
        "total_s1_entities": total_gt_entities,
        "true_singleton_count": singleton_count,
        "true_singleton_pct": round(singleton_pct, 4),
        "matched_entity_count": matched_entity_count,
        "matched_entity_pct": round(matched_pct, 4),
        "both_sources_pct": round(p_both, 4),
        "s2_only_pct": round(p_s2_only, 4),
        "s3_only_pct": round(p_s3_only, 4),
        "avg_matches_per_matched_s1": round(avg_matches, 4),
        "deterministic_prepass": {
            "eval_sample_size": len(eval_s1_ids),
            "phone_resolved_entities": len(s1_resolved_by_phone),
            "phone_precision": round(p_prec, 4),
            "domain_resolved_entities": len(s1_resolved_by_domain),
            "domain_precision": round(d_prec, 4),
            "name_postal_resolved_entities": len(s1_resolved_by_np),
            "name_postal_precision": round(np_prec, 4),
            "union_resolved_entities": len(s1_resolved_by_any),
            "union_precision": round(u_prec, 4),
            "residual_ml_percentage": round((len(eval_s1_ids) - len(s1_resolved_by_any)) / len(eval_s1_ids) * 100, 2),
        },
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"\nAudit Report saved to: {output_report_path}")

    logger.info(f"\nAudit completed in {time.time() - t_start:.2f}s.")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — Ground Truth Audit")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of S1 entities to evaluate for exact resolution audit (0 = full dataset, default: 50000)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "audit_report.json"),
        help="Path to save output JSON audit report",
    )
    args = parser.parse_args()

    audit_ground_truth(
        sample_size=args.sample_size,
        output_report_path=Path(args.output_report) if args.output_report else None,
    )
