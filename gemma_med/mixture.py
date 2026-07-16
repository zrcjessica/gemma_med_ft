"""The MedGemma-proxy text mixture.

Grounded in the MedGemma technical report (arXiv:2507.05201), Table 1. The
report is explicit that no medical text was added during pretraining and that
27B-text used "the post-training stage alone" -- so an SFT-only reproduction of
the *text* recipe is faithful in kind, not a shortcut.

Two components of the real mix are not reproducible and are substituted here:

  1. 200k synthetic questions -- never released. Substituted with public
     reasoning corpora (ReasonMed / medical-o1 / MedReason).
  2. Distillation on teacher *logits* from an internal IT model. No public API
     exposes logits, so we do response-level SFT on public CoT instead. This is
     the single biggest fidelity gap and the main reason our numbers should not
     be expected to land on MedGemma's.

HealthSearchQA (3,375 in Table 1) is questions-only in every public release, so
it cannot be used for SFT. It is registered as eval-only.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    hf_id: str
    config: str | None = None
    split: str = "train"
    # Count reported in MedGemma Table 1, where the source is part of the real mix.
    paper_n: int | None = None
    license: str = "unknown"
    # "tier1" = named in Table 1; "synthetic_proxy" = stands in for the 200k
    # synthetic set; "replay" = general-instruction anti-forgetting data.
    role: str = "tier1"
    # Requires an approved access request on the Hub. build_mixture() warns and
    # skips these rather than failing the whole build.
    gated: bool = False
    notes: str = ""


# Named in MedGemma Table 1 and publicly downloadable. Row counts for MedExpQA
# (434) and LiveQA (635 vs 634) match the paper almost exactly, which indicates
# Google used the public splits close to as-is.
TIER1: list[Source] = [
    Source("medqa", "GBaker/MedQA-USMLE-4-options", paper_n=9275, license="CC-BY-4.0",
           notes="Paper's 9,275 < public ~10,178 train; Google filtered ~9% by an unstated method."),
    Source("medmcqa", "openlifescienceai/medmcqa", paper_n=182806, license="Apache-2.0"),
    Source("pubmedqa", "qiaojin/PubMedQA", config="pqa_labeled", paper_n=1000, license="MIT"),
    Source("medexpqa", "HiTZ/MedExpQA", config="en", paper_n=434, license="CC-BY-4.0"),
    Source("afrimedqa", "intronhealth/afrimedqa_v2", paper_n=1003, license="CC-BY-SA-4.0", gated=True,
           notes="Gated despite the CC-BY-SA license: request access at "
                 "huggingface.co/datasets/intronhealth/afrimedqa_v2. Only ~1k of ~195k rows, "
                 "so the build warns and continues without it."),
    Source("liveqa", "truehealth/liveqa", paper_n=634, license="unstated"),
]

# Stand-ins for the unreleased 200k synthetic distillation set.
SYNTHETIC_PROXY: list[Source] = [
    Source("medical_o1", "FreedomIntelligence/medical-o1-reasoning-SFT", config="en",
           license="Apache-2.0", role="synthetic_proxy",
           notes="90k GPT-4o CoT, verifier-checked (HuatuoGPT-o1)."),
    Source("medreason", "UCSC-VLAA/MedReason", license="Apache-2.0", role="synthetic_proxy",
           notes="33k knowledge-graph-grounded chains; different failure mode from pure CoT."),
    Source("reasonmed", "lingshu-medical-mllm/ReasonMed", license="Apache-2.0", role="synthetic_proxy",
           notes="1.1M; largest available. Subsampled -- see MixtureSpec.cap."),
]

# The report documents a real alignment tax (MMLU Pro 43.6 -> 39.1 at 4B,
# 67.5 -> 60.2 at 27B). Replay data is our lever against it.
REPLAY: list[Source] = [
    Source("aloe_general", "HPAI-BSC/Aloe-Beta-General-Collection",
           license="Apache-2.0", role="replay"),
]

# Questions with no reference answers -> unusable for SFT. Eval only.
EVAL_ONLY: list[Source] = [
    Source("healthsearchqa", "katielink/healthsearchqa", split="train", paper_n=3375,
           license="unknown", role="eval_only",
           notes="Questions only, no answers. Public row counts conflict (3,173 vs 4,576)."),
]

ALL_SOURCES = {s.key: s for s in TIER1 + SYNTHETIC_PROXY + REPLAY + EVAL_ONLY}


@dataclass
class MixtureSpec:
    """Which sources to build, and how many rows to take from each."""
    sources: list[str]
    # Per-source row cap applied after dedup. ReasonMed alone is 1.1M and would
    # otherwise drown the Table 1 data.
    cap: dict[str, int] = field(default_factory=dict)
    seed: int = 42


# Table 1 text rows only (~195k). The closest we can get to the literal paper
# mix, minus the 200k synthetic set. Useful as a fidelity reference point.
PAPER_ONLY = MixtureSpec(sources=[s.key for s in TIER1])

# Default: Table 1 + a 200k synthetic stand-in, sized to mirror the paper's
# roughly 1:1 ratio of real QA (195k) to synthetic (200k), + 10% replay.
DEFAULT = MixtureSpec(
    sources=[s.key for s in TIER1] + ["medical_o1", "medreason", "reasonmed", "aloe_general"],
    cap={"reasonmed": 80_000, "aloe_general": 40_000},
)

# Cheap mixture for the 1b LR sweep and pipeline smoke tests.
SMALL = MixtureSpec(
    sources=["medqa", "pubmedqa", "medexpqa", "medical_o1"],
    cap={"medical_o1": 5_000},
)

MIXTURES = {"paper_only": PAPER_ONLY, "default": DEFAULT, "small": SMALL}
