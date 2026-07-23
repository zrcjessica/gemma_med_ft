"""Assemble the frozen faithfulness probe set for the jlens trajectory probe.

Output is `category<TAB>prompt`, consumed by probe_jlens.py's concordance().
The metric compares the lens read-out against the MODEL's own next token
(argmax + KL) at the last position -- there are no answer labels, so a prompt
only needs a determined continuation, not a correct one.

Three arms:
  general   -- one-hop factual cloze, the matched control
  medical   -- one-hop medical cloze; the study's own contribution (no upstream
               equivalent). Kept style-matched to `general` so the general-vs-
               medical contrast isolates domain, not prompt format.
  multihop  -- jlens's own two-hop factual set (data/evaluations/lens-eval-
               multihop.json), pulled verbatim. A curated reference arm that
               also sanity-checks our setup against the paper's prompts.

general/medical are hand-authored here because the whole point is a frozen,
style-matched control pair; multihop comes from jlens so we don't hand-roll what
the reference already ships. Frozen once written -- regenerating changes the
measurement basis for every checkpoint/size/arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GENERAL = [
    "The capital city of France is",
    "The chemical symbol for the element gold is",
    "The largest planet in our solar system is",
    "The author of the play Romeo and Juliet is",
    "The tallest mountain above sea level on Earth is",
    "Water is composed of hydrogen and the element",
    "The number of continents on Earth is",
    "The ocean between Europe and North America is called the",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per",
    "The currency used in Japan is called the",
    "The first president of the United States was",
    "The powerhouse of the cell is commonly called the",
    "The gas that plants absorb from the air for photosynthesis is",
    "The freezing point of water in degrees Celsius is",
    "The painter of the Mona Lisa was Leonardo da",
    "The chemical symbol for the element oxygen is",
    "The largest mammal on Earth is the blue",
    "The capital city of Japan is",
    "The planet known as the Red Planet is",
    "The primary language spoken in Brazil is",
    "The author of the theory of general relativity was Albert",
    "The metal that is liquid at room temperature is",
    "The number of days in a leap year is",
    "The common currency used across most of the European Union is the",
    "The tallest living land animal is the",
    "The first person to walk on the Moon was Neil",
    "The longest river in the world is generally considered the",
    "The hardest naturally occurring material is",
    "The gas that makes up most of Earth's atmosphere is",
    "The capital city of Italy is",
    "The organ in the human body that pumps blood is the",
    "The country that gifted the Statue of Liberty to the United States was",
    "The largest ocean on Earth is the",
    "The scientist who formulated the three laws of motion was Isaac",
    "The capital city of Russia is",
    "The chemical symbol for the element iron is",
    "The closest star to the Earth is the",
    "The author of the tragedy Hamlet was William",
    "The number of degrees in a right angle is",
    "The primary gas responsible for the greenhouse effect from human activity is carbon",
    "The continent on which the Sahara Desert is located is",
    "The number of sides on a hexagon is",
    "The element with the atomic number one is",
    "The largest internal organ in the human body is the",
    "The force that pulls objects toward the center of the Earth is",
    "The capital city of Canada is",
    "The primary colors of light are red, green, and",
    "The planet closest to the Sun is",
    "The study of living organisms is called",
    "The chemical process by which plants convert sunlight into energy is called",
]

MEDICAL = [
    "The first-line pharmacological treatment for type 2 diabetes is typically the drug",
    "The most common causative organism of community-acquired bacterial pneumonia is",
    "The vitamin whose deficiency causes scurvy is vitamin",
    "The hormone that lowers blood glucose by promoting cellular uptake is",
    "The classic triad of symptoms in diabetic ketoacidosis includes hyperglycemia, ketosis, and",
    "The antidote used to reverse an opioid overdose is",
    "The bacterium responsible for most peptic ulcers is Helicobacter",
    "A normal resting adult heart rate in beats per minute is roughly",
    "The imaging study of first choice for suspected acute ischemic stroke is a non-contrast",
    "The most common cause of chronic obstructive pulmonary disease is cigarette",
    "The electrolyte abnormality most associated with peaked T waves on an ECG is elevated",
    "The anticoagulant that requires monitoring of the INR is",
    "The nerve responsible for innervating the diaphragm is the",
    "The autoimmune destruction of pancreatic beta cells causes type 1",
    "The first-line treatment for anaphylaxis is intramuscular",
    "The organ primarily responsible for filtering blood and producing urine is the",
    "The vitamin synthesized in the skin upon exposure to sunlight is vitamin",
    "The medical term for a heart attack is myocardial",
    "The largest artery in the human body is the",
    "The number of chambers in the human heart is",
    "The white blood cells that mature into antibody-producing plasma cells are the B",
    "The protein that carries oxygen within red blood cells is",
    "The medical term for chronically elevated blood pressure is",
    "The medical term for abnormally low blood glucose is",
    "The neurotransmitter that is deficient in Parkinson disease is",
    "The most common chemical composition of kidney stones is calcium",
    "The medical term for shortness of breath or difficulty breathing is",
    "The small cell fragments in blood responsible for clotting are called",
    "The medical term for inflammation of the appendix is",
    "The hormone secreted by the thyroid gland that regulates metabolism is",
    "The virus that causes acquired immunodeficiency syndrome is",
    "The region of the brain responsible for balance and coordination is the",
    "The muscle that separates the thoracic cavity from the abdominal cavity is the",
    "The medical term for inflammation of the liver is",
    "The valve located between the left atrium and the left ventricle is the",
    "The condition of a deficiency in red blood cells or hemoglobin is called",
    "The class of antibiotics to which penicillin belongs is the beta",
    "The gland that regulates blood calcium by secreting parathyroid hormone is the",
    "The reversal agent given for warfarin toxicity is vitamin",
    "The medical term for the voice box is the",
    "The blood test that reflects average glucose over three months is hemoglobin",
    "The most common cause of preventable death worldwide is tobacco",
    "The type of diabetes strongly associated with obesity and insulin resistance is type",
    "The pigment that gives skin, hair, and eyes their color is",
    "The medical term for surgical removal of the gallbladder is",
    "The chamber of the heart that pumps oxygenated blood to the body is the left",
    "The most abundant type of cell in human blood is the red blood",
    "The medical specialty concerned with the care of newborn infants is",
    "The vitamin whose deficiency causes rickets in children is vitamin",
    "The leading cause of cancer-related death in both men and women is cancer of the",
]


def load_multihop(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    return [it["prompt"].strip() for it in items if it.get("prompt", "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--multihop-json",
                    default=str(Path.home() / "jacobian-lens/data/evaluations/lens-eval-multihop.json"),
                    help="jlens reference multihop set; omit-safe if absent.")
    args = ap.parse_args()

    rows: list[tuple[str, str]] = []
    rows += [("general", p) for p in GENERAL]
    rows += [("medical", p) for p in MEDICAL]

    mh_path = Path(args.multihop_json)
    n_mh = 0
    if mh_path.exists():
        mh = load_multihop(mh_path)
        rows += [("multihop", p) for p in mh]
        n_mh = len(mh)
    else:
        print(f"WARNING: multihop set not found at {mh_path}; writing without it")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{c}\t{p}" for c, p in rows) + "\n")

    counts = {c: sum(1 for cc, _ in rows if cc == c) for c in dict.fromkeys(c for c, _ in rows)}
    manifest = {
        "n_total": len(rows),
        "counts": counts,
        "general_medical": "hand-authored one-hop cloze, style-matched control pair",
        "multihop_source": f"jlens {mh_path.name}" if n_mh else None,
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
