"""
Query classification — maps text features to Chestahedron face weights.
No ML. Pure keyword/pattern analysis with stdlib only.
"""

import re
import sys
from .geometry import NUM_FACES, FACE_NAMES


# Pattern lists for each face
_PATTERNS: dict[str, list[str]] = {
    "DIAGNOSE": [
        r"\bwhat\s+is\b", r"\bwhy\s+does\b", r"\bhow\s+does\b", r"\bexplain\b",
        r"\bwhat\s+happened\b", r"\banalyze\b", r"\bcompare\b", r"\bevaluate\b",
        r"\bassess\b", r"\bwhat\s+do\s+you\s+think\b", r"\bwhat\s+are\b",
        r"\bdiagnose\b", r"\bunderstand\b", r"\bmeaning\b", r"\bdefine\b",
        r"\bwhy\b", r"\bhow\b",
    ],
    "LOAD": [
        r"\bremember\b", r"\bwe\s+discussed\b", r"\blast\s+time\b", r"\bpreviously\b",
        r"\bmy\s+project\b", r"\bour\s+work\b", r"\bthe\s+thing\b",
        r"\bfind\b", r"\bsearch\b", r"\blook\s+up\b", r"\bwhat\s+does\s+\w+\s+say\b",
        r"\bfile\b", r"\bdocument\b", r"\brepo\b", r"\blink\b", r"\burl\b",
        r"\bcontext\b", r"\bhistory\b", r"\bknowledge\b",
        r"https?://",  # URLs
    ],
    "CONSTRAIN": [
        r"\bonly\b", r"\bjust\b", r"\bdon'?t\b", r"\bwithout\b", r"\blimit\b",
        r"\bfocus\s+on\b", r"\bkeep\s+it\s+short\b", r"\bbullet\s+points?\b",
        r"\bone\s+paragraph\b", r"\bnot\s+\w+\b", r"\bavoid\b", r"\bexclude\b",
        r"\bbrief\b", r"\bconcise\b", r"\bno\s+more\s+than\b", r"\bsimple\b",
    ],
    "GENERATE": [
        r"\bbuild\b", r"\bcreate\b", r"\bmake\b", r"\bwrite\b", r"\bdesign\b",
        r"\bdraft\b", r"\bideas?\b", r"\boptions?\b", r"\bpossibilit", r"\bwhat\s+if\b",
        r"\bhow\s+could\b", r"\bimplement\b", r"\bcode\b", r"\bfunction\b",
        r"\bscript\b", r"\bapp\b", r"\bgenerate\b", r"\bbrainstorm\b",
        r"\bcompose\b", r"\bprogram\b",
    ],
    "BREAK": [
        r"\bwhat'?s\s+wrong\b", r"\bcritique\b", r"\bfind\s+flaws?\b",
        r"\bfind\b.*\bflaws?\b", r"\bflaws?\b",
        r"\bargue\s+against\b", r"\bwhy\s+wouldn'?t\b", r"\brisks?\b",
        r"\btest\b", r"\bverify\b", r"\bcheck\b", r"\bdoes\s+this\s+work\b",
        r"\bedge\s+cases?\b", r"\bproblem\b", r"\bweak\b",
        r"\bchallenge\b", r"\bstress\b", r"\bwrong\b",
    ],
    "CUT": [
        r"\bwhich\s+one\b", r"\bbest\s+option\b", r"\brecommend\b", r"\bchoose\b",
        r"\bsummar", r"\btldr\b", r"\bkey\s+points?\b", r"\bbottom\s+line\b",
        r"\bpick\b", r"\brank\b", r"\bprioritize\b", r"\bmost\s+important\b",
        r"\bdecide\b", r"\bselect\b", r"\btop\s+\d\b",
    ],
    "ROUTE": [
        r"\bfirst\b.*\bthen\b", r"\bfirst\b.*,\s*then\b",
        r"\bstep\s+\d\b", r"\bsearch\s+for\b",
        r"\bcreate\s+a\s+file\b", r"\bcheck\s+my\b", r"\bhandle\s+this\b",
        r"\btake\s+care\b", r"\bfigure\s+out\b", r"\bhelp\s+me\b",
        r"\bplan\b", r"\bworkflow\b", r"\bprocess\b", r"\bpipeline\b",
        r"\bthen\s+create\b", r"\bthen\s+\w+\b.*\bthen\b",
    ],
}

# Compile patterns once
_COMPILED: dict[str, list[re.Pattern]] = {
    face: [re.compile(p, re.IGNORECASE) for p in patterns]
    for face, patterns in _PATTERNS.items()
}


def classify_query(query: str, history: list[dict] | None = None) -> list[float]:
    """
    Analyze query text and conversation history to produce a 7D face weight vector.
    Returns list of 7 floats (one per face) that sum to 1.0.
    Minimum weight per face = 0.02 (no face is ever fully off).
    """
    history = history or []
    query_lower = query.lower().strip()

    # Base scores from pattern matching
    scores = [0.0] * NUM_FACES
    for i, face_name in enumerate(FACE_NAMES):
        patterns = _COMPILED.get(face_name, [])
        for pattern in patterns:
            matches = pattern.findall(query_lower)
            scores[i] += len(matches)

    # --- Additional modifiers ---

    # Long queries boost DIAGNOSE (complex input needs analysis)
    if len(query) > 200:
        scores[0] += 1.5  # DIAGNOSE

    # URLs boost LOAD
    if re.search(r"https?://", query):
        scores[1] += 2.0  # LOAD

    # Code/backticks boost GENERATE and BREAK
    if "`" in query or "```" in query:
        scores[3] += 1.5  # GENERATE
        scores[4] += 1.0  # BREAK

    # Question marks boost DIAGNOSE
    qmark_count = query.count("?")
    if qmark_count > 0:
        scores[0] += 0.5 * qmark_count

    # History-based adjustments
    if history:
        # Repeated similar queries → boost CUT (user wants convergence)
        if len(history) >= 3:
            recent_hashes = [h.get("query_hash", "") for h in history[-3:]]
            if len(set(recent_hashes)) < len(recent_hashes):
                scores[5] += 2.0  # CUT

        # Low quality in recent history → boost DIAGNOSE and CONSTRAIN
        recent_qualities = [h.get("quality", 0.5) for h in history[-3:] if "quality" in h]
        if recent_qualities and sum(recent_qualities) / len(recent_qualities) < 0.5:
            scores[0] += 1.0  # DIAGNOSE
            scores[2] += 1.0  # CONSTRAIN

    # If no patterns matched at all, give a baseline to DIAGNOSE and ROUTE
    if sum(scores) < 0.01:
        scores[0] = 1.0  # DIAGNOSE
        scores[6] = 0.5  # ROUTE

    # --- Normalize with minimum floor ---
    MIN_WEIGHT = 0.02

    # First normalize scores to sum to 1
    total = sum(scores)
    if total > 0:
        weights = [s / total for s in scores]
    else:
        weights = [1.0 / NUM_FACES] * NUM_FACES

    # Enforce floor: redistribute from above-floor weights to bring all up
    for _ in range(3):  # Iterate to convergence
        deficit = 0.0
        above_floor_total = 0.0
        for i in range(NUM_FACES):
            if weights[i] < MIN_WEIGHT:
                deficit += MIN_WEIGHT - weights[i]
                weights[i] = MIN_WEIGHT
            else:
                above_floor_total += weights[i]
        if deficit > 0 and above_floor_total > 0:
            scale = (above_floor_total - deficit) / above_floor_total
            for i in range(NUM_FACES):
                if weights[i] > MIN_WEIGHT:
                    weights[i] *= scale

    # Final normalization to exactly 1.0
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    return weights


def classify_and_describe(query: str, history: list[dict] | None = None) -> dict:
    """Classify and return a rich description with face names and weights."""
    weights = classify_query(query, history)
    result = {}
    for i, (name, weight) in enumerate(zip(FACE_NAMES, weights)):
        result[name] = round(weight, 4)
    return result


# ---- Self-Test ----

_TEST_CASES = [
    ("What is quantum computing?", "DIAGNOSE"),
    ("Remember what we discussed about the project last time?", "LOAD"),
    ("Keep it short, just bullet points, no fluff", "CONSTRAIN"),
    ("Build me a REST API in Python with authentication", "GENERATE"),
    ("What's wrong with this approach? Find the flaws.", "BREAK"),
    ("Which option is best? Summarize the top 3.", "CUT"),
    ("First search for recent papers, then create a summary file", "ROUTE"),
    ("Write code to implement a binary search tree", "GENERATE"),
    ("Analyze and compare these two architectures", "DIAGNOSE"),
    ("Look up what our knowledge graph says about topology", "LOAD"),
]


def self_test() -> bool:
    log = lambda msg: print(msg, file=sys.stderr)
    log("=== Classify Self-Test ===")
    passed = 0
    failed = 0

    for query, expected_dominant in _TEST_CASES:
        weights = classify_query(query)
        dominant_idx = weights.index(max(weights))
        dominant_name = FACE_NAMES[dominant_idx]

        ok = dominant_name == expected_dominant
        if ok:
            passed += 1
            log(f"  PASS: \"{query[:50]}...\" → {dominant_name}")
        else:
            failed += 1
            desc = classify_and_describe(query)
            log(f"  FAIL: \"{query[:50]}...\" → {dominant_name} (expected {expected_dominant})")
            log(f"         weights: {desc}")

        # Verify constraints
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights don't sum to 1: {sum(weights)}"
        assert all(w >= 0.02 - 1e-6 for w in weights), f"Weight below minimum: {min(weights)}"

    log(f"\n=== Results: {passed}/{passed + failed} passed ===")
    return failed == 0


if __name__ == "__main__":
    success = self_test()
    sys.exit(0 if success else 1)
