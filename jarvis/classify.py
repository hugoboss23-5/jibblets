"""
Query classification — maps text features to Chestahedron face weights.
No ML. Pure keyword/pattern analysis with stdlib only.
"""

import re
import sys
from .geometry import NUM_FACES, FACE_NAMES


# ---- Phrase patterns (processed FIRST, 2x weight) ----
# These override single-word matches to prevent misclassification.
_PHRASE_PATTERNS: dict[str, list[str]] = {
    "BREAK": [
        r"\bmake\s+sense\b", r"\bmakes?\s+sense\b",
        r"\bam\s+i\s+tripping\b", r"\bam\s+i\s+wrong\b", r"\bam\s+i\s+crazy\b",
        r"\bdoes\s+this\s+work\b", r"\bdoes\s+this\s+make\b",
        r"\bis\s+this\s+right\b", r"\bis\s+this\s+correct\b",
        r"\bwhat\s+do\s+you\s+think\b", r"\bwhat\s+are\s+the\s+risks?\b",
        r"\bcan\s+\w+\s+build\s+it\b", r"\bjust\s+yes\s+or\s+no\b",
        r"\bdoes\s+\w+\s+even\b",
    ],
    "LOAD": [
        r"\bwe\s+talked\s+about\b", r"\bwe\s+went\s+over\b",
        r"\byou\s+said\b", r"\byou\s+mentioned\b", r"\byou\s+told\s+me\b",
        r"\blast\s+chat\b", r"\bearlier\s+today\b",
        r"\bthat\s+thing\b", r"\bthe\s+thing\s+with\b",
        r"\bwhat\s+was\s+that\b",
        r"\bremind\s+me\b",
        r"\bin\s+my\s+drive\b", r"\bmy\s+email\b", r"\bmy\s+calendar\b", r"\bmy\s+slack\b",
        r"\bwhat\s+did\s+you\s+tell\b",
    ],
    "DIAGNOSE": [
        r"\bfigure\s+out\b",  # Not ROUTE
    ],
    "CUT": [
        r"\bsummariz\w*\b",  # "summarize" is strongly CUT, not just pattern-level
    ],
}

# Compile phrase patterns
_PHRASE_COMPILED: dict[str, list[re.Pattern]] = {
    face: [re.compile(p, re.IGNORECASE) for p in patterns]
    for face, patterns in _PHRASE_PATTERNS.items()
}

# ---- GENERATE exclusion phrases (single words that DON'T mean creation) ----
_GENERATE_EXCLUSIONS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bmake\s+sense\b", r"\bmakes?\s+sense\b",
        r"\bmake\s+sure\b", r"\bmake\s+it\s+work\b",
        r"\bmake\s+up\b", r"\bmake\s+do\b",
        r"\bdoes\s+this\s+make\b", r"\bdoesn'?t\s+make\b", r"\bcan'?t\s+make\b",
        r"\bmake\s+of\b",  # "what do you make of this"
    ]
]

# ---- Single-word pattern lists for each face ----
_PATTERNS: dict[str, list[str]] = {
    "DIAGNOSE": [
        r"\bwhat\s+is\b", r"\bwhy\s+does\b", r"\bhow\s+does\b", r"\bexplain\b",
        r"\bwhat\s+happened\b", r"\banalyze\b", r"\bcompare\b", r"\bevaluate\b",
        r"\bassess\b", r"\bwhat\s+are\b",
        r"\bdiagnose\b", r"\bunderstand\b", r"\bmeaning\b", r"\bdefine\b",
        r"\bwhy\b", r"\bhow\b",
    ],
    "LOAD": [
        r"\bremember\b", r"\bwe\s+discussed\b", r"\blast\s+time\b", r"\bpreviously\b",
        r"\bmy\s+project\b", r"\bour\s+work\b", r"\bthe\s+thing\b",
        r"\bfind\b", r"\bsearch\b", r"\blook\s+up\b", r"\bwhat\s+does\s+\w+\s+say\b",
        r"\bfile\b", r"\bdocument\b", r"\brepo\b", r"\blink\b", r"\burl\b",
        r"\bcontext\b", r"\bhistory\b", r"\bknowledge\b",
        r"https?://",
        # Past conversation references (Problem 3)
        r"\bwe\s+talked\b", r"\bwe\s+went\s+over\b",
        r"\byou\s+said\b", r"\byou\s+mentioned\b", r"\byou\s+told\s+me\b",
        r"\blast\s+chat\b", r"\bearlier\s+today\b",
        r"\bthat\s+thing\b", r"\bthe\s+thing\s+with\b",
        r"\bwhat\s+was\s+that\b",
        r"\bremind\s+me\b",
        # Tool triggers
        r"\bin\s+my\s+drive\b", r"\bmy\s+email\b", r"\bmy\s+calendar\b",
        r"\bcheck\s+(my|the)\b",
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
        r"\btake\s+care\b", r"\bhelp\s+me\b",
        r"\bplan\b", r"\bworkflow\b", r"\bprocess\b", r"\bpipeline\b",
        r"\bthen\s+create\b", r"\bthen\s+\w+\b.*\bthen\b",
    ],
}

# Compile single-word patterns once
_COMPILED: dict[str, list[re.Pattern]] = {
    face: [re.compile(p, re.IGNORECASE) for p in patterns]
    for face, patterns in _PATTERNS.items()
}


# ---- Trivial query detection (Fix B) ----
_TRIVIAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(hey|hi|hello|yo|sup|what'?s?\s*(up|good|happening|new|poppin))[\s!?.]*$",
        r"^yo\s+what'?s?\s*(up|good|happening|poppin)[\s!?.]*$",
        r"^(good\s*(morning|afternoon|evening|night))[\s!?.]*$",
        r"^(thanks?|thank\s*you|thx|ty|appreciate\s+it)(\s+\w+)*[\s!?.]*$",
        r"^(ok|okay|got\s*it|understood|cool|nice|bet|dope|word|aight|yep|yup|nah)[\s!?.]*$",
        r"^(how\s+are\s+you|how'?s?\s+it\s+going)[\s!?.]*$",
        r"^(gm|gn|lol|lmao|haha)[\s!?.]*$",
    ]
]


def is_trivial(query: str) -> bool:
    """Detect greetings and trivial queries that need no tools."""
    q = query.strip()
    if len(q) > 40:
        return False
    return any(p.match(q) for p in _TRIVIAL_PATTERNS)


# ---- Question-type detection (Fix: Problem 2 part 4) ----
_EVAL_QUESTION = re.compile(
    r"\b(does|is|can|should|would|could|will|did|has|have|are)\b.*\?",
    re.IGNORECASE
)


def _is_evaluation_question(query: str) -> bool:
    """Detect questions asking for evaluation (DIAGNOSE/BREAK) not creation (GENERATE)."""
    return bool(_EVAL_QUESTION.search(query))


# ---- Past-reference detection for LOAD boost (Fix C) ----
_PAST_REF_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwe\b", r"\btalked\s+about\b", r"\blast\s+time\b",
        r"\bremember\b", r"\bpreviously\b", r"\byou\s+said\b",
        r"\byou\s+told\b", r"\byou\s+mentioned\b", r"\bthat\s+thing\b",
        r"\bwhat\s+was\b.*\b(that|the)\b", r"\bwhat\s+did\s+you\b",
    ]
]


def has_past_reference(query: str) -> bool:
    """Detect queries referencing past conversations."""
    return any(p.search(query) for p in _PAST_REF_PATTERNS)


def classify_query(query: str, history: list[dict] | None = None) -> list[float]:
    """
    Analyze query text and conversation history to produce a 7D face weight vector.
    Returns list of 7 floats (one per face) that sum to 1.0.
    Minimum weight per face = 0.02 (no face is ever fully off).
    """
    history = history or []
    query_lower = query.lower().strip()

    scores = [0.0] * NUM_FACES

    # --- Phase 1: Phrase patterns (2x weight, processed FIRST) ---
    phrase_hits = set()  # Track which text ranges matched phrases
    for face_name, patterns in _PHRASE_COMPILED.items():
        face_idx = FACE_NAMES.index(face_name)
        for pattern in patterns:
            matches = pattern.findall(query_lower)
            if matches:
                scores[face_idx] += len(matches) * 2.0  # 2x weight
                phrase_hits.update(m.lower() if isinstance(m, str) else m for m in matches)

    # --- Phase 2: Check GENERATE exclusions ---
    has_generate_exclusion = any(p.search(query_lower) for p in _GENERATE_EXCLUSIONS)

    # --- Phase 3: Single-word patterns ---
    for i, face_name in enumerate(FACE_NAMES):
        patterns = _COMPILED.get(face_name, [])
        for pattern in patterns:
            matches = pattern.findall(query_lower)
            count = len(matches)
            if count == 0:
                continue

            # If this is GENERATE and exclusion phrases are present,
            # suppress the match for excluded words
            if face_name == "GENERATE" and has_generate_exclusion:
                # Only suppress "make" specifically, not all GENERATE patterns
                if pattern.pattern in (r"\bmake\b",):
                    continue

            scores[i] += count

    # --- Phase 4: Question-type detector ---
    if _is_evaluation_question(query):
        # Evaluation questions boost DIAGNOSE/BREAK, suppress GENERATE
        scores[0] += 1.0  # DIAGNOSE
        scores[4] += 0.5  # BREAK
        scores[3] = max(0, scores[3] - 0.5)  # Suppress GENERATE

    # --- Phase 5: Additional modifiers ---

    # Long queries boost DIAGNOSE
    if len(query) > 200:
        scores[0] += 1.5

    # URLs boost LOAD
    if re.search(r"https?://", query):
        scores[1] += 3.0  # Strong LOAD signal

    # Code/backticks boost GENERATE and BREAK
    if "`" in query or "```" in query:
        scores[3] += 1.5
        scores[4] += 1.0

    # Question marks boost DIAGNOSE
    qmark_count = query.count("?")
    if qmark_count > 0:
        scores[0] += 0.5 * qmark_count

    # Past-reference queries get strong LOAD boost (Fix Problem 3 + C)
    # BUT only when CUT isn't already winning (user wants summary, not retrieval)
    if has_past_reference(query) and scores[5] < 1.0:
        scores[1] += 2.5  # Strong LOAD signal overrides DIAGNOSE "what"

    # History-based adjustments
    if history:
        if len(history) >= 3:
            recent_hashes = [h.get("query_hash", "") for h in history[-3:]]
            if len(set(recent_hashes)) < len(recent_hashes):
                scores[5] += 2.0

        recent_qualities = [h.get("quality", 0.5) for h in history[-3:] if "quality" in h]
        if recent_qualities and sum(recent_qualities) / len(recent_qualities) < 0.5:
            scores[0] += 1.0
            scores[2] += 1.0

    # If no patterns matched at all, give a baseline to DIAGNOSE and ROUTE
    if sum(scores) < 0.01:
        scores[0] = 1.0
        scores[6] = 0.5

    # --- Normalize with minimum floor ---
    MIN_WEIGHT = 0.02

    total = sum(scores)
    if total > 0:
        weights = [s / total for s in scores]
    else:
        weights = [1.0 / NUM_FACES] * NUM_FACES

    # Enforce floor
    for _ in range(3):
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
    # New cases for Problems 2 and 3
    ("does this architecture even make sense or am I tripping", "BREAK"),
    ("what was that thing we talked about with the chestahedron", "LOAD"),
    ("what did you tell me about the receiver last time", "LOAD"),
    ("make me a pitch deck for SUNLIGHT", "GENERATE"),
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
            log(f"  PASS: \"{query[:55]}\" → {dominant_name}")
        else:
            failed += 1
            desc = classify_and_describe(query)
            log(f"  FAIL: \"{query[:55]}\" → {dominant_name} (expected {expected_dominant})")
            log(f"         weights: {desc}")

        # Verify constraints
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights don't sum to 1: {sum(weights)}"
        assert all(w >= 0.02 - 1e-6 for w in weights), f"Weight below minimum: {min(weights)}"

    # Test trivial detection
    trivial_cases = [
        ("yo whats good", True),
        ("thanks that was helpful", True),
        ("build me a REST API", False),
        ("hey", True),
        ("ok", True),
    ]
    for query, expected in trivial_cases:
        result = is_trivial(query)
        if result == expected:
            passed += 1
            log(f"  PASS: is_trivial(\"{query}\") = {result}")
        else:
            failed += 1
            log(f"  FAIL: is_trivial(\"{query}\") = {result} (expected {expected})")

    log(f"\n=== Results: {passed}/{passed + failed} passed ===")
    return failed == 0


if __name__ == "__main__":
    success = self_test()
    sys.exit(0 if success else 1)
