"""
14-query verification test for the JARVIS routing cortex calibration.
Verifies all 7 problems + additional fixes are resolved.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jarvis.route import route_query
from jarvis.classify import classify_query, is_trivial, FACE_NAMES
from jarvis.geometry import self_test as geo_self_test


def run_verification():
    log = lambda msg: print(msg, file=sys.stderr)
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            log(f"  PASS: {name}")
        else:
            failed += 1
            log(f"  FAIL: {name} — {detail}")

    # Use a temp state file for isolation
    td = tempfile.mkdtemp()
    sf = os.path.join(td, "test_state.json")

    log("=== 14-Query Verification Test ===\n")

    # ---- 1. Trivial queries → NO tools ----
    log("--- Trivial queries (no tools) ---")

    r = route_query("yo whats good", state_file=sf)
    check("Q1: 'yo whats good' tools_needed=false",
          r["tools_needed"] == False,
          f"tools_needed={r['tools_needed']}, tools={len(r['tool_sequence'])}")

    r = route_query("thanks that was helpful", state_file=sf)
    check("Q2: 'thanks that was helpful' tools_needed=false",
          r["tools_needed"] == False,
          f"tools_needed={r['tools_needed']}, tools={len(r['tool_sequence'])}")

    # ---- 2. BREAK not GENERATE (phrase awareness) ----
    log("\n--- Phrase awareness ---")

    r = route_query("does this architecture even make sense or am I tripping", state_file=sf)
    dom = r["dominant_operations"]
    check("Q3: 'make sense' → BREAK dominant",
          dom[0] == "BREAK",
          f"dominant={dom[:3]}")

    r = route_query("Can Claude code build it right now? Just yes or no.", state_file=sf)
    dom = r["dominant_operations"]
    check("Q4: evaluation question → DIAGNOSE or BREAK dominant",
          dom[0] in ("DIAGNOSE", "BREAK"),
          f"dominant={dom[:3]}")

    # ---- 3. LOAD with conversation_search high ----
    log("\n--- Memory retrieval (LOAD + conversation_search) ---")

    r = route_query("what was that thing we talked about with the chestahedron and plasma", state_file=sf)
    dom = r["dominant_operations"]
    tools = [t["tool"] for t in r["tool_sequence"]]
    check("Q5: past-reference → LOAD dominant",
          dom[0] == "LOAD",
          f"dominant={dom[:3]}")
    # conversation_search should be in top 3 tools
    conv_rank = tools.index("conversation_search") if "conversation_search" in tools else 99
    check("Q5: conversation_search in top 3",
          conv_rank < 3,
          f"conversation_search at rank {conv_rank}, tools={tools[:4]}")

    r = route_query("what did you tell me about the receiver last time", state_file=sf)
    dom = r["dominant_operations"]
    tools = [t["tool"] for t in r["tool_sequence"]]
    check("Q6: 'you told me last time' → LOAD dominant",
          dom[0] == "LOAD",
          f"dominant={dom[:3]}")
    conv_rank = tools.index("conversation_search") if "conversation_search" in tools else 99
    check("Q6: conversation_search in top 3",
          conv_rank < 3,
          f"conversation_search at rank {conv_rank}, tools={tools[:4]}")

    # ---- 4. GENERATE with create_file high ----
    log("\n--- Creation requests (GENERATE + create_file) ---")

    r = route_query("make me a pitch deck for SUNLIGHT", state_file=sf)
    dom = r["dominant_operations"]
    check("Q7: 'make me a pitch deck' → GENERATE dominant",
          dom[0] == "GENERATE",
          f"dominant={dom[:3]}")
    tools = [t["tool"] for t in r["tool_sequence"]]
    check("Q7: create_file in tools",
          "create_file" in tools,
          f"tools={tools[:5]}")

    r = route_query("write a python script that scrapes procurement data", state_file=sf)
    dom = r["dominant_operations"]
    check("Q8: 'write a python script' → GENERATE dominant",
          dom[0] == "GENERATE",
          f"dominant={dom[:3]}")
    tools = [t["tool"] for t in r["tool_sequence"]]
    check("Q8: create_file in tools",
          "create_file" in tools,
          f"tools={tools[:5]}")

    # ---- 5. CUT ----
    log("\n--- Compression/summary (CUT) ---")

    r = route_query("summarize everything we have done today in one paragraph", state_file=sf)
    dom = r["dominant_operations"]
    check("Q9: 'summarize' → CUT dominant",
          dom[0] == "CUT",
          f"dominant={dom[:3]}")

    # ---- 6. URL → LOAD ----
    log("\n--- URL detection ---")

    r = route_query("https://github.com/hugoboss23-5/jibblets check this repo", state_file=sf)
    dom = r["dominant_operations"]
    check("Q10: URL → LOAD dominant",
          dom[0] == "LOAD",
          f"dominant={dom[:3]}")

    # ---- 7. DIAGNOSE (analytical) ----
    log("\n--- Analytical question (DIAGNOSE) ---")

    r = route_query("What is spacial computing and how does it affect you?", state_file=sf)
    dom = r["dominant_operations"]
    check("Q11: analytical question → DIAGNOSE dominant",
          dom[0] == "DIAGNOSE",
          f"dominant={dom[:3]}")

    # ---- 8. Constraints generated (Problem 5) ----
    log("\n--- Constraints generation ---")

    r = route_query("explain quantum computing", state_file=sf)
    check("Q12: has constraints",
          len(r["constraints"]) > 0,
          f"constraints={r['constraints']}")

    # ---- 9. Adjacency check passes sometimes (Problem 4) ----
    log("\n--- Adjacency check ---")

    r = route_query("build a REST API", state_file=sf)
    check("Q13: adjacency_ok can be true",
          r["adjacency_ok"] == True,
          f"adjacency_ok={r['adjacency_ok']}, adjacency check may need calibration")

    # ---- 10. ROUTE for multi-step ----
    log("\n--- Multi-step routing ---")

    r = route_query("first search for recent papers on plasma physics, then create a summary doc", state_file=sf)
    dom = r["dominant_operations"]
    check("Q14: multi-step → ROUTE dominant",
          dom[0] == "ROUTE",
          f"dominant={dom[:3]}")

    # ---- Additional checks ----
    log("\n--- Additional quality checks ---")

    # Check that tool reasons are NOT generic face descriptions
    r = route_query("search for information about fusion energy", state_file=sf)
    if r["tool_sequence"]:
        reason = r["tool_sequence"][0]["reason"]
        has_face_desc = "Read the situation" in reason or "Pull context" in reason
        check("Tool reasons are query-specific (not face descriptions)",
              not has_face_desc,
              f"reason='{reason}'")
    else:
        check("Tool reasons are query-specific", True)

    # Check max_tools discrimination
    r_simple = route_query("hey", state_file=sf)
    r_complex = route_query("first search for papers on CRISPR gene editing, then create a literature review file, then analyze the key findings", state_file=sf)
    check("Simple query gets fewer tools than complex",
          len(r_simple["tool_sequence"]) < len(r_complex["tool_sequence"]),
          f"simple={len(r_simple['tool_sequence'])}, complex={len(r_complex['tool_sequence'])}")

    log(f"\n=== Results: {passed}/{passed + failed} passed ===")
    return failed == 0


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
