"""
Unified Chestohedron Language Model — Saturn + KD INSIDE the Architecture
=========================================================================

This is NOT a wrapper. Saturn and KD are baked directly into ChestohedronBlock.
The block's forward pass has injection ports at MIRROR, INHERIT, and VERIFY.
When Saturn/KD are connected, their signals enter the geometry itself.
When offline, zero-vectors pass through — pure topology still works.

The model takes your input, cycles through the geometry with live perception
and live memory, and generates the response itself. No middleman.

Training data: V9-structured reasoning chains (topology determines data).
Architecture: V9 cycling block with external signal injection.
The data and architecture are CO-DESIGNED through the chestohedron.

Scaled up from chestohedron_lm.py:
  - hidden_dim 96 -> 256
  - 5 problem types -> 10 problem types
  - 3000 examples -> 10000 examples
  - Saturn 96-feature temporal perception -> MIRROR injection
  - KD spreading activation + domain knowledge -> INHERIT injection
  - KD contradiction_scan -> VERIFY injection
  - Saturn + KD density -> GATE 6 mode bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import random
import json

from chestohedron_lm import (
    PHI, STAGE_TOKENS, STAGE_END_TOKENS,
    V9Tokenizer, _spectral_scale, make_topology_matrices,
    prepare_batches, VanillaLSTM_LM, VanillaGRU_LM,
)
from ccn_integrated import (
    MCPClient, SaturnBridge, KDBridge,
    _extract_text, _parse_saturn_data, _clamp, _parse_nodes,
)


# ============================================================
# SCALED V9-STRUCTURED DATA GENERATION
# ============================================================

class ScaledV9DataGenerator:
    """
    V9-structured reasoning examples — scaled up.

    10 problem types that naturally decompose into V9 stages:
    1. Logic chains (transitive, conditional, elimination)
    2. Arithmetic (add, sub, mul, div, multi-step)
    3. Pattern completion (arithmetic seq, repeat, fibonacci-like, geometric)
    4. Analogy resolution (relationship mapping)
    5. Constraint satisfaction (find valid state under rules)
    6. Set operations (union, intersection, difference, membership)
    7. Comparison chains (ordering, ranking, relative reasoning)
    8. Causal reasoning (if-then chains, counterfactuals)
    9. Classification (property-based grouping, category assignment)
    10. Transformation (input->rule->output mapping)

    Each type exercises the V9 cycle differently.
    The topology dictates the categories — we don't choose them.
    """

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_logic_chain(self):
        entities = ["cat", "dog", "bird", "fish", "fox", "owl", "bear", "deer",
                     "wolf", "hawk", "snake", "frog", "lion", "tiger", "eagle"]
        properties = ["fast", "slow", "large", "small", "quiet", "loud",
                       "old", "young", "strong", "weak", "sharp", "dull"]
        actions = ["runs", "sleeps", "hunts", "hides", "swims", "flies",
                    "climbs", "digs", "watches", "waits", "stalks", "rests"]

        e1, e2, e3 = self.rng.sample(entities, 3)
        p1, p2 = self.rng.sample(properties, 2)
        a1, a2 = self.rng.sample(actions, 2)

        chain_type = self.rng.choice(["transitive", "conditional", "elimination",
                                       "double_transitive", "negation"])

        if chain_type == "transitive":
            mirror = f"what does the {e1} do"
            inherit = f"the {e1} is {p1} . if {p1} then {a1}"
            bound = f"only {p1} things {a1} . the {e1} is {p1}"
            express = f"the {e1} {a1}"
            verify = f"is the {e1} {p1} yes . do {p1} things {a1} yes"
            remove = f"the {e1} {a1}"
            gate6 = f"answer the {e1} {a1}"

        elif chain_type == "conditional":
            mirror = f"does the {e1} {a1}"
            inherit = f"the {e1} is {p1} . {p1} things {a1} . {p2} things {a2}"
            bound = f"the {e1} is {p1} not {p2}"
            express = f"the {e1} {a1} because {p1}"
            verify = f"could the {e1} {a2} no because not {p2}"
            remove = f"{e1} {a1}"
            gate6 = f"yes the {e1} {a1}"

        elif chain_type == "elimination":
            mirror = f"what does the {e1} not do"
            inherit = f"the {e1} is {p1} . {p1} things {a1} . {p2} things {a2} . the {e1} is not {p2}"
            bound = f"the {e1} cannot {a2} because not {p2}"
            express = f"the {e1} does not {a2}"
            verify = f"the {e1} is {p1} so {a1} yes . the {e1} is not {p2} so not {a2} correct"
            remove = f"not {a2}"
            gate6 = f"the {e1} does not {a2}"

        elif chain_type == "double_transitive":
            p3 = self.rng.choice([p for p in properties if p not in [p1, p2]])
            mirror = f"what does the {e1} do given it is {p1}"
            inherit = f"the {e1} is {p1} . if {p1} then {p2} . if {p2} then {a1}"
            bound = f"{p1} implies {p2} . {p2} implies {a1}"
            express = f"the {e1} is {p1} therefore {p2} therefore {a1}"
            verify = f"{e1} is {p1} yes . {p1} means {p2} yes . {p2} means {a1} yes . chain valid"
            remove = f"{e1} {a1} via {p1} to {p2}"
            gate6 = f"answer the {e1} {a1}"

        else:  # negation
            mirror = f"can the {e1} {a1}"
            inherit = f"only {p1} things {a1} . the {e1} is {p2} . {p2} is not {p1}"
            bound = f"need {p1} for {a1} . {e1} has {p2} not {p1}"
            express = f"the {e1} cannot {a1} because {p2} not {p1}"
            verify = f"is {p2} same as {p1} no . does {e1} have {p1} no . cannot {a1} correct"
            remove = f"no {a1}"
            gate6 = f"answer no the {e1} cannot {a1}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_arithmetic(self):
        op = self.rng.choice(["add", "sub", "mul", "div", "multi_step"])

        if op == "add":
            a = self.rng.randint(2, 50)
            b = self.rng.randint(2, 50)
            result = a + b
            mirror = f"find the sum of {a} and {b}"
            inherit = f"addition combines two quantities"
            bound = f"two numbers {a} and {b} operation is addition"
            express = f"{a} plus {b} equals {result}"
            verify = f"check {result} minus {b} equals {a} yes"
            remove = f"{a} plus {b} is {result}"
            gate6 = f"answer {result}"

        elif op == "sub":
            a = self.rng.randint(10, 100)
            b = self.rng.randint(2, a)
            result = a - b
            mirror = f"find the difference of {a} and {b}"
            inherit = f"subtraction finds the gap between quantities"
            bound = f"two numbers {a} and {b} operation is subtraction {a} is larger"
            express = f"{a} minus {b} equals {result}"
            verify = f"check {result} plus {b} equals {a} yes"
            remove = f"{a} minus {b} is {result}"
            gate6 = f"answer {result}"

        elif op == "mul":
            a = self.rng.randint(2, 15)
            b = self.rng.randint(2, 15)
            result = a * b
            mirror = f"find the product of {a} and {b}"
            inherit = f"multiplication is repeated addition"
            bound = f"two numbers {a} and {b} operation is multiplication"
            express = f"{a} times {b} equals {result}"
            verify = f"check {result} divided by {b} equals {a} yes"
            remove = f"{a} times {b} is {result}"
            gate6 = f"answer {result}"

        elif op == "div":
            b = self.rng.randint(2, 12)
            result = self.rng.randint(2, 15)
            a = b * result
            mirror = f"find {a} divided by {b}"
            inherit = f"division splits a quantity into equal parts"
            bound = f"dividend {a} divisor {b} operation is division"
            express = f"{a} divided by {b} equals {result}"
            verify = f"check {result} times {b} equals {a} yes"
            remove = f"{a} over {b} is {result}"
            gate6 = f"answer {result}"

        else:  # multi_step
            a = self.rng.randint(2, 20)
            b = self.rng.randint(2, 20)
            c = self.rng.randint(1, 10)
            step1 = a + b
            result = step1 * c
            mirror = f"find {a} plus {b} then multiply by {c}"
            inherit = f"order of operations first addition then multiplication"
            bound = f"step 1 add {a} and {b} step 2 multiply by {c}"
            express = f"{a} plus {b} is {step1} then {step1} times {c} is {result}"
            verify = f"check {step1} is {a} plus {b} yes . {result} is {step1} times {c} yes"
            remove = f"result is {result}"
            gate6 = f"answer {result}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_pattern(self):
        pattern_type = self.rng.choice(["arithmetic_seq", "repeat", "fibonacci_like",
                                         "geometric", "alternating"])

        if pattern_type == "arithmetic_seq":
            start = self.rng.randint(1, 20)
            step = self.rng.randint(1, 8)
            seq = [start + i * step for i in range(5)]
            answer = seq[-1] + step
            mirror = f"find the next number in {' '.join(map(str, seq))}"
            inherit = f"sequences follow rules the difference between terms can be constant"
            bound = f"the differences are {' '.join([str(step)] * 4)} all equal to {step}"
            express = f"next is {seq[-1]} plus {step} equals {answer}"
            verify = f"check {seq[-2]} plus {step} equals {seq[-1]} yes pattern holds"
            remove = f"add {step} to get {answer}"
            gate6 = f"answer {answer}"

        elif pattern_type == "repeat":
            cycle_len = self.rng.randint(2, 5)
            cycle = [self.rng.randint(1, 9) for _ in range(cycle_len)]
            seq = (cycle * 4)[:8]
            answer = cycle[len(seq) % cycle_len]
            mirror = f"find the next in {' '.join(map(str, seq))}"
            inherit = f"some sequences repeat a cycle"
            bound = f"the cycle is {' '.join(map(str, cycle))} length {cycle_len}"
            express = f"position {len(seq)} in cycle is index {len(seq) % cycle_len} which is {answer}"
            verify = f"check cycle {' '.join(map(str, cycle))} repeats yes matches sequence"
            remove = f"cycle repeats next is {answer}"
            gate6 = f"answer {answer}"

        elif pattern_type == "fibonacci_like":
            a, b = self.rng.randint(1, 8), self.rng.randint(1, 8)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
            answer = seq[-1] + seq[-2]
            mirror = f"find the next in {' '.join(map(str, seq))}"
            inherit = f"each term may be the sum of previous two"
            bound = f"check {seq[2]} equals {seq[0]} plus {seq[1]} yes fibonacci like"
            express = f"next is {seq[-1]} plus {seq[-2]} equals {answer}"
            verify = f"check {seq[-1]} equals {seq[-2]} plus {seq[-3]} which is {seq[-2] + seq[-3]} yes"
            remove = f"sum last two gives {answer}"
            gate6 = f"answer {answer}"

        elif pattern_type == "geometric":
            start = self.rng.randint(1, 5)
            ratio = self.rng.randint(2, 4)
            seq = [start * (ratio ** i) for i in range(5)]
            answer = seq[-1] * ratio
            mirror = f"find the next in {' '.join(map(str, seq))}"
            inherit = f"geometric sequences multiply by a constant ratio"
            bound = f"ratio is {seq[1]} over {seq[0]} equals {ratio}"
            express = f"next is {seq[-1]} times {ratio} equals {answer}"
            verify = f"check {seq[1]} over {seq[0]} is {ratio} and {seq[2]} over {seq[1]} is {ratio} yes"
            remove = f"multiply by {ratio} to get {answer}"
            gate6 = f"answer {answer}"

        else:  # alternating
            a_val = self.rng.randint(1, 10)
            b_val = self.rng.randint(1, 10)
            step_a = self.rng.randint(1, 3)
            seq = []
            for i in range(4):
                seq.append(a_val + i * step_a)
                seq.append(b_val)
            answer = a_val + 4 * step_a
            mirror = f"find the next in {' '.join(map(str, seq))}"
            inherit = f"alternating sequences interleave two patterns"
            bound = f"odd positions increase by {step_a} even positions stay {b_val}"
            express = f"next odd position is {seq[-2]} plus {step_a} equals {answer}"
            verify = f"check odd values {seq[0]} {seq[2]} {seq[4]} {seq[6]} increase by {step_a} yes"
            remove = f"next is {answer}"
            gate6 = f"answer {answer}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_analogy(self):
        analogies = [
            ("hot", "cold", "up", "down", "opposites"),
            ("big", "bigger", "small", "smaller", "comparative"),
            ("cat", "kitten", "dog", "puppy", "young form"),
            ("day", "night", "light", "dark", "opposites"),
            ("one", "first", "two", "second", "ordinal"),
            ("water", "ice", "rain", "snow", "frozen form"),
            ("hand", "glove", "foot", "shoe", "covering"),
            ("eye", "see", "ear", "hear", "function"),
            ("pen", "write", "knife", "cut", "function"),
            ("tree", "forest", "star", "galaxy", "collection"),
            ("book", "read", "song", "listen", "action"),
            ("teacher", "school", "doctor", "hospital", "workplace"),
            ("fire", "hot", "ice", "cold", "temperature"),
            ("bird", "nest", "bee", "hive", "home"),
            ("paint", "canvas", "ink", "paper", "surface"),
            ("sun", "day", "moon", "night", "time"),
            ("seed", "tree", "egg", "bird", "growth"),
            ("wheel", "car", "wing", "plane", "movement"),
        ]

        a, b, c, d, rel = self.rng.choice(analogies)
        mirror = f"what is to {c} as {b} is to {a}"
        inherit = f"{a} relates to {b} and {c} relates to something by the same rule"
        bound = f"the relationship between {a} and {b} is {rel}"
        express = f"{c} has the same {rel} relationship to {d}"
        verify = f"{a} to {b} is {rel} . {c} to {d} is {rel} . same pattern yes"
        remove = f"{c} to {d}"
        gate6 = f"answer {d}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_constraint(self):
        target = self.rng.randint(3, 30)
        lower = target - self.rng.randint(1, 4)
        upper = target + self.rng.randint(1, 4)
        parity = "even" if target % 2 == 0 else "odd"

        mirror = f"find a number that is {parity} and between {lower} and {upper}"
        inherit = f"{parity} numbers are divisible by 2" if parity == "even" else f"{parity} numbers are not divisible by 2"
        bound = f"must be {parity} . must be greater than {lower} . must be less than {upper}"

        valid = [n for n in range(lower + 1, upper) if (n % 2 == 0) == (parity == "even")]
        if not valid:
            valid = [target]

        answer = self.rng.choice(valid)
        express = f"{answer} is {parity} and between {lower} and {upper}"
        verify = f"is {answer} {parity} {'yes' if (answer % 2 == 0) == (parity == 'even') else 'no'} . is {answer} between {lower} and {upper} {'yes' if lower < answer < upper else 'no'}"
        remove = f"{answer}"
        gate6 = f"answer {answer}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_set_operation(self):
        """Set operations in V9 structure."""
        items_pool = ["red", "blue", "green", "yellow", "orange", "purple",
                       "black", "white", "pink", "brown", "gray", "silver"]
        n1 = self.rng.randint(3, 6)
        n2 = self.rng.randint(3, 6)
        set_a = self.rng.sample(items_pool, n1)
        set_b = self.rng.sample(items_pool, n2)

        op = self.rng.choice(["union", "intersection", "difference"])

        if op == "union":
            result = sorted(set(set_a) | set(set_b))
            mirror = f"find the union of {' '.join(set_a)} and {' '.join(set_b)}"
            inherit = f"union combines all elements from both sets removing duplicates"
            bound = f"set a is {' '.join(set_a)} . set b is {' '.join(set_b)}"
            express = f"union is {' '.join(result)}"
            shared = sorted(set(set_a) & set(set_b))
            verify = f"all of set a present yes . all of set b present yes . shared items {' '.join(shared) if shared else 'none'} counted once yes"
            remove = f"{' '.join(result)}"
            gate6 = f"answer {' '.join(result)}"

        elif op == "intersection":
            result = sorted(set(set_a) & set(set_b))
            mirror = f"find the intersection of {' '.join(set_a)} and {' '.join(set_b)}"
            inherit = f"intersection finds elements in both sets"
            bound = f"must be in set a and in set b"
            if result:
                express = f"intersection is {' '.join(result)}"
                verify = f"check each {' '.join(result)} in set a yes in set b yes"
            else:
                express = f"intersection is empty no shared elements"
                verify = f"no element in both sets confirmed"
            remove = f"{' '.join(result) if result else 'empty'}"
            gate6 = f"answer {' '.join(result) if result else 'empty set'}"

        else:  # difference
            result = sorted(set(set_a) - set(set_b))
            mirror = f"find elements in {' '.join(set_a)} but not in {' '.join(set_b)}"
            inherit = f"set difference removes elements found in the second set"
            bound = f"keep elements from set a that are not in set b"
            if result:
                express = f"difference is {' '.join(result)}"
                verify = f"check each {' '.join(result)} in set a yes not in set b yes"
            else:
                express = f"difference is empty all elements of a are in b"
                verify = f"every element of set a found in set b confirmed"
            remove = f"{' '.join(result) if result else 'empty'}"
            gate6 = f"answer {' '.join(result) if result else 'empty set'}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_comparison(self):
        """Comparison/ordering in V9 structure."""
        items = self.rng.sample(["ant", "cat", "dog", "horse", "whale", "mouse",
                                   "elephant", "rabbit", "bear", "tiger"], 4)
        sizes = sorted(range(4), key=lambda x: self.rng.random())

        size_order = [(items[i], sizes[i]) for i in range(4)]
        size_order.sort(key=lambda x: x[1])
        ordered = [x[0] for x in size_order]

        comp_type = self.rng.choice(["largest", "smallest", "between"])

        if comp_type == "largest":
            mirror = f"which is the largest among {' '.join(items)}"
            inherit = f"{ordered[0]} is smaller than {ordered[1]} . {ordered[1]} is smaller than {ordered[2]} . {ordered[2]} is smaller than {ordered[3]}"
            bound = f"compare all four find the maximum"
            express = f"{ordered[3]} is the largest"
            verify = f"is {ordered[3]} larger than {ordered[2]} yes . larger than {ordered[1]} yes . larger than {ordered[0]} yes"
            remove = f"{ordered[3]}"
            gate6 = f"answer {ordered[3]}"

        elif comp_type == "smallest":
            mirror = f"which is the smallest among {' '.join(items)}"
            inherit = f"{ordered[0]} is smaller than {ordered[1]} . {ordered[1]} is smaller than {ordered[2]} . {ordered[2]} is smaller than {ordered[3]}"
            bound = f"compare all four find the minimum"
            express = f"{ordered[0]} is the smallest"
            verify = f"is {ordered[0]} smaller than {ordered[1]} yes . smaller than {ordered[2]} yes . smaller than {ordered[3]} yes"
            remove = f"{ordered[0]}"
            gate6 = f"answer {ordered[0]}"

        else:  # between
            mirror = f"what is between {ordered[0]} and {ordered[3]} in size"
            inherit = f"order is {' then '.join(ordered)} from smallest to largest"
            bound = f"must be larger than {ordered[0]} and smaller than {ordered[3]}"
            express = f"{ordered[1]} and {ordered[2]} are between {ordered[0]} and {ordered[3]}"
            verify = f"is {ordered[1]} larger than {ordered[0]} yes . smaller than {ordered[3]} yes . correct"
            remove = f"{ordered[1]} and {ordered[2]}"
            gate6 = f"answer {ordered[1]} and {ordered[2]}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_causal(self):
        """Causal reasoning in V9 structure."""
        causes = [
            ("rain", "wet ground", "puddles form", "mud"),
            ("heat", "water boils", "steam rises", "condensation"),
            ("cold", "water freezes", "ice forms", "slippery surface"),
            ("wind", "trees sway", "leaves fall", "bare branches"),
            ("study", "learn facts", "pass test", "good grade"),
            ("exercise", "muscles grow", "strength increases", "endurance"),
            ("plant seed", "roots grow", "stem sprouts", "flower blooms"),
            ("mix colors", "new color forms", "shade changes", "gradient"),
        ]

        c1, c2, c3, c4 = self.rng.choice(causes)
        chain_len = self.rng.choice([2, 3, 4])

        if chain_len == 2:
            mirror = f"what happens when {c1}"
            inherit = f"when {c1} then {c2}"
            bound = f"{c1} causes {c2}"
            express = f"if {c1} then {c2}"
            verify = f"does {c1} cause {c2} yes . is this always true generally yes"
            remove = f"{c1} leads to {c2}"
            gate6 = f"answer {c2}"

        elif chain_len == 3:
            mirror = f"what is the result of {c1} after two steps"
            inherit = f"when {c1} then {c2} . when {c2} then {c3}"
            bound = f"chain is {c1} causes {c2} causes {c3}"
            express = f"{c1} leads to {c2} which leads to {c3}"
            verify = f"does {c1} cause {c2} yes . does {c2} cause {c3} yes . chain valid"
            remove = f"final result is {c3}"
            gate6 = f"answer {c3}"

        else:
            mirror = f"trace the full causal chain from {c1}"
            inherit = f"{c1} then {c2} then {c3} then {c4}"
            bound = f"four step chain {c1} to {c2} to {c3} to {c4}"
            express = f"{c1} causes {c2} causes {c3} causes {c4}"
            verify = f"each step follows yes . {c1} to {c2} yes . {c2} to {c3} yes . {c3} to {c4} yes"
            remove = f"end of chain is {c4}"
            gate6 = f"answer {c4}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_classification(self):
        """Classification/grouping in V9 structure."""
        categories = {
            "animal": ["cat", "dog", "fish", "bird", "snake", "frog", "bear"],
            "color": ["red", "blue", "green", "yellow", "orange", "purple"],
            "shape": ["circle", "square", "triangle", "star", "oval", "diamond"],
            "number": ["one", "two", "three", "four", "five", "six", "seven"],
            "fruit": ["apple", "banana", "cherry", "grape", "lemon", "peach"],
        }

        cat1, cat2 = self.rng.sample(list(categories.keys()), 2)
        items1 = self.rng.sample(categories[cat1], min(3, len(categories[cat1])))
        items2 = self.rng.sample(categories[cat2], min(2, len(categories[cat2])))
        item_to_classify = self.rng.choice(items1)
        mixed = items1 + items2
        self.rng.shuffle(mixed)

        mirror = f"what category does {item_to_classify} belong to from {' '.join(mixed)}"
        inherit = f"{cat1} includes {' '.join(categories[cat1][:4])} . {cat2} includes {' '.join(categories[cat2][:4])}"
        bound = f"classify {item_to_classify} as either {cat1} or {cat2}"
        express = f"{item_to_classify} is a {cat1}"
        verify = f"is {item_to_classify} in {cat1} list yes . is it in {cat2} list no . classification correct"
        remove = f"{item_to_classify} is {cat1}"
        gate6 = f"answer {cat1}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def generate_transformation(self):
        """Input->rule->output transformation in V9 structure."""
        transform_type = self.rng.choice(["double", "reverse_add", "shift", "conditional_transform"])

        if transform_type == "double":
            inp = self.rng.randint(2, 25)
            out = inp * 2
            mirror = f"apply the rule to {inp}"
            inherit = f"the rule is double the input"
            bound = f"input is {inp} rule is multiply by 2"
            express = f"{inp} times 2 equals {out}"
            verify = f"check {out} divided by 2 equals {inp} yes"
            remove = f"{out}"
            gate6 = f"answer {out}"

        elif transform_type == "reverse_add":
            digits = [self.rng.randint(1, 9) for _ in range(3)]
            num = digits[0] * 100 + digits[1] * 10 + digits[2]
            rev = digits[2] * 100 + digits[1] * 10 + digits[0]
            result = num + rev
            mirror = f"reverse {num} and add to original"
            inherit = f"reversing digits of a number and adding"
            bound = f"number is {num} reverse is {rev}"
            express = f"{num} plus {rev} equals {result}"
            verify = f"reverse of {num} is {rev} yes . {num} plus {rev} is {result} yes"
            remove = f"{result}"
            gate6 = f"answer {result}"

        elif transform_type == "shift":
            seq = [self.rng.randint(1, 9) for _ in range(5)]
            shifted = seq[1:] + [seq[0]]
            mirror = f"shift {' '.join(map(str, seq))} left by one"
            inherit = f"left shift moves each element one position left first goes to end"
            bound = f"sequence {' '.join(map(str, seq))} shift direction left amount 1"
            express = f"shifted is {' '.join(map(str, shifted))}"
            verify = f"first element {seq[0]} moved to end yes . others shifted left yes"
            remove = f"{' '.join(map(str, shifted))}"
            gate6 = f"answer {' '.join(map(str, shifted))}"

        else:  # conditional_transform
            inp = self.rng.randint(1, 30)
            if inp % 2 == 0:
                result = inp // 2
                rule_used = "halve"
            else:
                result = inp * 3 + 1
                rule_used = "triple plus one"
            mirror = f"apply collatz rule to {inp}"
            inherit = f"if even halve it . if odd triple it and add one"
            bound = f"input {inp} is {'even' if inp % 2 == 0 else 'odd'}"
            express = f"apply {rule_used} to {inp} gives {result}"
            verify = f"is {inp} {'even' if inp % 2 == 0 else 'odd'} yes . {rule_used} of {inp} is {result} yes"
            remove = f"{result}"
            gate6 = f"answer {result}"

        return self._format_example(mirror, inherit, bound, express, verify, remove, gate6)

    def _format_example(self, mirror, inherit, bound, express, verify, remove, gate6):
        stages = [mirror, inherit, bound, express, verify, remove, gate6]
        parts = []
        for text, start, end in zip(stages, STAGE_TOKENS, STAGE_END_TOKENS):
            parts.append(f"{start} {text} {end}")
        return " ".join(parts)

    def generate_dataset(self, n_examples=10000):
        generators = [
            self.generate_logic_chain,
            self.generate_arithmetic,
            self.generate_pattern,
            self.generate_analogy,
            self.generate_constraint,
            self.generate_set_operation,
            self.generate_comparison,
            self.generate_causal,
            self.generate_classification,
            self.generate_transformation,
        ]

        examples = []
        per_type = n_examples // len(generators)
        for gen in generators:
            for _ in range(per_type):
                examples.append(gen())

        while len(examples) < n_examples:
            gen = self.rng.choice(generators)
            examples.append(gen())

        self.rng.shuffle(examples)
        return examples


# ============================================================
# UNIFIED CHESTOHEDRON BLOCK — Saturn/KD INSIDE the geometry
# ============================================================

class UnifiedChestohedronBlock(nn.Module):
    """
    V9 cycling block with external signal injection ports.

    Saturn and KD don't wrap the block — they're INSIDE it.
    The block's forward pass has injection points at:
      - MIRROR: Saturn 96-feature temporal perception
      - INHERIT: KD spreading activation + domain knowledge
      - VERIFY: KD contradiction signal

    When signals are None (offline), zero injection — pure topology.
    When connected, external signals enter the geometry directly.
    The topology matrices are still FIXED. Only thin learned gates
    modulate the injection strength.
    """

    def __init__(self, dim, seed=42):
        super().__init__()
        self.dim = dim

        # FIXED topology matrices — the chestohedron IS the architecture
        matrices = make_topology_matrices(dim, seed)
        for name, W in matrices.items():
            self.register_buffer(f'W_{name}', W)

        # LEARNED: thin per-stage scale/shift
        self.stage_scale = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for _ in range(7)
        ])
        self.stage_shift = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(7)
        ])

        # LEARNED: input gate
        self.input_gate = nn.Linear(dim, dim, bias=False)

        # LEARNED: stage-context mixing
        self.stage_embed = nn.Embedding(7, dim)

        # LEARNED: injection gates — how external signals enter the geometry
        # These are the ONLY learned components that modulate external input.
        # Thin: dim -> dim, initialized small so topology dominates.
        self.saturn_gate = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.saturn_gate.weight, std=0.05 / math.sqrt(dim))

        self.kd_gate = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.kd_gate.weight, std=0.05 / math.sqrt(dim))

        self.verify_gate = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.verify_gate.weight, std=0.05 / math.sqrt(dim))

        # LEARNED: gate6 external bias projection
        self.gate6_bias_proj = nn.Linear(3, dim, bias=False)
        nn.init.normal_(self.gate6_bias_proj.weight, std=0.02)

    def _get_stage_matrices(self):
        return [
            self.W_mirror, self.W_inherit, self.W_bound,
            self.W_express, self.W_verify, self.W_remove,
        ]

    def _gate6_forward(self, state, u, gate6_bias=None):
        """
        GATE 6: Three-mode adaptive closer.
        Intrinsic energy gating + optional extrinsic bias from Saturn+KD.
        """
        energy = (state ** 2).sum(dim=-1, keepdim=True)
        energy_norm = energy / (energy.max() + 1e-8)

        g_tamam = torch.clamp(energy_norm - 0.66, min=0) * 3
        g_darash = torch.clamp(1 - (energy_norm - 0.5).abs() * 4, min=0)
        g_zakat = torch.clamp(0.33 - energy_norm, min=0) * 3

        # Extrinsic: Saturn temporal signals + KD knowledge density -> mode bias
        if gate6_bias is not None:
            # gate6_bias: (batch, 3) — [tamam, darash, zakat] weights
            bias_signal = self.gate6_bias_proj(gate6_bias)  # (batch, dim)
            # Modulate gate weights with external bias
            g_tamam = g_tamam + gate6_bias[:, 0:1] * 0.3
            g_darash = g_darash + gate6_bias[:, 1:2] * 0.3
            g_zakat = g_zakat + gate6_bias[:, 2:3] * 0.3

        g_total = g_tamam + g_darash + g_zakat + 1e-8

        out = ((g_tamam / g_total) * (state @ self.W_gate6_tamam) +
               (g_darash / g_total) * (state @ self.W_gate6_darash) +
               (g_zakat / g_total) * (state @ self.W_gate6_zakat))

        h = out * self.stage_scale[6] + self.stage_shift[6] + u
        if gate6_bias is not None:
            h = h + bias_signal * 0.1
        return torch.tanh(h)

    def forward(self, state, x, stage_idx=None,
                saturn_signal=None, kd_signal=None, verify_signal=None,
                gate6_bias=None):
        """
        Process one timestep through the V9 cycle.

        External signals enter the geometry at specific stages:
          saturn_signal:  (batch, dim) -> injected at MIRROR stage
          kd_signal:      (batch, dim) -> injected at INHERIT stage
          verify_signal:  (batch, dim) -> injected at VERIFY stage
          gate6_bias:     (batch, 3)   -> [tamam, darash, zakat] weights for GATE 6

        When any signal is None, zero injection at that stage.
        The topology matrices are FIXED. The geometry does the work.
        """
        u = self.input_gate(x)

        if stage_idx is not None:
            stage_ctx = self.stage_embed(stage_idx)
            u = u + stage_ctx * 0.3

        matrices = self._get_stage_matrices()

        for i, W in enumerate(matrices):
            h = state @ W + u
            h = h * self.stage_scale[i] + self.stage_shift[i]

            # ── INJECTION POINTS ──
            if i == 0 and saturn_signal is not None:
                # MIRROR: Saturn's temporal perception enters here
                # The symmetric reflection matrix + live keystroke data
                h = h + self.saturn_gate(saturn_signal) * 0.3

            if i == 1 and kd_signal is not None:
                # INHERIT: KD's accumulated knowledge enters here
                # Spreading activation + domain knowledge weighted by confidence
                h = h + self.kd_gate(kd_signal) * 0.3

            if i == 4 and verify_signal is not None:
                # VERIFY: KD's contradiction signal amplifies adversarial check
                # If existing beliefs conflict, VERIFY hits harder
                h = h + self.verify_gate(verify_signal) * 0.3

            state = torch.tanh(h)

        # GATE 6: three-mode adaptive closer with optional external bias
        state = self._gate6_forward(state, u, gate6_bias)
        return state


# ============================================================
# UNIFIED CHESTOHEDRON LANGUAGE MODEL
# ============================================================

class UnifiedChestohedronLM(nn.Module):
    """
    Language model where Saturn and KD are INSIDE the architecture.

    Not three systems duct-taped together. One system.
    The model takes input, cycles through the geometry with live
    perception and live memory, and generates the response itself.

    Forward pass:
    1. Embed tokens
    2. At each timestep, query Saturn + KD for live signals
    3. Feed signals directly into UnifiedChestohedronBlock
    4. Block processes through 7 stages with injections
    5. Output logits

    During training: Saturn/KD signals are simulated from the V9
    stage structure (the data itself encodes what each stage needs).
    During inference: real Saturn/KD MCP connections feed live data.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_cycles=1,
                 stage_token_ids=None, seed=42,
                 saturn_url=None, kd_url=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles

        # Stage token mapping
        self.stage_token_map = {}
        if stage_token_ids:
            for i, tid in enumerate(stage_token_ids):
                self.stage_token_map[tid] = i

        # Learned: embedding + projections
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj_in = nn.Linear(embed_dim, hidden_dim)

        # Unified chestohedron blocks (Saturn/KD injection baked in)
        self.blocks = nn.ModuleList([
            UnifiedChestohedronBlock(hidden_dim, seed=seed + i)
            for i in range(n_cycles)
        ])

        # Learned: output
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, vocab_size)

        # External system connections (optional)
        self._saturn_bridge = None
        self._kd_bridge = None
        self._online = {"saturn": False, "kd": False}

        if saturn_url:
            try:
                saturn_mcp = MCPClient(saturn_url)
                if saturn_mcp.connected:
                    self._saturn_bridge = SaturnBridge(saturn_mcp, hidden_dim)
                    self._online["saturn"] = True
            except Exception:
                pass

        if kd_url:
            try:
                kd_mcp = MCPClient(kd_url)
                if kd_mcp.connected:
                    self._kd_bridge = KDBridge(kd_mcp, hidden_dim)
                    self._online["kd"] = True
            except Exception:
                pass

    def _get_live_signals(self, input_text=None):
        """
        Query Saturn and KD for live signals.
        Returns (saturn_signal, kd_signal, verify_signal, gate6_bias) tensors,
        or None for each if offline.
        """
        saturn_sig = None
        kd_sig = None
        verify_sig = None
        gate6_bias = None

        if self._saturn_bridge:
            state, patterns = self._saturn_bridge.query()
            saturn_sig = self._saturn_bridge.mirror_signal(state, patterns)
            t, d, z = self._saturn_bridge.gate6_bias(state, patterns)
            gate6_bias = torch.tensor([t, d, z], dtype=torch.float32)

        if self._kd_bridge and input_text:
            graph = self._kd_bridge.graph_query(input_text)
            browse = self._kd_bridge.vault_browse("neural-architecture")
            kd_sig = self._kd_bridge.inherit_signal(graph, browse)

            contradictions = self._kd_bridge.contradiction_scan(input_text)
            verify_sig = self._kd_bridge.verify_signal(contradictions)

            # Adjust gate6 bias with KD density
            density = self._kd_bridge.knowledge_density(graph, browse)
            if gate6_bias is None:
                gate6_bias = torch.tensor([0.34, 0.33, 0.33])
            if density < 0.3:
                gate6_bias[1] += 0.15
                gate6_bias[0] -= 0.075
                gate6_bias[2] -= 0.075
            elif density > 0.7:
                gate6_bias[0] += 0.15
                gate6_bias[1] -= 0.075
                gate6_bias[2] -= 0.075
            gate6_bias = gate6_bias / gate6_bias.sum()

        return saturn_sig, kd_sig, verify_sig, gate6_bias

    def _simulate_stage_signals(self, stage_indices, batch_size):
        """
        During training, simulate external signals from V9 stage structure.

        The data itself tells us what stage we're in:
        - MIRROR stage tokens -> simulated perception signal
        - INHERIT stage tokens -> simulated knowledge signal
        - VERIFY stage tokens -> simulated contradiction signal

        This trains the injection gates so they're ready for real signals.
        """
        # Stage-dependent signal simulation using learned embeddings
        # The block's stage_embed already encodes stage context,
        # but we also create synthetic injection signals so the
        # injection gates learn meaningful weights during training.
        return None, None, None, None  # During training, block handles stages internally

    def forward(self, x, hidden=None, stage_indices=None,
                saturn_signal=None, kd_signal=None, verify_signal=None,
                gate6_bias=None):
        """
        x: (batch, seq_len)
        stage_indices: (batch, seq_len)
        saturn_signal: (batch, hidden_dim) or None
        kd_signal: (batch, hidden_dim) or None
        verify_signal: (batch, hidden_dim) or None
        gate6_bias: (batch, 3) or None
        """
        batch_size, seq_len = x.shape

        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                      for _ in range(self.n_cycles)]

        emb = self.embed(x)
        inp = self.proj_in(emb)

        outputs = []
        for t in range(seq_len):
            h = inp[:, t, :]
            si = stage_indices[:, t] if stage_indices is not None else None

            # Determine per-timestep injection signals
            # During inference with live connections, these come from Saturn/KD
            # During training, the stage-aware processing handles it
            ts_saturn = saturn_signal
            ts_kd = kd_signal
            ts_verify = verify_signal
            ts_gate6 = gate6_bias

            # Stage-aware signal routing during training:
            # Only inject at the appropriate stage
            if si is not None and saturn_signal is None and kd_signal is None:
                # Training mode: use stage indices to create lightweight
                # synthetic signals that train the injection gates
                # MIRROR (stage 0): create perception-like signal from input
                mirror_mask = (si == 0).float().unsqueeze(-1)
                if mirror_mask.sum() > 0:
                    ts_saturn = h * mirror_mask * 0.1

                # INHERIT (stage 1): create knowledge-like signal from input
                inherit_mask = (si == 1).float().unsqueeze(-1)
                if inherit_mask.sum() > 0:
                    ts_kd = h * inherit_mask * 0.1

                # VERIFY (stage 4): create contradiction-like signal
                verify_mask = (si == 4).float().unsqueeze(-1)
                if verify_mask.sum() > 0:
                    ts_verify = h * verify_mask * 0.1

            for i, block in enumerate(self.blocks):
                hidden[i] = block(hidden[i], h, si,
                                  saturn_signal=ts_saturn,
                                  kd_signal=ts_kd,
                                  verify_signal=ts_verify,
                                  gate6_bias=ts_gate6)
                h = hidden[i]

            outputs.append(self.ln(h))

        out = torch.stack(outputs, dim=1)
        logits = self.proj_out(out)
        return logits, hidden

    def post_cycle(self, input_text, output_text, mode="tamam", trace=None):
        """Post-cycle: deposit output back into KD."""
        if self._kd_bridge:
            self._kd_bridge.metabolize(output_text, mode, trace or [])
            if mode == "zakat":
                self._kd_bridge.run_decay()

    @property
    def learned_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def fixed_params(self):
        return sum(b.numel() for b in self.buffers())

    @property
    def status(self):
        return {k: ("online" if v else "offline") for k, v in self._online.items()}


# ============================================================
# TRAINING WITH V9-STRUCTURED DATA
# ============================================================

def train_model(model, batches, epochs, lr=0.003, clip=1.0, label="Model",
                use_stages=True):
    """Train with V9 stage awareness."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_b = 0
        hidden = None

        for x, y, si in batches:
            optimizer.zero_grad()

            if hidden is not None:
                if isinstance(hidden, tuple):
                    hidden = tuple(h.detach() for h in hidden)
                elif isinstance(hidden, list):
                    hidden = [h.detach() for h in hidden]
                else:
                    hidden = hidden.detach()

            stages = si if use_stages else None
            logits, hidden = model(x, hidden, stages)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = total_loss / max(n_b, 1)
        ppl = math.exp(min(avg_loss, 20))
        history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  [{label:>6s}] Epoch {epoch+1:3d}/{epochs}  "
                  f"Loss: {avg_loss:.4f}  PPL: {ppl:.1f}  "
                  f"Time: {elapsed:.1f}s")

    return history


# ============================================================
# INFERENCE — V9-STRUCTURED GENERATION
# ============================================================

def generate_v9_response(model, tokenizer, problem, hidden_dim,
                          input_text_for_signals=None):
    """
    Generate a V9-structured response.
    If the model has live Saturn/KD connections, they feed the generation.
    """
    model.eval()

    # Get live signals if available
    saturn_sig, kd_sig, verify_sig, gate6_bias = None, None, None, None
    if hasattr(model, '_get_live_signals') and input_text_for_signals:
        saturn_sig, kd_sig, verify_sig, gate6_bias = model._get_live_signals(
            input_text_for_signals)
        # Expand to batch dim
        if saturn_sig is not None:
            saturn_sig = saturn_sig.unsqueeze(0)
        if kd_sig is not None:
            kd_sig = kd_sig.unsqueeze(0)
        if verify_sig is not None:
            verify_sig = verify_sig.unsqueeze(0)
        if gate6_bias is not None:
            gate6_bias = gate6_bias.unsqueeze(0)

    with torch.no_grad():
        prompt = f"<MIRROR> {problem}"
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long)
        si = torch.zeros_like(x)

        hidden = None
        logits, hidden = model(x, hidden, si,
                                saturn_signal=saturn_sig,
                                kd_signal=kd_sig,
                                verify_signal=verify_sig,
                                gate6_bias=gate6_bias)

        generated = list(tokens)
        current_stage = 0
        stage_start_ids, _ = tokenizer.get_stage_token_ids()
        start_to_stage = {sid: i for i, sid in enumerate(stage_start_ids)}

        for _ in range(300):
            last_logits = logits[0, -1, :] / 0.7
            probs = F.softmax(last_logits, dim=0)
            idx = torch.multinomial(probs, 1).item()
            generated.append(idx)

            if idx in start_to_stage:
                current_stage = start_to_stage[idx]

            x = torch.tensor([[idx]], dtype=torch.long)
            si = torch.tensor([[current_stage]], dtype=torch.long)
            logits, hidden = model(x, hidden, si,
                                    saturn_signal=saturn_sig,
                                    kd_signal=kd_sig,
                                    verify_signal=verify_sig,
                                    gate6_bias=gate6_bias)

        return tokenizer.decode(generated)


# ============================================================
# MAIN — SCALED BENCHMARK
# ============================================================

def run_benchmark(n_examples=10000, embed_dim=64, hidden_dim=256,
                  seq_len=128, batch_size=32, epochs=30,
                  saturn_url=None, kd_url=None):
    """
    Unified V9-structured language model benchmark.

    Scaled up:
      - hidden_dim: 256 (was 96)
      - embed_dim: 64 (was 32)
      - examples: 10000 (was 3000)
      - problem types: 10 (was 5)
      - seq_len: 128 (was 96)
      - batch_size: 32 (was 16)
      - Saturn/KD injection ports in the architecture

    The data is V9-structured. The architecture matches the data.
    External systems feed specific stages. The geometry does the work.
    """
    print("=" * 70)
    print("  UNIFIED CHESTOHEDRON LANGUAGE MODEL")
    print("  Saturn + KD INSIDE the architecture")
    print("  V9-structured data x V9-structured architecture")
    print("=" * 70)

    # Generate SCALED V9-structured data
    print(f"\n  Generating {n_examples} V9-structured reasoning examples...")
    print(f"  10 problem types: logic, arithmetic, pattern, analogy,")
    print(f"  constraint, set ops, comparison, causal, classification, transform")
    gen = ScaledV9DataGenerator(seed=42)
    examples = gen.generate_dataset(n_examples)

    print(f"\n  Sample example:")
    sample = examples[0]
    for tok in STAGE_TOKENS:
        sample = sample.replace(tok, f"\n    {tok}")
    print(f"    {sample[:500]}")

    # Tokenize
    print(f"\n  Building tokenizer...")
    tokenizer = V9Tokenizer().build(examples)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Split
    split = int(len(examples) * 0.9)
    train_examples = examples[:split]
    test_examples = examples[split:]

    train_batches = prepare_batches(train_examples, tokenizer, seq_len, batch_size)
    test_batches = prepare_batches(test_examples, tokenizer, seq_len, batch_size)
    print(f"  Train batches: {len(train_batches)}  Test batches: {len(test_batches)}")

    # Stage token IDs
    stage_start_ids, _ = tokenizer.get_stage_token_ids()

    # Build models
    print(f"\n{'─' * 70}")
    print(f"  Building models (hidden_dim={hidden_dim}, embed_dim={embed_dim})...")

    ccn = UnifiedChestohedronLM(
        tokenizer.vocab_size, embed_dim, hidden_dim,
        n_cycles=1, stage_token_ids=stage_start_ids, seed=42,
        saturn_url=saturn_url, kd_url=kd_url,
    )
    lstm = VanillaLSTM_LM(tokenizer.vocab_size, embed_dim, hidden_dim)
    gru = VanillaGRU_LM(tokenizer.vocab_size, embed_dim, hidden_dim)

    print(f"\n  System status: {ccn.status}")

    print(f"\n  {'Model':<35s} {'Learned':>10s} {'Fixed':>10s} {'Total':>10s}")
    print(f"  {'─' * 65}")
    for name, m in [("CCN unified (saturn+kd inside)", ccn),
                     ("LSTM (vanilla)", lstm),
                     ("GRU (vanilla)", gru)]:
        lp = m.learned_params
        fp = m.fixed_params
        print(f"  {name:<35s} {lp:>10,d} {fp:>10,d} {lp+fp:>10,d}")

    # Train
    print(f"\n{'─' * 70}")
    print(f"  Training ({epochs} epochs, CPU)...")
    print(f"{'─' * 70}")

    print(f"\n  --- Unified CCN (V9-aware, Saturn/KD injection) ---")
    h_ccn = train_model(ccn, train_batches, epochs, lr=0.002,
                         label="CCN", use_stages=True)

    print(f"\n  --- LSTM (flat, no stage awareness) ---")
    h_lstm = train_model(lstm, train_batches, epochs, lr=0.002,
                          label="LSTM", use_stages=False)

    print(f"\n  --- GRU (flat, no stage awareness) ---")
    h_gru = train_model(gru, train_batches, epochs, lr=0.002,
                         label="GRU", use_stages=False)

    # Evaluate
    print(f"\n{'─' * 70}")
    print(f"  Evaluating on test set...")

    def evaluate(model, batches, use_stages=False):
        model.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for x, y, si in batches:
                stages = si if use_stages else None
                logits, _ = model(x, None, stages)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    test_ccn = evaluate(ccn, test_batches, use_stages=True)
    test_lstm = evaluate(lstm, test_batches)
    test_gru = evaluate(gru, test_batches)

    # Results
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — UNIFIED V9 LANGUAGE BENCHMARK (SCALED)")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<35s} {'Train':>8s} {'Test':>8s} {'TestPPL':>8s} {'Learned':>10s}")
    print(f"  {'─' * 69}")
    for name, tr, te, m in [
        ("CCN unified (saturn+kd)", h_ccn[-1], test_ccn, ccn),
        ("LSTM (all learned)", h_lstm[-1], test_lstm, lstm),
        ("GRU (all learned)", h_gru[-1], test_gru, gru),
    ]:
        ppl = math.exp(min(te, 20))
        print(f"  {name:<35s} {tr:>8.4f} {te:>8.4f} {ppl:>8.1f} {m.learned_params:>10,d}")

    # Capability density
    print(f"\n  Capability Density (quality per learned parameter):")
    cd_ccn = (1.0 / test_ccn) / ccn.learned_params * 1e6
    cd_lstm = (1.0 / test_lstm) / lstm.learned_params * 1e6
    cd_gru = (1.0 / test_gru) / gru.learned_params * 1e6

    print(f"  CCN:  {cd_ccn:.4f}")
    print(f"  LSTM: {cd_lstm:.4f}")
    print(f"  GRU:  {cd_gru:.4f}")
    if cd_ccn > cd_lstm:
        print(f"  -> CCN has {cd_ccn/cd_lstm:.1f}x higher capability density than LSTM")
    if cd_ccn > cd_gru:
        print(f"  -> CCN has {cd_ccn/cd_gru:.1f}x higher capability density than GRU")

    # Generation samples
    print(f"\n{'=' * 70}")
    print(f"  V9-STRUCTURED GENERATION (UNIFIED MODEL)")
    print(f"{'=' * 70}")

    test_problems = [
        "what does the fox do",
        "find the sum of 13 and 29",
        "find the next number in 3 6 12 24 48",
        "what is to foot as glove is to hand",
        "find elements in red blue green but not in blue yellow",
    ]

    for problem in test_problems:
        print(f"\n  Problem: {problem}")
        output = generate_v9_response(ccn, tokenizer, problem, hidden_dim,
                                       input_text_for_signals=problem)
        for tok in STAGE_TOKENS:
            output = output.replace(tok, f"\n    {tok}")
        for tok in STAGE_END_TOKENS:
            output = output.replace(tok, f" {tok}")
        print(f"  Response:{output[:400]}")

    # Architecture summary
    print(f"\n{'=' * 70}")
    print(f"  UNIFIED ARCHITECTURE")
    print(f"{'=' * 70}")
    print(f"""
  Saturn/KD are INSIDE the ChestohedronBlock, not wrapping it.

  UnifiedChestohedronBlock.forward(state, x, stage_idx,
      saturn_signal, kd_signal, verify_signal, gate6_bias):

  ┌─ MIRROR  ◄── saturn_signal (96 temporal keystroke features -> dim)
  │    symmetric matrix + Saturn injection gate (learned, thin)
  │
  ├─ INHERIT ◄── kd_signal (graph_query + vault_browse -> dim)
  │    near-orthogonal matrix + KD injection gate (learned, thin)
  │
  ├─ BOUND
  │    low-rank projection (pure topology)
  │
  ├─ EXPRESS
  │    phi-rotated orthogonal (ONLY generative stage, pure topology)
  │
  ├─ VERIFY  ◄── verify_signal (contradiction_scan -> dim)
  │    anti-symmetric matrix + verify injection gate (learned, thin)
  │
  ├─ REMOVE
  │    low-rank projection (double-compression bottleneck)
  │
  └─ GATE 6  ◄── gate6_bias [tamam, darash, zakat] from Saturn+KD
       3-mode adaptive closer + external bias projection

  Fixed topology: {ccn.fixed_params:,d} params
  Learned params: {ccn.learned_params:,d} (embeddings + calibration + injection gates)
  LSTM learned:   {lstm.learned_params:,d} (everything)
  GRU learned:    {gru.learned_params:,d} (everything)""")

    if ccn.fixed_params > 0:
        print(f"  Topology/learned ratio: {ccn.fixed_params/ccn.learned_params:.2f}x")

    print(f"""
  Training data: V9-structured reasoning chains ({n_examples} examples)
  10 problem types, topology determines the data structure.
  NOT Shakespeare. NOT flat text. V9 stages structure every example.

  Online mode: Saturn -> MIRROR, KD -> INHERIT + VERIFY, both -> GATE 6
  Offline mode: zero injection, pure topology still works.
  One system. No middleman.
""")

    return {
        "ccn": {"train": h_ccn[-1], "test": test_ccn,
                "learned": ccn.learned_params, "fixed": ccn.fixed_params},
        "lstm": {"train": h_lstm[-1], "test": test_lstm,
                 "learned": lstm.learned_params},
        "gru": {"train": h_gru[-1], "test": test_gru,
                "learned": gru.learned_params},
    }


if __name__ == "__main__":
    run_benchmark()
