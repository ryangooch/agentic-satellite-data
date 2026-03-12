# Take-Home Exercises

## Prerequisites

Clone the repo, run `make setup`, then `make demo` to verify everything works.

## Exercise 1: Add a New Tool

**Goal:** Understand tool definition and integration.

Implement `compute_savi(region: BoundingBox, L: float = 0.5) -> SAVIResult`:

```
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
```

1. Add `SAVIResult` dataclass to `agent/types.py`
2. Implement `compute_savi` in `agent/tools.py`
3. Write the JSON schema in `agent/schemas.py`
4. Add it to `_TOOL_MAP` in `agent/loop.py`
5. Write a test in `tests/test_tools.py`
6. Run the agent and check if it uses SAVI when appropriate

**Reflection:** Did the agent know when to use SAVI vs. NDVI? What in the schema description would make it more likely to choose correctly?

---

## Exercise 2: Modify the System Prompt

**Goal:** Understand how instructions shape agent behavior.

Create 3 variants of `SYSTEM_PROMPT` in `agent/loop.py`:

1. **Minimal** — Just the role, no workflow instructions
2. **Directive** — Add "Always call NDVI before any other index"
3. **Original** — The current prompt

Run Scene A with each. Compare:
- Tool call sequences (order and which tools are called)
- Quality of reasoning text
- Handling of the water stress ambiguity

**Reflection:** Which prompt is most useful? Which produces the most instructive trace?

---

## Exercise 3: Error-Handling Tool

**Goal:** Understand tool error handling and agent adaptation.

Modify `compare_to_baseline` to fail 50% of the time (use `random.random() < 0.5`).

Then add a new tool:
```python
check_baseline_exists(region: BoundingBox) -> BaselineCheckResult
# Returns whether a baseline exists for the current scene
```

Modify the system prompt to suggest checking for baselines before comparing.

**Reflection:** What are the tradeoffs between preventing errors (defensive tools) vs. recovering from errors (error results)?

---

## Exercise 4: Multi-Scene Comparison

**Goal:** Extend the agent to handle more complex workflows.

New user request: *"Compare crop health in Scene A vs. Scene B and identify which region has more severe stress."*

Options:
- Add a `switch_scene(scene_id: str)` tool
- Or: modify all tools to accept an optional `scene_id` parameter

**Reflection:** Does the agent develop a coherent comparison strategy? What breaks? What would you change?

---

## Exercise 5: Simple Evaluation

**Goal:** Understand evaluation challenges for agentic systems.

Ground truth: Scene A has water stress in the NW quadrant. Scene B has no localized anomaly.

Build an evaluator that checks:
1. Did the agent identify the correct issue (or correctly report no issue)?
2. Did it call reasonable tools in a sensible order?
3. Did it express appropriate confidence?

Run the agent 5 times on each scene (`temperature=0.5` for variety). Measure consistency.

**Reflection:** Is it harder to evaluate correctness of the final report, or quality of the reasoning process? Why does this distinction matter for production systems?
