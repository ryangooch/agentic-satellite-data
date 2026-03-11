# Agentic AI for Signal Processing: Lecture Plan
## Satellite Crop Health Assessment via Autonomous Multi-Band Image Analysis

---

## Context and Goals

This is a 75-minute lecture for senior undergraduates in electrical and computer engineering, plus a handful of first-year graduate students. The audience has solid signal processing and some deep learning background but is new to agentic AI.

The central pedagogical goal is to contrast a **fixed signal processing pipeline** (deterministic, pre-specified order of operations) with an **agentic pipeline** (a model that reasons about which tools to call, in what order, based on what it observes). The satellite imagery domain is chosen because it is visually engaging, uses familiar signal processing concepts (spectral indices, band math, anomaly detection), and naturally motivates sequential decision-making under uncertainty.

Students should leave understanding:
- What makes a system "agentic" vs. a standard ML pipeline
- How tool calling works at the API level (no framework magic)
- Why the reasoning loop matters — and how it can fail
- Where production abstractions like MCP fit in, and why we're not using them here

---

## API Key Strategy and Budget

**Decision required before lecture**: How will API calls be handled?

### Option 1: Student Keys (Recommended)
- Students bring their own Anthropic or OpenAI API keys
- Pros: No cost to instructor, students learn API setup
- Cons: Requires setup instructions, some students may lack credit cards
- **Estimated cost per student**: $0.50–2.00 for all exercises (Scene A + B + experiments)

### Option 2: Shared Institutional Key
- Provide a single API key with rate limits
- Pros: No student setup, guaranteed to work
- Cons: Budget required, need rate limiting to prevent abuse
- **Estimated cost**: $50–150 for a class of 30 students (depending on experimentation)

### Option 3: Mock Mode Only
- Use pre-recorded API responses, no live calls during lecture
- Pros: Zero cost, works offline, perfectly reproducible
- Cons: Less impressive, students don't see real API interaction
- **Use case**: Budget-constrained environments or as fallback

**Recommendation**: Use Option 1 (student keys) for exercises, Option 3 (mock mode) for lecture demos. This balances cost, reliability, and learning value.

---

## Repository Structure

```
agentic-satellite/
├── README.md                  # Setup instructions, links, overview
├── Makefile                   # Convenience targets: setup, download-data, test, demo
├── requirements.txt           # Python dependencies with pinned versions
├── EXERCISES.md               # Take-home student exercises
├── data/
│   ├── scenes/                # Pre-processed Sentinel-2 scenes (Scene A, Scene B)
│   │   ├── scene_a_metadata.json
│   │   ├── scene_a_bands.npy  # or GeoTIFF
│   │   ├── scene_b_metadata.json
│   │   └── scene_b_bands.npy
│   └── download_data.py       # Optional: Sentinel-2 download script
├── slides/
│   └── background.pdf         # NDVI, NDWI, EVI, IR bands, crop stress, ET
├── notebooks/
│   ├── 00_data_exploration.ipynb   # Band visualization, orientation
│   ├── 01_tools.ipynb              # Tools as plain Python, testable in isolation
│   ├── 02_agent_loop.ipynb         # The main teaching notebook
│   └── 03_failure_modes.ipynb      # Where and why agents break (take-home)
└── agent/
    ├── tools.py               # Tool implementations
    ├── schemas.py             # JSON schemas for each tool
    ├── loop.py                # The raw agent loop (~50-100 lines)
    └── mock.py                # Mock mode responses for offline demos
```

**Design goals**:
- Students can clone and run immediately (pre-processed scenes included)
- Total size < 200MB
- Core code < 500 lines
- All notebooks are self-contained (can be run in any order after 00)

---

## Background Slides Content (brief)

These slides are pre-reading / first 10 minutes of lecture. Keep them signal-processing focused, not AI focused.

- **Sentinel-2 band overview** — what each band measures, spatial resolution, revisit time
- **NDVI** — formula, interpretation, saturation in dense canopy
- **NDWI / NDMI** — moisture stress in vegetation vs. open water detection
- **EVI** — why it exists, when to prefer it over NDVI
- **Thermal/IR bands** — evapotranspiration, why stressed crops are warmer
- **Crop stress signatures** — what water stress, pest damage, and nutrient deficiency look like spectrally, and crucially, how they can be confused with each other
- **The ambiguity problem** — motivates why a fixed pipeline is brittle and an adaptive one is useful

---

## Data

Use **Sentinel-2 Level-2A** imagery (surface reflectance, atmospherically corrected). Free via:
- Copernicus Data Space (preferred, no account required for small downloads)
- Google Earth Engine (good for students with GEE access)
- AWS Open Data (Sentinel-2 COGs)

**Critical: Provide pre-processed scenes in the repository.** Data acquisition should not block the learning objectives. The repository should include ready-to-use scenes so students/instructors can run notebooks immediately.

Provide a `Makefile` target (e.g., `make download-data`) or `download_data.sh` script for those who want to understand the acquisition process, but treat this as optional/extra credit, not a prerequisite.

**Scene selection is important.** You need:
1. **Scene A** — a "scripted" scene with a clear but non-obvious anomaly (e.g., one field quadrant showing NDVI stress that turns out to be water stress, confirmed by NDWI and a timeseries showing recent onset). Used for the traced walkthrough. **Test with temperature=0 to ensure consistent tool call sequences across runs.**
2. **Scene B** — an "ambiguous" scene used for the live demo. Ideally the agent takes a different tool path than it did for Scene A.

Suggested region: Central Valley California, or any irrigated agricultural region with good Sentinel-2 coverage. Crop type maps are useful context but not required for the core demo.

Include at minimum: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B8A (Red-edge), B11 (SWIR), B12 (SWIR2). Optionally B05/B06/B07 for red-edge indices.

**Data format**: Store as GeoTIFF or NumPy arrays (`.npy`) with accompanying metadata JSON. Keep file sizes reasonable (< 50MB per scene if possible) by cropping to relevant agricultural areas.

---

## Tools

Each tool is a plain Python function with a clear docstring and a corresponding JSON schema for the API. No framework. The schema dict lives in `schemas.py` and is passed directly to the `tools` parameter of the API call.

### Tool List

```python
compute_ndvi(region: BoundingBox) -> NDVIResult
# Returns NDVI map array, mean, std, low-NDVI pixel fraction, and a rendered image path

compute_ndwi(region: BoundingBox) -> NDWIResult
# Normalized Difference Water Index — moisture in vegetation canopy

compute_evi(region: BoundingBox) -> EVIResult
# Enhanced Vegetation Index — less saturation than NDVI in dense canopy

get_pixel_timeseries(lat: float, lon: float, index: str) -> TimeseriesResult
# Returns a time series of the specified index at a point across available images
# Useful for distinguishing new anomalies from persistent conditions

flag_anomalous_regions(index: str, threshold: float, direction: str) -> AnomalyResult
# Returns bounding boxes and pixel counts of regions above/below threshold
# direction: "below" or "above"

compare_to_baseline(region: BoundingBox, index: str) -> DiffResult
# Diffs current image against a stored baseline (earlier date)
# Returns change map and summary statistics

lookup_crop_calendar(region: BoundingBox) -> CropCalendarResult
# Returns expected growth stage for crops in region given current date
# Useful for contextualizing whether NDVI values are expected or anomalous
# **OPTIONAL TOOL** — implementation complexity is high, consider omitting for 75-min lecture
```

### Design Notes for Tools
- Each tool should return both machine-readable data (for the agent) and a human-readable summary string. The summary string is what gets fed back into the conversation as the tool result.
- Tools should not call each other. Keep them flat.
- **Error handling strategy**: Tools should NOT raise exceptions. Instead, return a result object with a `success` boolean and either `data` + `summary` OR `error` + `error_message`. This allows the agent to reason about failures. Example:
  ```python
  if baseline_missing:
      return DiffResult(
          success=False,
          error_message="No baseline image found for this region. Cannot compute change."
      )
  ```
- Include realistic failure modes: `compare_to_baseline` should fail gracefully if no baseline exists. Test error paths explicitly.
- Test each tool in `01_tools.ipynb` in isolation before introducing the agent.

---

## The Agent Loop

Use the **Anthropic Messages API** (or OpenAI if preferred) with native tool calling. Do not use LangChain, LlamaIndex, or any agent framework. The loop should be readable in one screen.

**Model configuration for determinism**: Use `temperature=0` for Scene A to ensure consistent tool call sequences during the scripted walkthrough. For Scene B (live demo), you may use default temperature to show more natural variation.

### System Prompt (DRAFT)

This is the most critical pedagogical artifact after the loop itself. Draft and test this early.

```
You are an autonomous crop health analyst working with satellite imagery. Your job is to assess crop health in a given region and produce a diagnostic report.

You have access to tools that compute spectral indices (NDVI, NDWI, EVI), retrieve timeseries data, flag anomalous regions, and compare current conditions to historical baselines.

**Your workflow**:
1. Start with a broad assessment (e.g., compute NDVI for the region)
2. If you detect anomalies or low values, investigate further using additional indices or timeseries
3. Explain your reasoning in plain text before each tool call. Describe what you observed and why you're choosing the next tool.
4. Only call tools when warranted by your observations. You do not need to use all available tools.
5. If tools return errors, adapt your strategy and explain what happened.
6. When you have sufficient information, produce a final diagnostic report with:
   - Summary of findings
   - Specific locations of concern (if any)
   - Likely cause of stress (water, nutrients, pests) with confidence level
   - Recommended follow-up actions
7. Express uncertainty when appropriate. If indices give conflicting signals, say so.

**Important**: Be concise in your reasoning (2-3 sentences per tool call). The goal is clarity, not verbosity.
```

**Teaching notes**:
- Walk through each instruction and explain its pedagogical purpose
- The "explain your reasoning" instruction is what creates the trace students will learn from
- The "only call tools when warranted" instruction prevents the agent from exhaustively calling everything
- The "express uncertainty" instruction prevents overconfident reports (addresses a key failure mode)

### Loop Structure (pseudocode)

```python
messages = [{"role": "user", "content": user_request}]

while True:
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=messages
    )
    
    messages.append({"role": "assistant", "content": response.content})
    
    if response.stop_reason == "end_turn":
        print(response.content)  # Final report
        break
    
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[TOOL CALL] {block.name}")
                print(f"[INPUT] {block.input}")
                result = dispatch_tool(block.name, block.input)
                print(f"[RESULT] {result.summary}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.summary
                })
        messages.append({"role": "user", "content": tool_results})
```

The printed trace — tool name, input, result — is the main teaching artifact in the notebook. Students should be able to follow the agent's reasoning step by step from the output.

---

## Notebook 02: Agent Loop — Section Outline

**Goal**: Minimize time explaining code, maximize time observing agent behavior. Students can read code later; the live trace is the unique teaching moment.

1. **Setup** — imports, API key, load Scene A (2 min)
2. **One tool example** — Show ONE tool (`compute_ndvi`) called manually in Python, show its output structure. This proves tools are "just functions." (3 min)
3. **One schema example** — Show the JSON schema for that same tool. Explain: "This is how the model knows what parameters to send." Don't walk through all schemas. (2 min)
4. **The system prompt** — Read it together, discuss what each instruction is doing. This is time well spent. (5 min)
5. **The loop** — Walk through the code quickly. Focus on: "Model returns tool_use blocks → we execute → send results back." Don't explain Python basics. (5 min)
6. **Scripted run on Scene A** — Run the agent, read the trace together. Pause after each tool call to discuss: "Why did it call this? What did it learn? What might it do next?" This is the core teaching section. (20 min)
7. **Live run on Scene B** — Let it go, narrate what's happening in real time. Show that the agent adapts to different inputs. (10 min)

**Total: ~47 minutes**, leaving buffer for questions and API latency.

---

## Notebook 03: Failure Modes — Section Outline

**This is TAKE-HOME material, not lecture content.** Mention it briefly (2 min) at the end of lecture, tell students it's in the repo, and move on. This notebook teaches more about agentic design than the success cases, but there's no time to cover it live.

**Content for the notebook**:

1. **Bad tool description** — Rename a tool parameter ambiguously (e.g., `region` → `area`), show the agent misuse it or fail to call the tool. Lesson: tool descriptions must be unambiguous.

2. **Missing baseline** — Run `compare_to_baseline` when no baseline exists. Show two versions:
   - Tool raises an exception (bad: breaks the loop)
   - Tool returns an error message (good: agent can adapt)

3. **Contradictory indices** — Construct a scene where NDVI and EVI disagree (e.g., dense canopy saturates NDVI but EVI is fine). Does the agent notice? Does it explain the conflict? Add a system prompt variant that explicitly tells the agent to check for contradictions.

4. **Infinite loop risk** — Discuss what happens if `stop_reason` is never `"end_turn"`. Show a `max_iterations` guard in the loop:
   ```python
   for iteration in range(MAX_ITERATIONS):
       response = client.messages.create(...)
       if response.stop_reason == "end_turn":
           break
   else:
       print("Warning: Hit max iterations without completing")
   ```

5. **Overconfident report** — Remove the "express uncertainty" instruction from the system prompt. Show the agent produce a confident diagnosis even when data is ambiguous. Contrast with the properly-prompted version.

6. **Tool result hallucination** — Show a case where the agent ignores tool results and reasons from priors instead. Discuss mitigation strategies (stronger system prompt, output validation).

**Each section should**:
- Show the failure in a runnable cell
- Explain why it failed
- Show the fix
- Include a "reflection question" for students

---

## Lecture Flow (75 minutes)

| Time | Section |
|------|---------|
| 0–10 min | Background: Sentinel-2, spectral indices, the ambiguity problem |
| 10–20 min | Fixed pipeline vs. agentic pipeline — the core conceptual shift |
| 20–25 min | One tool + one schema example — prove tools are "just functions" |
| 25–30 min | The system prompt — read together, discuss each instruction |
| 30–35 min | The loop code — quick walkthrough, focus on tool_use flow |
| 35–55 min | Scripted run on Scene A — read trace together, discuss each tool call |
| 55–65 min | Live run on Scene B — narrate in real time, show adaptation |
| 65–73 min | Where this goes in practice: MCP, multi-agent, HITL, evaluation |
| 73–75 min | Closing: failure modes notebook (take-home), student exercises, Q&A buffer |

**Teaching notes**:
- The scripted run (Scene A) is the most important section — protect this time
- Have pre-recorded traces as backup for API issues
- Use temperature=0 for Scene A to ensure consistent behavior across rehearsals and live delivery

---

## Closing: Where This Goes in Practice (8 minutes)

**Framing**: "The 50-line loop you just saw is not a toy. It's the real abstraction. Production frameworks are just packaging what you now understand."

### The Core Loop is the Abstraction

The loop you learned:
```python
while True:
    response = call_model(messages, tools)
    if done: break
    results = execute_tools(response)
    messages.append(results)
```

**This is the actual pattern.** Frameworks like LangChain, LlamaIndex, and protocols like MCP don't replace this — they standardize and scale it.

### Production Extensions

**MCP (Model Context Protocol)** — Lets you define tools once and use them across agents, or serve tools over a network. The JSON schemas you wrote today are compatible with MCP. The loop stays the same.

**Multi-agent systems** — One agent's output becomes another agent's input. Example: your crop health agent produces a report, which feeds a resource allocation agent that optimizes irrigation schedules. The loop runs multiple times with different roles.

**Human-in-the-loop** — Add a tool called `escalate_to_human(reason: str, question: str)` that pauses the loop and waits for human input. The agent decides when it needs help. This is the same loop, just with a blocking tool.

**Evaluation** — How do you know the agent is doing the right thing?
- Trace logging (you saw this today)
- Tool call auditing (did it call reasonable tools in reasonable order?)
- Output validation (does the report match ground truth?)
- This is still an open research problem.

**The message**: You now understand the foundation. Everything else is engineering around this core pattern.

---

## Student Exercises (Take-Home)

These exercises reinforce the core concepts and let students build on what they learned. Include them in the README or as a separate `EXERCISES.md` file.

### Exercise 1: Add a New Tool (Easy)
**Goal**: Understand tool definition and integration.

Implement a new tool `compute_savi(region: BoundingBox, L: float = 0.5) -> SAVIResult` that computes the Soil-Adjusted Vegetation Index:
```
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
```

Requirements:
- Write the function in `tools.py`
- Write the JSON schema in `schemas.py`
- Add it to the tool list in the agent loop
- Run the agent and verify it calls your tool when appropriate

**Reflection question**: Did the agent know when to use SAVI vs. NDVI? Why or why not?

### Exercise 2: Modify the System Prompt (Medium)
**Goal**: Understand how instructions shape agent behavior.

Create three variants of the system prompt:
1. **Minimal**: Remove all instructions about reasoning and uncertainty. Just tell it the role.
2. **Directive**: Add explicit instructions like "Always call NDVI before NDWI" or "Always check timeseries for anomalies."
3. **Original**: The prompt from the lecture.

Run the same scene with all three prompts. Compare:
- Tool call sequences
- Quality of reasoning
- Handling of ambiguous cases

**Reflection question**: Which prompt produces the most useful behavior? Why?

### Exercise 3: Build an Error-Handling Tool (Medium)
**Goal**: Understand tool error handling and agent adaptation.

Modify `compare_to_baseline` to intentionally fail 50% of the time (random). Observe how the agent handles the error. Then create a new tool:
```python
check_baseline_exists(region: BoundingBox) -> BaselineCheckResult
# Returns whether a baseline exists for the region
```

Modify the system prompt to suggest the agent check for baselines before comparing. Does the agent use the new tool? Does it avoid the error?

**Reflection question**: What are the tradeoffs between preventing errors (defensive tools) vs. handling errors (error recovery)?

### Exercise 4: Build a Multi-Scene Comparison Agent (Advanced)
**Goal**: Extend the agent to handle more complex workflows.

Create a new user request: "Compare crop health in Scene A vs. Scene B and identify which region has more severe stress."

You'll need to:
- Modify the system prompt to handle multi-scene tasks
- Possibly add a tool like `switch_scene(scene_id: str)` or refactor tools to accept scene as a parameter
- Run the agent and analyze the trace

**Reflection question**: Does the agent develop a coherent comparison strategy, or does it get confused? What would you change?

### Exercise 5: Implement a Simple Evaluation (Advanced)
**Goal**: Understand evaluation challenges for agentic systems.

Create a ground truth dataset: manually label 3-5 scenes with known issues (e.g., "Scene 1: water stress in northwest quadrant, Scene 2: no anomalies").

Run your agent on each scene and build an evaluator that checks:
1. Did it identify the correct issue (if any)?
2. Did it call reasonable tools?
3. Did it express appropriate confidence?

**Reflection question**: What's harder to evaluate — correctness of the final report, or quality of the reasoning process? Why does this matter?

---

## Implementation Notes for Developer

### Model and API

- **Model**: Use `claude-sonnet-4-5` or later. Avoid smaller models — tool calling quality degrades noticeably and will distract from the pedagogy.
- **Temperature**: Set `temperature=0` for Scene A (scripted walkthrough) to ensure deterministic tool call sequences. Test this thoroughly — run Scene A 5-10 times and verify the tool path is consistent.
- **API keys**: Decide on key strategy:
  - **Option 1**: Students bring their own Anthropic/OpenAI keys (cheapest for you, but requires setup and budget)
  - **Option 2**: Provide a shared institutional key with rate limits (easier for students, but you need budget)
  - **Option 3**: Run in mock mode (see below) — no API calls, uses pre-recorded responses
- **API latency**: Pre-run all notebooks and cache outputs before lecture. Have pre-recorded traces as fallback.

### Tool Implementation

- Keep tool implementations simple and dependency-light. Core stack: `rasterio`, `numpy`, `matplotlib`
- Avoid `geopandas` or heavy geospatial stacks unless the audience is already familiar
- The `dispatch_tool` function is just a dictionary mapping tool name strings to Python functions:
  ```python
  TOOLS = {
      "compute_ndvi": compute_ndvi,
      "compute_ndwi": compute_ndwi,
      # ...
  }

  def dispatch_tool(name: str, input: dict):
      return TOOLS[name](**input)
  ```
  Show this explicitly — students often expect something more magical.

### Testing and Determinism

- Test Scene A with `temperature=0` multiple times before lecture. The tool call sequence should be identical across runs. If it varies, the scene may be ambiguous or the system prompt may be under-specified.
- Test all error paths in tools. Make sure `compare_to_baseline` fails gracefully when there's no baseline.
- Run the full agent loop on both scenes at least once per day leading up to the lecture. API behavior can drift with model updates.

### Mock Mode (Fallback)

Add a `--mock` flag to the agent loop that returns canned tool results without calling the API. Useful for:
- Offline development
- Classroom demos if API is down
- Cost control during testing

Implementation sketch:
```python
if args.mock:
    response = load_prerecorded_response(scene_id, turn_number)
else:
    response = client.messages.create(...)
```

### Repository Hygiene

- Keep total repo size under 200MB (scenes are the biggest contributor)
- Target < 500 lines of code for the core implementation (tools + loop + schemas)
- Include a `requirements.txt` or `environment.yml` with pinned versions
- Add a `Makefile` with targets:
  - `make setup` — install dependencies
  - `make download-data` — (optional) fetch Sentinel-2 scenes
  - `make test` — run tool tests from `01_tools.ipynb`
  - `make demo` — run Scene A in mock mode

### Data Acquisition Makefile Target

Example `Makefile` snippet:
```makefile
download-data:
	@echo "Downloading Sentinel-2 scenes..."
	@python data/download_data.py --region "37.5,-120.5,37.7,-120.3" --date "2023-07-15"
	@echo "Download complete. Scenes saved to data/scenes/"
```

Or if using a shell script:
```makefile
download-data:
	@bash data/download_data.sh
```

Make this optional and document it as "for learning purposes only — pre-processed scenes are already in the repo."

### Audience Considerations

- **Python proficiency**: Assume students can read intermediate Python (dicts, loops, list comprehensions). Don't explain language basics.
- **Geospatial background**: Do not assume. Avoid jargon like "CRS" or "reprojection" without explanation.
- **API familiarity**: Some students may never have called an LLM API. Show one raw `curl` example in slides or README so they understand it's just HTTP + JSON.

---

## Open Questions for Implementation

These decisions should be made during the build phase:

1. **Scene selection**: Which specific Sentinel-2 scenes will you use? Candidate regions:
   - Central Valley California (good crop diversity, well-documented)
   - Midwest US (corn/soy, well-studied)
   - Mediterranean agricultural regions (if you want international context)
   - **Action**: Scout 5-10 candidate scenes, test with the agent, select the two with best pedagogical properties

2. **Crop calendar tool**: Keep or drop?
   - Pro: Adds temporal reasoning, shows agent can handle uncertainty
   - Con: High implementation complexity, requires crop type maps
   - **Recommendation**: Start without it. Add only if time permits and you have reliable crop data sources.

3. **Mock mode implementation strategy**:
   - Option A: Record full API responses as JSON, replay them
   - Option B: Hardcode simplified responses in Python
   - **Recommendation**: Option A (JSON recording) is more maintainable and allows switching between mock/live easily

4. **Student Python proficiency baseline**: Will students be comfortable with:
   - Dictionary comprehensions?
   - Type hints?
   - Context managers (`with` statements)?
   - **Action**: Survey or check course prerequisites, adjust code complexity accordingly

5. **Notebook execution model**:
   - Should notebooks be run top-to-bottom in lecture, or will some cells be pre-run?
   - How will you handle long-running cells (e.g., NDVI computation)?
   - **Recommendation**: Pre-compute heavy operations, cache results, focus live execution on the agent loop

---

## Pre-Lecture Checklist

Use this checklist the week before lecture to ensure everything works.

### One Week Before
- [ ] Identify and download Scene A and Scene B
- [ ] Pre-process scenes to NumPy arrays or GeoTIFF, add metadata JSON
- [ ] Verify scene file sizes (target < 50MB each)
- [ ] Implement all tools in `tools.py` and test individually in `01_tools.ipynb`
- [ ] Write JSON schemas in `schemas.py`
- [ ] Implement the agent loop in `loop.py` or directly in `02_agent_loop.ipynb`
- [ ] Draft system prompt and test it

### Three Days Before
- [ ] Run Scene A with `temperature=0` ten times. Verify tool call sequence is consistent.
- [ ] If not consistent, refine scene selection or system prompt until deterministic
- [ ] Run Scene B at least once. Verify it takes a different path than Scene A.
- [ ] Create mock mode responses for both scenes (fallback for API issues)
- [ ] Pre-run all notebooks and save outputs (API latency buffer)
- [ ] Test on a clean Python environment to catch missing dependencies

### One Day Before
- [ ] Rehearse lecture with live notebooks. Time each section.
- [ ] Verify API key works (or that mock mode works if using that)
- [ ] Export notebooks to PDF as backup (in case Jupyter fails during lecture)
- [ ] Prepare fallback: pre-recorded terminal session showing Scene A trace
- [ ] Test on lecture room computer/projector if possible
- [ ] Have a backup plan for internet outage (mock mode + exported notebook PDFs)

### Day of Lecture
- [ ] Bring USB drive with full repo + exported PDFs + mock responses
- [ ] Test API connection before students arrive
- [ ] Have fallback plan ready (use mock mode or pre-recorded trace)
- [ ] Clear all notebook outputs before lecture (run from scratch for authenticity)
- [ ] Start Jupyter server before class, navigate to notebooks

### After Lecture
- [ ] Push any live-coding changes to repo
- [ ] Share link to repo with students
- [ ] Remind students about `EXERCISES.md` and Notebook 03 (take-home)
- [ ] (Optional) Set up office hours for students who want to discuss their agent experiments
