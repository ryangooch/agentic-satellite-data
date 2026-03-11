# Agentic AI for Signal Processing
## Satellite Crop Health Assessment via Autonomous Multi-Band Image Analysis

ECE Senior / First-Year Graduate Lecture · 75 min

---

## Goal

We will develop a ReAct-style LLM-based tool-calling agent that will be able to analyze and reason
about satellite imagery of crops and other vegetation

It will do this by using derived properties based on satellite measurements across different multi-
channel bands, made available in the form of functions (tools).

---

## Takeaways (I hope)

* Human-based subject matter expertise is critical
* Bespoke agents with relevant tools are extremely powerful
* Getting good at interacting with LLMs is important
* Domain knowledge baked into "tools" + text-based interaction with "intelligence" is also powerful
* Some new neuron connections will be built

---

## Agenda

**Part 1 -- The Signal Processing Domain** _(~10 min)_
- Sentinel-2 satellite imagery
- Spectral indices: NDVI, NDWI, EVI
- The ambiguity problem

**Part 2 -- Why an Agent?** _(~10 min)_
- Fixed pipelines, deep learning, and where they break
- Agents vs. bespoke models: real tradeoffs

**Part 3 -- How Agents Work** _(~12 min)_
- Tool calling at the API level
- The ReAct framework
- The loop (~80 lines of real code)

**Part 4 -- Modern Research** _(~8 min)_
- Foundation models for Earth observation
- Multi-agent systems, MCP, HITL
- Open problems and research frontiers

**Then: live demo** _(~30 min)_

---

## Why Satellite Imagery?

Remote sensing for agriculture has been an active research area since the 1970s (Landsat).
Sentinel-2 made high-resolution multi-spectral data free and globally available.

- Millions of pixels per tile, updated every 5 days globally
- 13 spectral bands
    - each band measures a different wavelength range
    - each correlated with different physical properties of the surface
- Temporal stack: changes between acquisitions often more diagnostic than any single image

The volume and dimensionality of this data imposes challenges of scale and velocity

---

## Sentinel-2: Geospatial Observation Platform

Operated by the European Space Agency (ESA) since 2015

| Property | Value |
|----------|-------|
| Orbits | Twin satellites (A + B) |
| Revisit time | **5 days** at equator |
| Spatial resolution | **10m** (visible/NIR), 20m (SWIR) |
| Spectral bands | **13 bands**: 443–2190 nm |
| Coverage | Global land surface |
| Access | **Free and open** (Copernicus) |

---

## The Sentinel-2 Band Map

```
Band  | Name         | λ (nm) | Res  | What it sees
------|--------------|--------|------|---------------------------
B02   | Blue         |  490   | 10m  | Reflectance, water bodies
B03   | Green        |  560   | 10m  | Vegetation peak
B04   | Red          |  665   | 10m  | Chlorophyll absorption
B05   | Red-edge     |  705   | 20m  | Canopy stress, senescence
B08   | NIR          |  842   | 10m  | Biomass, leaf area index
B8A   | Narrow NIR   |  865   | 20m  | Vegetation structure
B11   | SWIR-1       | 1610   | 20m  | Soil moisture, plant water
B12   | SWIR-2       | 2190   | 20m  | Dry biomass, geology
```

_Healthy plants absorb red and reflect NIR -- the key signature we exploit._

---

## Abbreviations and Descriptions

* NIR = Near Infrared
* SWIR = Short-wavelength infrared
* Chlorophyll => absorbs red & blue light, but reflects NIR more (healthy, dense canpoy)
* leaf water => absorbs SWIR, so higher SWIR => drier leaves
* NIR high, Red low => healthy, photosynthetically active vegetation
* SWIR low => leaves are water-rich and turgid, since water absorbs SWIR

---

## Abbreviations and Descriptions

* healthy plant cells scatter NIR strongly
  * stressed, dying plants lose that structure, so NIR returns drop
* reflectance = fraction of incoming solar radiation that bounces back from a surface
  * Sentinel-2 Level-2A gives atmos-corrected surface reflectance, removing signal from clouds, haze, etc
* vegetation peak = Green channel reflectance (usually greener plants are healthier plants)
* canopy stress = aggregate optical signal of a crop under physiological stress
* senescence = the natural aging and death of leaves at the end of a growing season
* biomass = total living plant material per unit area (NIR reflectance correlates, more leaf, more scatter)

---

## How to make an Irrigation Decision?

A crop that's being under-irrigated will:
  1. First lose leaf turgor (cells shrink) → SWIR reflectance increases → NDWI drops ← early warning, before visible
  symptoms
  2. Then start closing stomata and reducing photosynthesis → NIR drops slightly → NDVI starts declining
  3. Then if prolonged, begin chlorophyll degradation → Red reflectance rises → NDVI drops sharply
  4. The canopy also gets warmer (can't cool via transpiration) → thermal bands show elevated temperature

To reduce stress on the plant and improve health, it is best to catch this signs early!

---

## Transpiration -- Why Do Plants Actually Lose Water?

What is actually happening here?

---

## Transpiration -- Why Do Plants Actually Lose Water?

What is actually happening here?

* Plants use a physical process called Transpiration to move water from roots through stem to leaves
* Tiny pores under leaves (stomata) open to let CO2 in for photosynthesis
  * Water vapor diffuses out (more dry -> more diffusion)
  * This creates a negative pressure in the leaf cells, which pulls water up
* like a "wick"
* When it is hot/dry, the "vapor pressure deficit" (VPD) is higher, so plants lose water faster
* When it is sunny, stomata open wider, leading to more water escaping per unit time
* High wind can steepen the gradient

---

## Spectral Indices: NDVI

**Normalized Difference Vegetation Index**

```
NDVI = (NIR - Red) / (NIR + Red)
```

Range: **[-1, +1]**

| Value | Interpretation |
|-------|---------------|
| < 0   | Water, bare rock |
| 0–0.2 | Bare soil, sparse vegetation |
| 0.2–0.4 | Low/stressed vegetation |
| 0.4–0.7 | Moderate healthy vegetation |
| > 0.7 | Dense healthy canopy |

**Known limitation**: saturates above ~0.8 in dense canopy

---

## NDWI: Water in the Canopy

**Normalized Difference Water Index**

```
NDWI = (Green - SWIR) / (Green + SWIR)
```

- Positive → canopy has moisture (well-watered)
- Negative → water stress or dry soil

**Why we need it**: A field can have _normal_ NDVI but _low_ NDWI
→ crop is structurally intact but already experiencing water stress
→ NDWI catches stress earlier than NDVI

**Why earlier?** NDVI tracks canopy greenness and structure (chlorophyll + leaf area).
NDWI is more directly sensitive to liquid water in the leaf's spongy mesophyll -- the
loosely-packed cell layer where water is lost first under stress, before visible
wilting or chlorophyll breakdown begins.

---

## EVI: When NDVI Saturates

**Enhanced Vegetation Index**

```
EVI = 2.5 × (NIR - Red) / (NIR + C1·Red - C2·Blue + 1)
```

where C1 and C2 are tuneable parameters

- Less saturation in dense canopy
- More sensitive to canopy structure
- Reduces atmospheric and soil background noise

**Rule of thumb**:
- Start with NDVI (fast, interpretable)
- Confirm with EVI when NDVI is high or ambiguous
- Use NDWI when investigating water/moisture stress

---

## What Crop Stress Looks Like Spectrally

| Stress type | NDVI | NDWI | EVI | Other signals |
|-------------|------|------|-----|---------------|
| Water stress | ↓ moderate | ↓ early | ↓ | Thermal: warmer canopy |
| Nutrient deficiency | ↓ patchy | normal | ↓ patchy | Red-edge shift |
| Pest/disease | ↓ localized | varies | ↓ | Rapid change in timeseries |
| Healthy | high, stable | positive | moderate-high | -- |

**The core problem**: stress signatures _overlap spectrally_

A field with low NDWI could be:
1. Under-irrigated
2. Over-shaded by clouds at image time
3. Late-season senescence (normal)
4. Pest damage

---

## The Ambiguity Problem

> A fixed pipeline that computes NDVI → thresholds → alert
> cannot distinguish these cases.

* Context-sensitive reasoning under uncertainty

---

## The Ambiguity Problem

```
"NDWI is negative AND NDVI declined 0.15 since last week
 AND the baseline from June shows this field was healthy
 AND the crop calendar says this is mid-season peak...
 → water stress, high confidence. Recommend irrigation audit."
```

vs.

```
"NDWI is negative but NDVI is normal AND timeseries is stable
 AND cloud cover was 40% at acquisition time
 → inconclusive. Recommend re-image in 5 days."
```

---

## The Fixed Pipeline Approach

```python
# Traditional crop monitoring script
ndvi = compute_ndvi(scene)
if ndvi.mean < 0.35:
    alert("Low vegetation health", region=scene.bounds)
```

Shortcomings?

---

## The Fixed Pipeline Approach

```python
# Traditional crop monitoring script
ndvi = compute_ndvi(scene)
if ndvi.mean < 0.35:
    alert("Low vegetation health", region=scene.bounds)
```

Shortcomings:
- Threshold is arbitrary -- wrong for every crop/season combination
- No temporal context -- can't distinguish new stress from chronic
- No uncertainty -- outputs an alert or nothing, never "probably"
- Can't integrate multiple signals -- NDVI alone doesn't distinguish water stress from senescence

---

## The Deep Learning Approach

* Train a model end-to-end
* Estimate stress levels per-pixel
* Strengths?
* Weaknesses?

---

## The Deep Learning Approach

**Strengths**:
- Can learn complex non-linear spectral patterns
- State-of-the-art accuracy on benchmark datasets
- No hand-tuned thresholds
- (Perhaps) less bias

**Limitations for this problem**:
- Needs labeled training data per crop type, region, season
- Black box
- Hard to update when sensor changes or new indices are added
- Doesn't reason about uncertainty -- it produces a label
- **Can't call external tools (timeseries DB, crop calendar, weather)**

_Deep learning is excellent for perception; it's less suited for the reasoning layer._

---

## Agent vs. Deep Learning: Tradeoffs

|  | Deep Learning | Agentic LLM |
|--|---------------|-------------|
| Accuracy (benchmark) | ✅ High | 🔶 Depends on tools |
| Explainability | ❌ Black box | ✅ Generates reasoning trace |
| Data requirements | ❌ Large labeled sets | ✅ Tools + prompt |
| Adaptability | ❌ Retrain needed | ✅ Prompt + tool update |
| Cost per query | ✅ Cheap inference | 🔶 API cost |
| Inference Latency | ✅ Milliseconds | 🔶 Seconds |
| Tool integration | ❌ Bespoke wrappers | ✅ Native tool calling |
| Uncertainty expression | ❌ Rare | ✅ Built into prompt |

Hybrid architectures often win: DL for perception, agent for reasoning and orchestration.

---

## What Makes a System "Agentic"?

1. Tool calling
2. Multi-turn reasoning
3. Conditional execution
4. Grounded output

```
User: "Assess crop health."

Loop:
  [Model reasons] "I'll start with NDVI" → [calls compute_ndvi]
  [Tool returns] NDVI result with low fraction = 0.23
  [Model reasons] "Low NDVI. Let me check moisture." → [calls compute_ndwi]
  [Tool returns] NDWI = -0.18 (water stressed)
  [Model reasons] "Declining trend?" → [calls get_pixel_timeseries]
  [Tool returns] NDVI declined 0.15 in past 10 days
  [Model reasons] → "Water stress confirmed. Writing report." → END
```

_The agent decided which tools to call, in what order, based on what it observed._

---

## Tool Calling: The API Mechanic

The model API accepts a `tools` parameter: a list of JSON schemas.

```json
{
  "name": "compute_ndvi",
  "description": "Compute NDVI for a spatial region. Returns mean, std, ...",
  "input_schema": {
    "type": "object",
    "properties": {
      "region": {
        "type": "object",
        "properties": {
          "row_min": {"type": "integer"},
          "row_max": {"type": "integer"},
          "col_min": {"type": "integer"},
          "col_max": {"type": "integer"}
        },
        "required": ["row_min", "row_max", "col_min", "col_max"]
      }
    },
    "required": ["region"]
  }
}
```

Q: what sticks out to you about this approach? benefits? weaknesses?

---

## Tool = JSON Schema + Python Function

```python
# In tools.py -- just a function
def compute_ndvi(region: BoundingBox) -> NDVIResult:
    """Compute NDVI = (NIR - Red) / (NIR + Red)."""
    nir  = scene.get_band("B08", region)
    red  = scene.get_band("B04", region)
    ndvi = (nir - red) / (nir + red + 1e-8)
    mean = float(ndvi.mean())
    low_frac = float((ndvi < 0.3).mean())
    return NDVIResult(
        mean=mean, low_fraction=low_frac,
        summary=f"Mean NDVI={mean:.3f}. {low_frac:.1%} pixels stressed."
    )
```

The dispatcher is just a dict:

```python
TOOL_MAP = {
    "compute_ndvi": compute_ndvi,
    "compute_ndwi": compute_ndwi,
    "compute_evi":  compute_evi,
    # ...
}

def dispatch_tool(name, input_dict):
    return TOOL_MAP[name](**input_dict)
```

_tools are just functions_

---

## ReAct: Reason + Act

The widely-adopted prompting pattern for tool-calling agents.

> Yao et al. (2023). _"ReAct: Synergizing Reasoning and Acting in Language Models."_ ICLR 2023.

The core idea: interleave **thought** and **action** in the context window.

```
Thought: NDVI mean is 0.31, with 23% of pixels below 0.3.
         This is concerning -- let me check water content.
Action:  compute_ndwi(region)
Obs:     mean NDWI = -0.18. Status: water-stressed.

Thought: NDWI confirms moisture deficit. Is this recent or chronic?
Action:  get_pixel_timeseries(lat=50, lon=75, index="ndvi")
Obs:     NDVI declined 0.15 in past 10 days. Trend: declining.

Thought: Recent onset + water stress → irrigation problem.
         High confidence. Writing final report.
Action:  [end_turn]
```

---

## The System Prompt

Every behavior we want is specified here:

```
You are an autonomous crop health analyst working with satellite imagery.

Your workflow:
1. Start with a broad assessment (e.g., compute NDVI for the region)
2. If you detect anomalies, investigate further
3. Explain your reasoning before each tool call
4. Only call tools when warranted -- you do not need all of them
5. If tools return errors, adapt your strategy
6. When done, write a diagnostic report with:
   - Summary of findings
   - Specific locations of concern
   - Likely cause of stress with confidence level
   - Recommended follow-up actions
7. Express uncertainty when indices give conflicting signals
```

---

## The Agent Loop

_The entire agentic pattern in ~30 lines._

```python
messages = [{"role": "user", "content": user_request}]

for _ in range(MAX_ITERATIONS):
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=messages,
        temperature=0,
    )

    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        return extract_text(response)   # Done

    if response.stop_reason == "tool_use":
        results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.summary,
                })
        messages.append({"role": "user", "content": results})
```

---

## What the Loop Looks Like Running

```
[TOOL CALL] compute_ndvi
[INPUT]  {"region": {"row_min": 0, "row_max": 256, ...}}
[RESULT] Mean NDVI=0.312, std=0.091. Pixels below 0.3: 23.4%. ⚠️ Significant stressed area.

[TOOL CALL] flag_anomalous_regions
[INPUT]  {"index": "ndvi", "threshold": 0.3, "direction": "below"}
[RESULT] Found 2 anomalous regions. Region 1: rows 64–128, cols 128–192, mean NDVI=0.241.

[TOOL CALL] compute_ndwi
[INPUT]  {"region": {"row_min": 64, "row_max": 128, "col_min": 128, "col_max": 192}}
[RESULT] NDWI mean=-0.21. Status: water-stressed. Negative pixels: 78.3%.

[TOOL CALL] get_pixel_timeseries
[INPUT]  {"lat": 96, "lon": 160, "index": "ndvi"}
[RESULT] NDVI declining (Δ≈-0.17 -- possible recent stress onset).

============================================================
FINAL REPORT
Anomalous region in rows 64–128, cols 128–192 shows...
```

---

## Architecture

```
User request
      │
      ▼
┌─────────────┐     TOOL_SCHEMAS (JSON)    ┌──────────────────┐
│  Agent Loop │ ─────────────────────────► │  Claude API      │
│  (loop.py)  │ ◄───── tool_use blocks ─── │  (claude-sonnet) │
└──────┬──────┘                            └──────────────────┘
       │
       ▼ dispatch_tool(name, input)
┌────────────────────────────────────────────┐
│               tools.py                     │
│  compute_ndvi   compute_ndwi   compute_evi │
│  flag_anomalous_regions                    │
│  get_pixel_timeseries                      │
│  compare_to_baseline                       │
└───────────────┬────────────────────────────┘
                │
                ▼
         scene.py (synthetic Sentinel-2 data)
         NumPy arrays + timeseries + baseline
```

---

## Our Toolset

| Tool | What it does | Why the agent needs it |
|------|-------------|------------------------|
| `compute_ndvi` | NDVI map, mean, stressed-pixel fraction | First-pass overview |
| `compute_ndwi` | Water/moisture stress index | Distinguish water vs. nutrient stress |
| `compute_evi` | Less-saturated vegetation index | Cross-check NDVI in dense canopy |
| `flag_anomalous_regions` | Find grid cells below/above threshold | Localize stress spatially |
| `get_pixel_timeseries` | NDVI history at a point | Distinguish new stress from chronic |
| `compare_to_baseline` | Diff current vs. earlier date | Quantify change, confirm onset |

Design rules:
- Tools are **flat** -- no tool calls another tool
- Tools **never raise** -- return `success=False` with an error message
- Every tool returns a `summary` string → that's what goes into the context window

---

## Now: Demo

**Scene A** -- scripted walkthrough (temperature=0)
_We'll read the trace together and ask: why did it call that? what did it learn?_

**Scene B** -- live run
_Same agent, different data. Watch it take a different path._

```bash
make demo           # Scene A, mock mode (offline)
uv run python -m agent.loop --scene scene_a
uv run python -m agent.loop --scene scene_b
```

→ _Notebooks: `02_agent_loop.ipynb`_

---

## Modern Research: Foundation Models for Earth Observation

LLMs are not the only foundation model in this space.

**Prithvi** (NASA + IBM, 2023)
- 300M-parameter vision transformer pre-trained on Harmonized Landsat Sentinel-2 (HLS) data
- Multi-temporal, multi-spectral -- understands temporal change natively
- Fine-tuned for flood mapping, crop segmentation, wildfire scars

**SatMAE** (Cong et al., 2022)
- Masked autoencoder pre-training on multi-spectral satellite data
- Learns spatial-spectral representations without labels

**RemoteCLIP** (Liu et al., 2023)
- CLIP-style vision-language model for satellite images
- Enables zero-shot scene understanding from natural language queries

_These are the "perception" layer -- the deep learning models that agents could invoke as tools._

---

## Foundation Models + Agents: The Emerging Stack

```
Natural language query
      │
      ▼
  Agent (LLM)                ← reasoning, planning, tool selection
      │
      ├── compute_ndvi()     ← classical signal processing
      ├── run_prithvi()      ← foundation model inference
      ├── query_timeseries() ← database tool
      └── escalate_to_human()← HITL
```

The agent:
* orchestrates heterogeneous tools:
  * classical DSP
  * neural networks
  * database queries
  * etc

---

## GeoAI and Agentic Remote Sensing: Active Research

**GeoLLM** (Manvi et al., 2023) -- augmenting LLMs with geospatial context
**EarthGPT** (Zhang et al., 2024) -- multimodal remote sensing understanding
**AgriAgent** (various 2024 papers) -- agents for precision agriculture

**Open problems being actively researched**:

1. **Grounding**: Getting agents to reason about _coordinates_ and _spatial relationships_ reliably
2. **Multi-temporal reasoning**: Understanding sequences of satellite images, not just single scenes
3. **Sensor fusion**: Combining Sentinel-2, SAR (Sentinel-1), thermal (ECOSTRESS), weather data
4. **Evaluation**: How do you score an agent's reasoning process, not just its final answer?
5. **Calibration**: Agents tend toward overconfidence -- how do you quantify their uncertainty?

---

## Production Extensions: Where This Goes

**MCP -- Model Context Protocol**

Define tools once; serve them over a network; use from any agent.

```
JSON schema from today → MCP server → any LLM agent
```

**Agent Skills**

Define workflows in markdown, plain text, that agents can follow.

```
Encode tools in `./scripts.py` and explain to the model how to use them
```


**Multi-agent systems**

```
Crop health agent → report → Irrigation agent → schedule → Operator dashboard
```

One agent's output becomes another's input. The 50-line loop runs multiple times.

**Human-in-the-loop**

```python
def escalate_to_human(reason: str, question: str) -> HumanResponse:
    """Agent calls this when uncertain. Blocks until human responds."""
    ...
```

---

## The Evaluation Problem

How do you know if your agent is doing the right thing?

Easy:
- Did it produce a final report?
- Did it call tools in a plausible order?

Hard:
- Was the _reasoning_ correct?
- Did it express uncertainty when it should have?
- Did it fail gracefully when tools errored?
- Would it generalize to a new crop type / region?

Current practice:
- Trace logging
- LLM-as-judge
- Human expert review

---

## Take-Home: Notebook 03

**Failure modes** -- where agents break, and what to do about it:

1. Bad tool description → agent misuses or ignores tool
2. Unhandled tool exceptions → loop crashes
3. Contradictory indices → agent fails to notice
4. No max iteration guard → potential infinite loop
5. Missing "express uncertainty" instruction → overconfident reports
6. Tool result hallucination → agent ignores observed data

Each section: failure in a runnable cell → explanation → fix → reflection question.

```bash
# In the repo
jupyter notebook notebooks/03_failure_modes.ipynb
```

See `EXERCISES.md` for 5 take-home problems (add a new tool, modify the prompt, build an evaluator...).

---

## References

- Yao et al. (2023). **ReAct: Synergizing Reasoning and Acting in Language Models.** ICLR 2023.
- Wei et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** NeurIPS 2022.
- Jakubik et al. (2023). **Foundation Models for Generalist Geospatial AI (Prithvi).** arXiv:2310.18660.
- Cong et al. (2022). **SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.** NeurIPS 2022.
- Liu et al. (2023). **RemoteCLIP: A Vision Language Foundation Model for Remote Sensing.** arXiv:2306.11029.
- Manvi et al. (2023). **GeoLLM: Extracting Geospatial Knowledge from LLMs.** ICLR 2024.
- Rouse et al. (1974). **Monitoring the vernal advancement of retrogradation of natural vegetation.** (Original NDVI paper.)
- Gao (1996). **NDWI -- A normalized difference water index for remote sensing of vegetation liquid water.** Remote Sensing of Environment.
- Huete et al. (2002). **Overview of the radiometric and biophysical performance of the MODIS vegetation indices.** (EVI.)
- Anthropic (2024). **Model Context Protocol.** modelcontextprotocol.io

---

## Questions?

**Repo**: clone and run immediately -- no data download required

```bash
git clone <repo>
uv sync
make demo                                    # Scene A, mock mode, no API key needed
uv run python -m agent.loop --scene scene_a  # or run directly
uv run python -m agent.loop --scene scene_b
```

**Notebooks**:
- `00_data_exploration.ipynb` -- band visualization
- `01_tools.ipynb` -- tools as plain functions
- `02_agent_loop.ipynb` -- **start here**
- `03_failure_modes.ipynb` -- take-home

Estimated cost for all exercises: **$0.10–0.50** with `claude-sonnet-4-6`
