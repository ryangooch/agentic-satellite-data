# Agentic AI for Signal Processing
## Satellite Crop Health Assessment via Autonomous Multi-Band Image Analysis

A lecture on broader trends in systems powered by agentic AI through the lens of a crop irrigation
and monitoring challenge.

---

## Goal

We will develop a ReAct-style LLM-based tool-calling agent that will be able to analyze and reason
about satellite imagery of crops and other vegetation

It will do this by using derived properties based on satellite measurements across different multi-
channel bands, made available in the form of functions (tools)

We will leverage other Agentic AI patterns as well, including RAG, MCP, and Agent Skills

---

## Takeaways (I hope)

* Human-based subject matter expertise is critical
* Bespoke agents with relevant tools are extremely powerful
* Getting good at interacting with LLMs is important
* Domain knowledge baked into "tools" + text-based interaction with "intelligence" is also powerful
* Some new neuron connections will be built

---

## Agenda

**Part 0 -- What is the Problem and why do we care?**
- Overview of almond plantations and water stress
- Introduction to Satellites as a means of measurement of core properties

**Part 1 -- Current State of Agentic AI**
- Common patterns and terminology
- A bit of history
- When and how to use Agentic AI
- The ReAct Pattern

**Part 2 -- Satellites, Almonds, and Signal Processing**
- Almond irrigation -- challenges and impact
- Sentinel-2 satellite imagery
- Spectral indices: NDVI, EVI
- Temperature indices: CWSI

**Part 3 -- The How and Why of AI Agents**
- When to reach for an Agent vs fixed, deterministic patterns
- Tool calling at the API level
- The ReAct framework

**live demo**
- Visualize the data
- Examine the code
- Run the agentic loop

**Part 4 -- Modern Research**
- Foundation models for Earth observation
- Multi-agent systems, MCP, HITL
- Open problems and research frontiers

---

## About Me

- PhD in E.E. at CSU
- Deep Learning and Weather Radar
- Postdoc: NREL
  - Hybrid deep learning / computer vision pipeline for automatic solar heliostat calibration
- Startups: PlanetiQ
  - Satellite obs. of atmosphere with RO; Signal processing, software engineering, optimization, ML
- Startups: Tellus
  - Life and biorhythm monitoring of older adults with FMCW radar, deep learning, and signal processing
- Consulting: phData
  - Production deployment patterns and infrastructure for millions of ML models for Fortune 500

---

## Part 0: What is the Problem and Why Do We Care?

---

## Our Problem

- Almond plantations need a ton of water, and are typically grown in water-starved areas (CA)
- Water resource management is critically important
- How to know if we are watering "enough" and only "enough"?
- Water scarce and stress are global issues, worsening due to Climate Change

---

## Satellite Observations

- Satellite measurements can help us here by providing estimates of critical properties like
  - How much water is currently in the plants?
  - How much is this amount changing? (time series + evapotranspiration)
  - What does the situation look like for the entire area (field) of interest? 

---

## Satellite Imagery

Remote Sensing gives:
- Revisits and time series measurements
- Efficient measurements (relative to point instruments)
  - Temperature, Moisture in plants, etc
- Publicly funded and freely available options (Sentinel-2)
  - Commercial options and opportunities

---

## Part 1: Agentic AI

---

## A bit of history


---

### 1985: Programming as Theory Building

- Danish computer scientist and Turing Award winner, Peter Naur
- Building off work in philosophy by Ryle (1949)
- "Very briefly, a person who has or possesses a theory in this sense* knows how to do certain things and in addition can support the actual doing with explanations, justifications, and answers to queries, about the activity of concern"

[*] programmers' knowledge as a theory; the knowledge built up through the course of time and effort to solve problems in a specific role/use case

---


### 2017: Transformer Paper

- We can now efficiently train massive LLMs (large language models) to produce credible responses
- Knowledge can be built from text alone
- A "Theory of Mind" in a language model

---


### 2018: Bidirectional encoder representations from transformers (BERT)

- From Google
- Building on transformer work, self-supervised system to represent text
- Masked token prediction and next sentence prediction
- Learns contextual, latent representation of tokens* in their context
- 110M parameters

[*] integers mapped from subwords; represents vocabulary of the model

---


### 2019: Generative pretrained Transformer 2 (GPT-2)

- OpenAI
- general-purpose learner
- tenfold increase from GPT-1; 1.5B parameters
- more data, bigger model, more better


---

### 2020: Generative pretrained Transformer 3 (GPT-3)

- OpenAI
- like GPT-2, decoder-only transformer deep neural network model
- hundredfold increase from GPT-2; 175B parameters
- more data, bigger model, more better


---

### 2021: Github Copilot

- LLM-generated, "smart" autocomplete
- Leveraged GPT-3
- Browser extension in Jetbrains, then VS Code, Neovim, etc
- First coding model that was democratized widely and used broadly
- Not quite "agentic" but a big turning point moment


---

### 2022-23: Chat-based models

- 2022: ChatGPT (OpenAI)
- 2023: Claude (Anthropic)
- 2023: Grok

Each company releasing and upgrading their functionality, but as of 2023 the models were "stuck" in
the chat interfaces...


---

### 2022: ReAct -- Reason + Act

- Yao et al
- Synergizing Reasoning and Acting in Language Models
- Defines a patter where LLMs generate "reasoning traces" and "task-specific actions" in an interleaved manner
- A model can form an action plan, issue tool calls, and monitor responses, then update its plan and understanding iteratively
- Kind of like the powerful "Chain of Thought" reasoning (CoT, also 2022) but with access to "tools"

**Tool**: a function that can be called by an LLM

---

### ReAct Example, modified from the paper

**Question** What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?

**Thought 1** I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.

**Action 1** _Search[Colorado orogeny]_

**Observation 1** The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.

**Thought 2** It does not mention the eastern sector. So I need to look up eastern
sector.

**Action 2** _Lookup[eastern sector]_

**Observation 2** (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.

**Thought 3** The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.

**Action 3** _Search[High Plains]_

**Observation 3** High Plains refers to one of two distinct land regions

**Thought 4** I need to instead search High Plains (United States).

**Action 4** _Search[High Plains (United States)]_

**Observation 4** The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]

**Thought 5** High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.

**Action 5** _Finish[1,800 to 7,000 ft]_

...

---

### 2020-Current: Tooling Developments

- 2020: "RAG" first mentioned
  - Retrieval-augmented generation
  - Adds ability to ground LLM responses in specific and relevant knowledge
  - Explosion of development since, active even today
    - Chunking, storage mechanisms, document parsing, data modalities, search, vector DBs, etc

- 2024: Anthropic introduces model context protocol (MCP)
  - LLM-friendly "API", allows tool-calling and access to external resources
  - Downside -- Authentication not baked in from the start (added later)
  - Downside -- Verbose and token-heavy; context window saturation, LLM response degradation
  - Companies frenetically add 

---

### 2020-Current: Tooling Developments

- Oct 2025: Cloudflare (and others) move away from MCP 
  - "code mode"
  - models are excellent at generating executable code on the fly, just use that
  - Anthropic picked this up and ran with it as well, culminating in...

- Nov 2025: Agent Skills
  - specification for how to define workflows a model can follow
  - replaces MCP for many use cases; now, an LLM executes workflow defined in a Skill, generates code as needed
  - a Skill is a directory, with:
    - SKILL.md -- defines workflow, YAML frontmatter
    - references/ -- sort of local RAG, houses relevant data artifacts
    - scripts/ -- actual code that can be called in the context of a Skill
  - "progressive disclosure

---

### Agentic IDEs

What happens when you take the MCPs, RAGs, Skills, wrap them into a command-line tool that can call
high-quality LLMs, which, using the ReAct patterns, can directly interact with the machines they're
running on to:
- Read code and structures
- Execute code and bash commands
- Find relevant information locally and externally
- Write and run code + tests
?

You get Agentic IDEs.

- Feb 2025 -- Anthropic's Claude Code
- May 2025 -- OpenAI's Codex
- Jun 2025 -- Google's Gemini CLI
- Aug 2025 -- Amazon's Kiro
- Oct 2025 -- open-source Opencode (v1.0.0 release)

---

## Back to Almonds

How can we use these agentic patterns to help us make irrigation decisions about almonds?

Approach:
- Get data from satellite data providers
- Have agent call tools to calculate relevant quantities/variables from satellite data
- RAG documents to give specific details as needed on irrigation, crop properties, etc
- MCP to hit external weather API to inform tool call calculations as needed
- Develop custom "agentic loop" to implement ReAct pattern with above tools

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

- Healthy plants:
  - Reflect Green and absorb Red
  - Reflect NIR

---

## Abbreviations and Descriptions

* NIR = Near Infrared
* SWIR = Short-wavelength infrared
* Chlorophyll => absorbs red & blue light, but reflects green & NIR more (healthy, dense canopy)
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

--> We need to know this VPD, and compute a variable that accounts for the temp/humidity gradients (CWSI)

---

## EVI: Enhanced Vegetation Index

```
EVI = 2.5 × (NIR - Red) / (NIR + C1·Red - C2·Blue + 1)
```

where C1 and C2 are tuneable parameters

- Less saturation in dense canopy
- More sensitive to canopy structure
- Reduces atmospheric and soil background noise

---

## CWSI: Crop Water Stress Index

```
CWSI = [(T_c - T_a) - (T_cl - T_a)] / [(T_cu - T_a) - (T_cl - T_a)]
```

- `T_c` -- measured canopy temp
- `T_a` -- air temp
- `T_cl` -- canopy temp of non-stressed crop (min. `T_c`)
- `T_cu` -- canopt temp of a stressed crop (max `T_c`)

---

## What Crop Stress Looks Like Spectrally

|         Stress type |         NDVI |      CWSI |           EVI |              Other signals |
|---------------------|--------------|-----------|---------------|----------------------------|
|        Water stress |            ↓ | up early  |             ↓ |     Thermal: warmer canopy |
| Nutrient deficiency |            ↓ |   varies  |             ↓ |             Red-edge shift |
|        Pest/disease |            ↓ |   varies  |             ↓ | Rapid change in timeseries |
|             Healthy |         high |      low  | moderate-high |                         -- |

Challenges:
- Crops exhibit different patterns and have different properties, like Kc
- "noise" like haze, clouds, etc can impact satellite measurements
- late season senescence
- pest damage

---

## How can an agent help make irrigation decision?

- Fixed pipeline would be brittle
- Fractal-like complexity of thresholds for crops, regions, seasons, etc to bake in
- "wine terroir"
- An agent can:
  - call different analysis functions on relevant data (or pull more)
  - check current and recent weather
  - verify info on "fractal-like" complexities from a document corpus
  - analyze historical and time series data
  - escalate to human
  - ==> handle uncertainty, ambiguity, and "reason intelligently"

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

Deep learning is excellent for perception; it's less suited for the reasoning layer

---

## Agent vs. Deep Learning: Tradeoffs

- Accuracy (DL)
- Explainability (Agent)
- Data Requirements (Agent)
- Adaptability (Agent)
- Cost per Query (DL)
- Inference Latency (DL)
- Tool Integration (Agent)
- Uncertainty Expression (Agent)

Hybrid architectures often win: DL for perception, agent for reasoning and orchestration.

---

## Live Demo

- Tool Calling
  - JSON Schema
  - Python Functions
- System Prompt
- Agentic Loop
- Visualizations

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
│  compute_ndvi   compute_cwsi   compute_evi │
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

**Repo**: https://github.com/ryangooch/agentic-satellite-data

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
- `02_agent_loop.ipynb` -- the agentic loop, tool-calling
- `03_failure_modes.ipynb` -- how can things go wrong and what we can do about them
- `04_visualizations.ipynb` -- real data plots

See `EXERCISES.md` for 5 take-home problems (add a new tool, modify the prompt, build an evaluator...).
