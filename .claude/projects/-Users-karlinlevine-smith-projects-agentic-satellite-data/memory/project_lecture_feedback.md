---
name: Post-Lecture Feedback (2026-03-12)
description: What worked and didn't in first lecture delivery — guides material improvements
type: project
---

**What worked well:**
- Agentic flow concept landed with students
- Deep dive into crop health/spectral indices was engaging

**What fell flat:**
- Code tour (notebook 01 tools walkthrough) — students flagged
- Deeper specifics of crop classification lost them
- Synthetic data was unconvincing — students didn't connect with it
- Agent loop in Jupyter didn't have "punch" — was "mildly interesting" instead of "eureka"

**Decided improvement plan (agreed 2026-03-12):**
1. Real Sentinel-2 data — California Central Valley, multi-temporal, with USDA CropScape ground truth
2. Local RAG — county ag reports, extension service bulletins for the area
3. Move live demo out of Jupyter — run in Claude Code CLI for streaming impact
4. Agent Skill — wrap spectral tools + workflow as a Claude Code Agent Skill (use skill builder). This replaces the raw tool-calling approach and clarifies the modern pattern
5. MCP server for weather — real historical + forecast data (e.g., Open-Meteo), shows MCP concept alongside the skill approach. Agent uses this to check natural irrigation / correlate stress with weather
6. Future: integrate user's custom deep learning model for transformer-based crop classification into the agentic pipeline

**Why:** The goal is to make the demo undeniably impressive so students viscerally understand why agentic approaches matter. The skill + MCP split also teaches two modern patterns in one demo.
**How to apply:** Prioritize changes that increase demo impact and real-world feel. Reduce time on code walkthroughs that lose engagement. The DL model integration is a future phase — don't scope it into current work.
