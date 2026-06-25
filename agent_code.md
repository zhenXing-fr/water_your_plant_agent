# Garden Agent — Complete Technical Specification
### Guide for AI Coding Agents (Copilot, Claude Code, Cursor, etc.)

> **Purpose**: This document is the single source of truth for implementing the garden
> watering agent. Read it fully before writing any code. All decisions are final unless
> a section explicitly says otherwise.

---

## 0. Project goal

Build a **garden watering agent** that takes a weekly weather forecast + garden state
(plant types, growth stages) and outputs a structured watering plan.

This is a **learning project** whose real goal is to practice enterprise-level agent
patterns: hexagonal architecture, tool calling, RAG, LangGraph orchestration,
observability, evaluation, and fine-tuning. Each phase introduces exactly one new
concept on top of a working system.

---

## 1. Final decisions (do not re-litigate these)

| Decision | Choice | Reason |
|----------|--------|--------|
| Architecture | Hexagonal (ports & adapters) | Adapters swap without touching domain |
| Package manager | `uv` | Faster than pip, native pyproject.toml |
| Domain models | `pydantic` v2, `frozen=True` | Free validation + serialisation, immutable |
| Port interfaces | Python `Protocol` (PEP 544) | Structural subtyping, no import coupling |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) | Primary learning target |
| Orchestration | LangGraph (Phase 4+) | Enterprise multi-agent standard |
| Vector DB | ChromaDB (Phase 3+) | Local-first, no infra needed |
| Observability | Langfuse (Phase 5+) | Open-source AgentOps |
| Fine-tuning | HuggingFace PEFT + LoRA (Phase 7) | Standard adapter approach |
| Lint/format | `ruff` | Single tool, fastest |
| CI | GitHub Actions | Simple, free |

---

## 2. Architecture — hexagonal (ports & adapters)

```
┌─────────────────────────────────────────────────────────┐
│  Adapters (infrastructure — only layer with ext imports) │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ OpenWeather  │ │ JSON Garden  │ │  Claude adapter  │ │
│  │  adapter     │ │    repo      │ │  (LLMPort impl)  │ │
│  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘ │
│         │                │                  │           │
│  ┌──────▼────────────────▼──────────────────▼─────────┐ │
│  │         Ports (Python Protocol interfaces)          │ │
│  │   WeatherPort    GardenRepositoryPort    LLMPort    │ │
│  └──────────────────────────┬──────────────────────────┘ │
│                             │                            │
│  ┌──────────────────────────▼──────────────────────────┐ │
│  │            Application service layer                 │ │
│  │              WateringPlannerService                  │ │
│  └──────────────────────────┬──────────────────────────┘ │
│                             │                            │
│  ┌──────────────────────────▼──────────────────────────┐ │
│  │          Domain (pure Python + Pydantic)             │ │
│  │   Plant · Garden · WeatherForecast · WateringPlan   │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Dependency rule (NEVER violate this)

```
adapters → ports → application → domain
```

- `domain/` imports **nothing** except Python builtins and Pydantic
- `ports/` imports **nothing** except `domain/`
- `application/` imports `ports/` and `domain/`
- `adapters/` imports `ports/`, `domain/`, and external libraries
- **No layer imports from a layer to its right in the arrow chain**

---

## 3. Repository layout

```
garden-agent/
├── src/
│   └── garden_agent/
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── models.py          ← ALL domain entities live here (COMPLETE)
│       │   └── services.py        ← Pure watering logic (COMPLETE)
│       ├── ports/
│       │   ├── __init__.py
│       │   ├── weather.py         ← WeatherPort Protocol (COMPLETE)
│       │   ├── garden.py          ← GardenRepositoryPort Protocol (COMPLETE)
│       │   └── llm.py             ← LLMPort Protocol (COMPLETE)
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── weather/
│       │   │   ├── __init__.py
│       │   │   └── openweather.py ← Phase 2
│       │   ├── garden/
│       │   │   ├── __init__.py
│       │   │   └── json_repo.py   ← Phase 2
│       │   └── llm/
│       │       ├── __init__.py
│       │       └── claude.py      ← Phase 2
│       ├── application/
│       │   ├── __init__.py
│       │   └── watering_planner.py ← Phase 1 complete; Phase 2 adds LLM call
│       ├── skills/                 ← Phase 3 (RAG retrieval)
│       └── config.py              ← pydantic-settings (COMPLETE)
├── tests/
│   ├── conftest.py                ← Fixtures + fake adapters (COMPLETE)
│   ├── unit/
│   │   ├── domain/
│   │   │   ├── test_models.py     ← (COMPLETE)
│   │   │   └── test_services.py   ← (COMPLETE)
│   │   └── application/
│   │       └── test_planner.py    ← (COMPLETE)
│   └── integration/               ← Real API calls, skipped in CI
├── data/
│   └── garden.json                ← Sample garden state (COMPLETE)
├── CLAUDE.md                      ← Claude Code project context (COMPLETE)
├── .github/workflows/ci.yml       ← (COMPLETE)
├── .pre-commit-config.yaml        ← ruff on every commit (COMPLETE)
├── .env.example                   ← (COMPLETE)
├── pyproject.toml                 ← uv + ruff + pytest config (COMPLETE)
└── README.md                      ← Phase 8
```

---

## 4. Domain models reference (COMPLETE — do not modify)

```python
# src/garden_agent/domain/models.py

class GrowthStage(str, Enum):      # "seedling" | "established" | "mature"
class SoilType(str, Enum):         # "sandy" | "clay" | "loamy" | "well_draining"

class Plant(BaseModel):            # frozen=True
    name: str                      # unique within garden
    plant_type: str                # "tomato", "succulent", "lavender" …
    growth_stage: GrowthStage
    soil_type: SoilType
    last_watered: date | None
    notes: str

class Garden(BaseModel):           # frozen=True
    id: str
    location: str                  # used by weather adapter for geocoding
    plants: list[Plant]

class WeatherForecast(BaseModel):  # frozen=True
    date: date
    temperature_max_c: float
    temperature_min_c: float
    precipitation_mm: float
    humidity_percent: float
    is_rain_expected: bool

class WateringAction(BaseModel):   # frozen=True
    plant_name: str
    amount_liters: float           # Field(gt=0)
    time_of_day: str               # "morning" | "evening"
    reason: str                    # REQUIRED — agent must justify every action

class DailyPlan(BaseModel):        # frozen=True
    date: date
    actions: list[WateringAction]
    skip_reason: str | None        # set when rain_covers_watering() is True

class WateringPlan(BaseModel):     # frozen=True
    garden_id: str
    week_start: date
    daily_plans: list[DailyPlan]
```

---

## 5. Coding rules (agent must follow all of these)

### Python style
- Python 3.12+, use `from __future__ import annotations` on every file
- Type hints everywhere — no `Any` unless unavoidable (document why)
- `frozen=True` on all Pydantic domain models — never mutate, use `.model_copy(update={})`
- `Protocol` for all port interfaces — never `ABC`
- Enums inherit from `(str, Enum)` so they serialise as strings

### Architecture rules
- `domain/` and `ports/` have **zero** `import anthropic`, `import httpx`, `import chromadb`
- All external library imports belong in `adapters/`
- Application service imports only from `ports/` and `domain/`, never `adapters/`
- If in doubt: can this module be tested without network access? If yes, it belongs in domain or ports. If no, it belongs in adapters.

### Testing rules
- Unit tests use **fake adapters** from `tests/conftest.py` — never `unittest.mock`
- Fake adapters satisfy Port protocols structurally (no inheritance from ports)
- Integration tests (real API calls) go in `tests/integration/` and are skipped in CI via `@pytest.mark.integration`
- Every new module needs at least one test file
- Test naming: `test_<what>_<when>_<expected>`

### Commit hygiene
- Run `uv run pre-commit install` once after clone
- `ruff` runs automatically on `git commit` — fix all warnings before committing
- CI must pass before merging any branch

---

## 6. Commands reference

```bash
# Setup (run once)
uv sync
uv run pre-commit install
cp .env.example .env            # then fill in API keys

# Daily workflow
uv run pytest                   # all tests
uv run pytest tests/unit/ -v    # unit tests only (no network)
uv run pytest -k "test_rain"    # run matching tests
uv run ruff check src/ tests/   # lint
uv run ruff format src/ tests/  # format
uv run mypy src/                # type check

# Add a dependency
uv add <package>                # runtime
uv add --dev <package>          # dev only
```

---

## 7. Phase roadmap

| Phase | Name | Status | Key addition |
|-------|------|--------|-------------|
| 1 | Foundation | ✅ COMPLETE | uv · Pydantic models · Hexagonal arch · CI |
| 2 | Tools & first agent | 🔲 Next | Anthropic SDK · tool calling · real adapters |
| 3 | RAG & skills | 🔲 | ChromaDB · plant knowledge base · retrieval |
| 4 | LangGraph orchestration | 🔲 | Graph nodes · multi-agent · human-in-the-loop |
| 5 | Observability | 🔲 | Langfuse · tracing · cost tracking |
| 6 | Evaluation | 🔲 | Claude-as-judge · regression suite |
| 7 | Fine-tuning | 🔲 | Synthetic data · PEFT · LoRA adapter |
| 8 | Enterprise hardening | 🔲 | Full CI/CD · retry · MCP |

---

## 8. Phase 2 — Tools & First Agent (implement next)

### Goal
Make the first real Claude API call. The agent receives the weather forecast and garden
state as context, calls tools to retrieve them, and produces a `WateringPlan`.

### New dependencies to add
```bash
uv add anthropic httpx
```

### 8.1 OpenWeather adapter

**File**: `src/garden_agent/adapters/weather/openweather.py`

```python
# Implements WeatherPort structurally (no inheritance)
# Uses httpx to call https://api.openweathermap.org/data/2.5/forecast
# API key from settings.openweather_api_key
# Maps raw JSON → list[WeatherForecast] domain objects
# Raises WeatherError (define in this file) on HTTP failure or bad response
# Cache responses for 1 hour using a simple dict {(location, days): (timestamp, result)}
```

OpenWeather endpoint:
```
GET https://api.openweathermap.org/data/2.5/forecast
    ?q={location}&cnt={days}&appid={api_key}&units=metric
```

Response mapping:
```python
# Each item in response["list"] → one WeatherForecast
# date: datetime.fromisoformat(item["dt_txt"]).date()
# temperature_max_c: item["main"]["temp_max"]
# temperature_min_c: item["main"]["temp_min"]
# precipitation_mm: item.get("rain", {}).get("3h", 0.0)
# humidity_percent: item["main"]["humidity"]
# is_rain_expected: any("rain" in w["main"].lower() for w in item["weather"])
```

### 8.2 JSON Garden repository

**File**: `src/garden_agent/adapters/garden/json_repo.py`

```python
# Implements GardenRepositoryPort structurally
# Reads/writes data/garden.json (path from settings.garden_data_path)
# get_garden(): load JSON → Garden.model_validate(data)
# save_garden(): garden.model_dump(mode="json") → write to file
# Raise KeyError if garden_id not found
# Thread-safe: use a file lock or keep it single-process for Phase 2
```

### 8.3 Claude adapter

**File**: `src/garden_agent/adapters/llm/claude.py`

```python
# Implements LLMPort structurally
# Uses anthropic.Anthropic(api_key=settings.anthropic_api_key)
# Model: "claude-sonnet-4-6"
# generate(prompt, tools=None) → str
#   - If tools is None: simple messages call
#   - If tools provided: include tools parameter, handle tool_use response blocks
#   - Extract text from response.content blocks
#   - Raise LLMError on API failure
```

Anthropic SDK pattern:
```python
import anthropic

client = anthropic.Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    system="You are a garden watering expert...",
    messages=[{"role": "user", "content": prompt}],
    tools=tools or [],   # omit key if tools is None/empty
)

# Extract text from response
text = next(
    block.text for block in response.content
    if block.type == "text"
)
```

### 8.4 Tool definitions

Define these as Python dicts in `src/garden_agent/application/tools.py`:

```python
GET_WEATHER_TOOL = {
    "name": "get_weather_forecast",
    "description": "Get the 7-day weather forecast for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Paris, France'"},
            "days": {"type": "integer", "default": 7}
        },
        "required": ["location"]
    }
}

GET_GARDEN_TOOL = {
    "name": "get_garden_state",
    "description": "Retrieve the current garden state including all plants.",
    "input_schema": {
        "type": "object",
        "properties": {
            "garden_id": {"type": "string"}
        },
        "required": ["garden_id"]
    }
}
```

### 8.5 Updated WateringPlannerService

Extend (do not replace) `src/garden_agent/application/watering_planner.py`:

```python
# Add create_plan_with_llm(garden_id, week_start) → WateringPlan
# This method:
#   1. Builds a system prompt explaining the agent's role
#   2. Builds a user prompt with garden_id and week_start
#   3. Calls self._llm.generate(prompt, tools=[GET_WEATHER_TOOL, GET_GARDEN_TOOL])
#   4. Parses tool_use blocks from the response
#   5. Executes the requested tools (get_forecast / get_garden)
#   6. Feeds tool results back to Claude
#   7. Parses Claude's final text response into a WateringPlan
#      (ask Claude to respond in JSON matching WateringPlan schema)
```

System prompt:
```
You are a precision garden watering agent. Given a garden ID and week start date,
you will:
1. Fetch the weather forecast for the garden's location
2. Fetch the current garden state
3. Produce a detailed day-by-day WateringPlan in JSON format

Rules:
- Skip days where precipitation_mm >= 10.0
- Water in the morning when max temperature >= 25°C, otherwise evening
- Sandy soil needs 40% more water than loamy; clay needs 30% less
- Seedlings need 0.3L base; established 0.5L; mature 0.8L
- Every WateringAction MUST include a specific reason explaining why
- Output must be valid JSON matching the WateringPlan schema exactly
```

### 8.6 CLI runner

**File**: `src/garden_agent/main.py`

```python
# Simple CLI: python -m garden_agent.main --garden-id g-001
# Wires: OpenWeatherAdapter + JSONGardenRepository + ClaudeAdapter
# Creates WateringPlannerService
# Calls create_weekly_plan() (pure domain, no LLM) or create_plan_with_llm()
# Prints the plan as formatted JSON
# Use argparse or click
```

### 8.7 Tests for Phase 2

```
tests/unit/adapters/test_claude_adapter.py
    - test with mock httpx / mock anthropic client
    - test tool_use response parsing
    - test error handling

tests/unit/application/test_planner_with_llm.py
    - use FakeWeatherAdapter, FakeGardenRepository, FakeLLMAdapter
    - FakeLLMAdapter returns a hardcoded JSON WateringPlan string
    - verify the planner parses it correctly

tests/integration/test_openweather.py
    @pytest.mark.integration  (skipped in CI)
    - real API call to OpenWeather
    - verify returns list[WeatherForecast]
```

---

## 9. Phase 3 — RAG & Skills (implement after Phase 2)

### Goal
Give the agent deep plant-specific knowledge through retrieval-augmented generation.
Before producing a plan, the agent retrieves relevant care guidelines for each plant.

### New dependencies
```bash
uv add chromadb sentence-transformers
```

### Knowledge base structure

**File**: `src/garden_agent/skills/plant_knowledge.py`

```python
# ChromaDB collection: "plant_care"
# Each document: plant care guideline for a specific plant type
# Metadata: {"plant_type": "tomato", "topic": "watering", "source": "manual"}
# Embedding: sentence-transformers "all-MiniLM-L6-v2" (local, no API cost)
```

**File**: `src/garden_agent/skills/data/plant_care.json`

Populate with ~20 entries covering: tomato, succulent, lavender, rose, basil,
pepper, lettuce, strawberry, mint, oregano. Each entry:
```json
{
  "plant_type": "tomato",
  "topic": "watering",
  "content": "Tomatoes need consistent moisture. Established plants need 1-2 inches
               of water per week. Water deeply 2-3 times per week rather than
               shallowly every day. Sandy soil drains faster and may need daily
               watering in hot weather. Mulch to retain moisture.",
  "drought_tolerance": "low",
  "overwatering_risk": "medium"
}
```

### Retrieval pattern

```python
# src/garden_agent/skills/retriever.py
# query_plant_care(plant_type: str, n_results: int = 3) → list[str]
# Returns the text content of the top-n matching documents
# Used by WateringPlannerService to enrich the LLM prompt
```

### Integration with agent

In `WateringPlannerService.create_plan_with_llm()`:
```python
# Before building the prompt, retrieve care context for each plant:
plant_contexts = {
    plant.name: retriever.query_plant_care(plant.plant_type)
    for plant in garden.plants
}
# Inject into system prompt as a "Plant care reference" section
```

---

## 10. Phase 4 — LangGraph Orchestration

### Goal
Replace the simple tool-call loop with a proper LangGraph state machine.
Introduce human-in-the-loop approval before watering actions are finalised.

### New dependencies
```bash
uv add langgraph langchain-anthropic
```

### Graph design

```
START → fetch_context → plan_generation → human_review → execute_plan → END
                              ↑                  |
                              └── request_revision ┘
```

**Nodes**:
- `fetch_context`: calls WeatherPort + GardenRepositoryPort, stores in state
- `plan_generation`: calls LLM with context + retrieved plant knowledge, produces draft WateringPlan
- `human_review`: `interrupt()` — waits for human approval or revision request
- `execute_plan`: saves finalised plan, triggers notifications (Phase 8)
- `request_revision`: feeds human feedback back to `plan_generation`

**State definition**:
```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    garden_id: str
    week_start: date
    garden: Garden | None
    forecasts: list[WeatherForecast]
    draft_plan: WateringPlan | None
    final_plan: WateringPlan | None
    human_feedback: str | None
    messages: Annotated[list, add_messages]
```

**File structure**:
```
src/garden_agent/graph/
    __init__.py
    state.py        ← AgentState TypedDict
    nodes.py        ← one function per node
    edges.py        ← conditional routing logic
    graph.py        ← assemble and compile the graph
```

---

## 11. Phase 5 — Observability with Langfuse

### Goal
Trace every agent run: which tools were called, how long each step took,
how much it cost, and whether the output was rated good/bad.

### New dependencies
```bash
uv add langfuse
```

### Integration points

```python
# src/garden_agent/observability/tracer.py

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
)

# Wrap every LLM call with a trace
# Wrap every tool call with a span
# Record input tokens, output tokens, latency, model name
# Score each completed plan (later used in Phase 6 eval)
```

Add to `.env.example`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 12. Phase 6 — Evaluation

### Goal
Systematically measure the quality of the agent's outputs.

### Eval harness

**File**: `evals/`
```
evals/
    fixtures/           ← golden input/output pairs
        case_01_hot_dry.json
        case_02_rainy_week.json
        case_03_mixed.json
    test_agent_outputs.py
    judge_prompt.py     ← Claude-as-judge system prompt
    run_evals.py        ← CLI: python -m evals.run_evals
```

### Claude-as-judge pattern

```python
# For each golden case:
#   1. Run the agent on the input
#   2. Send (input, agent_output, expected_output) to a judge Claude call
#   3. Judge returns: {"score": 0-10, "reasoning": "...", "issues": [...]}
#   4. Aggregate scores across all cases
#   5. Fail if mean score < threshold (e.g. 7.0)

JUDGE_SYSTEM_PROMPT = """
You are a senior horticulturalist evaluating AI watering recommendations.
Score the plan from 0-10 on:
- Correctness: Does the plan skip rainy days? Are amounts reasonable?
- Reasoning: Does each WateringAction have a specific, accurate reason?
- Completeness: Are all plants addressed on non-rainy days?

Respond in JSON: {"score": <int>, "reasoning": "<str>", "issues": [<str>]}
"""
```

---

## 13. Phase 7 — Fine-tuning with LoRA

### Goal
Train a small LoRA adapter that specialises in watering plan generation,
reducing LLM cost and latency for production use.

### New dependencies
```bash
uv add datasets peft transformers torch accelerate
```

### Synthetic data generation

**File**: `scripts/generate_training_data.py`

```python
# Use Claude to generate 500+ training pairs:
#   Input: {garden_state, weather_forecast, plant_care_context}
#   Output: {watering_plan_json}
# Vary: plant types, seasons, soil types, growth stages, weather patterns
# Store as JSONL: data/training/watering_plans.jsonl
# Format each pair as a chat template:
#   {"messages": [
#       {"role": "system", "content": "..."},
#       {"role": "user", "content": "<input>"},
#       {"role": "assistant", "content": "<output_json>"}
#   ]}
```

### LoRA training script

**File**: `scripts/train_lora.py`

```python
# Base model: "microsoft/Phi-3.5-mini-instruct" (small, fast, good for fine-tuning)
# LoRA config: r=16, alpha=32, target_modules=["q_proj", "v_proj"]
# Train for 3 epochs, batch_size=4, lr=2e-4
# Save adapter to: models/watering-lora/
```

### Second LLM adapter

**File**: `src/garden_agent/adapters/llm/lora_adapter.py`

```python
# Implements LLMPort structurally
# Loads the trained LoRA adapter with PEFT
# Used as drop-in replacement for ClaudeAdapter
# Test A/B: compare output quality with Claude-as-judge
```

---

## 14. Phase 8 — Enterprise Hardening

### Goal
Production-ready: retries, fallbacks, secrets management, full CI/CD pipeline.

### Retry / fallback

```python
# src/garden_agent/adapters/llm/claude.py additions:
# - Retry with exponential backoff on rate limit (429) and server error (5xx)
# - Max 3 retries, base delay 1s
# - Fallback: if Claude fails after retries, fall back to pure domain logic
#   (create_weekly_plan() without LLM) and log the degradation
```

### Secrets management

```bash
uv add python-dotenv  # already via pydantic-settings
# For production: use GitHub Secrets → ANTHROPIC_API_KEY env var in Actions
# Never commit .env
# Add secret scanning to CI: detect-secrets pre-commit hook
```

### Full CI/CD pipeline

`.github/workflows/ci.yml` additions:
```yaml
# Existing: lint + test on every push
# Add:
#   - mypy type check
#   - eval run on main branch (requires ANTHROPIC_API_KEY secret)
#   - Build and push Docker image on tag
#   - Deploy to cloud on release
```

### MCP server (bonus)

```python
# Expose the watering planner as an MCP server so other AI tools can call it
# src/garden_agent/mcp_server.py
# Tool: "create_watering_plan" with inputs: garden_id, week_start
# Run with: uv run python -m garden_agent.mcp_server
```

---

## 15. Fake adapters reference (for tests)

These already exist in `tests/conftest.py`. Use them in all unit tests.
Do not create new mock objects — extend these instead.

```python
class FakeWeatherAdapter:
    """Satisfies WeatherPort structurally."""
    def __init__(self, forecasts: list[WeatherForecast]) -> None: ...
    def get_forecast(self, location: str, days: int = 7) -> list[WeatherForecast]: ...
    call_count: int  # how many times get_forecast was called

class FakeGardenRepository:
    """Satisfies GardenRepositoryPort structurally."""
    def __init__(self, garden: Garden) -> None: ...
    def get_garden(self, garden_id: str) -> Garden: ...
    def save_garden(self, garden: Garden) -> None: ...
    save_count: int  # how many times save_garden was called

# To add a FakeLLMAdapter for Phase 2 tests, add to conftest.py:
class FakeLLMAdapter:
    """Satisfies LLMPort structurally."""
    def __init__(self, response: str) -> None: ...
    def generate(self, prompt: str, tools=None) -> str: ...
    call_count: int
```

---

## 16. Environment variables

Full `.env` for all phases:

```bash
# Phase 2 — required
ANTHROPIC_API_KEY=sk-ant-...
OPENWEATHER_API_KEY=...
GARDEN_DATA_PATH=data/garden.json

# Phase 5 — Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Phase 7 — fine-tuning (optional, local)
LORA_MODEL_PATH=models/watering-lora/
BASE_MODEL_NAME=microsoft/Phi-3.5-mini-instruct
```

---

## 17. What NOT to do

- ❌ Do not use `unittest.mock` — use fake adapters from conftest.py
- ❌ Do not import `anthropic` or `httpx` in `domain/` or `ports/`
- ❌ Do not mutate Pydantic models — use `.model_copy(update={})`
- ❌ Do not hardcode API keys — use `settings` from `config.py`
- ❌ Do not use `ABC` for ports — use `Protocol`
- ❌ Do not skip the `reason` field on `WateringAction` — it is required
- ❌ Do not add more than one concept per phase — keep phases isolated
- ❌ Do not run integration tests in CI — mark with `@pytest.mark.integration`
- ❌ Do not use `Any` type without a comment explaining why
- ❌ Do not create files outside the structure defined in Section 3

---

## 18. Quick-start checklist for new phases

Before starting any phase:

- [ ] Read the relevant section (8–14) in this document fully
- [ ] Run `uv run pytest` — all existing tests must pass before you add new code
- [ ] Add new dependencies with `uv add <package>`
- [ ] Create test file(s) before or alongside the implementation
- [ ] Run `uv run ruff check src/ tests/` — fix all warnings
- [ ] Run `uv run pytest` again — new + existing tests must pass
- [ ] Update `CLAUDE.md` phase status when phase is complete