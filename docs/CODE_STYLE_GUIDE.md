# DSL Code Style Guide

## Table of Contents

1. [Documentation Language](#1-documentation-language)
2. [Naming Conventions](#2-naming-conventions)
3. [Function Structure](#3-function-structure)
   - 3.1 [Docstrings](#31-docstrings)
   - 3.2 [Function Length](#32-function-length)
   - 3.3 [Step-by-Step Comments](#33-step-by-step-comments)
4. [Comments](#4-comments)
5. [Comprehensions & Generator Expressions](#5-comprehensions--generator-expressions)
6. [Exceptions](#6-exceptions)
7. [Imports](#7-imports)

---

## 1. Documentation Language

Write docstrings and comments with clarity and precision. Assume the reader has programming knowledge but may not be familiar with your implementation details. Avoid unnecessary jargon while using technical terms correctly.

### ✅ DO: Use simple, direct sentences

```python
def _generate_all_variables(config, location, shared_cache) -> dict:
    """Generate time series for each variable at this location.
    
    Instantiate the generator specified in the config for each variable and
    create a time series. Cache shared (regional) variables to ensure
    consistency across locations.
    
    Args:
        config: The scenario configuration.
        location: The location identifier (e.g., "north", "south").
        shared_cache: Dict caching location-independent variables.
    
    Returns:
        Dict mapping variable names to time series arrays:
        {"rainfall": [50.5, 59.3, ...], "humidity": [65.2, 68.1, ...]}
    """
    ...
```

### ✅ DO: Use active voice

```python
# ✅ GOOD
"""Generate all variables, then build the disease signal from them."""

# BAD: Passive, less direct
"""All variables are generated, and the disease signal is built from them."""
```

### ✅ DO: Be concrete—show structure and values

```python
# ✅ GOOD: Shows what the data looks like
# Now drivers = {"rainfall": [50.5, 59.3, ...], "humidity": [65.2, 68.1, ...]}

# BAD: Abstract
# Now drivers contains the generated variables
```

### ✅ DO: Explain the purpose, not just the mechanism

```python
# ✅ GOOD: Why and what
"""Apply a lag of 2 months to rainfall to model the delayed effect on disease.
The value at month 5 becomes the value at month 3, with early periods as NaN."""

# BAD: Just the what
"""Shift the rainfall array back by 2 periods."""
```

### ✅ DO: List steps explicitly for multi-step processes

```python
"""Build the disease incidence signal in 7 steps:
1. Start with the baseline rate (from config)
2. Add weighted, lagged effects of drivers (rainfall, etc.)
3. Apply transformations (e.g., missing data masking)
4. Squash through sigmoid to constrain to [0, 1]
5. Scale by population
6. Draw Poisson-distributed case counts
7. Mark first 2 periods as NaN (lag warm-up)
"""
```

### ❌ DON'T: Use jargon without context

```python
# BAD: Assumes epidemiology knowledge
"""Compute incidence using a GLM with Poisson link."""

# ✅ GOOD: Explains the choice
"""Compute disease counts using a Poisson model, which is standard for
count data because it naturally handles the discrete nature of cases."""
```

### ❌ DON'T: Hide edge cases

```python
# BAD: Vague about behavior
"""Resolve the population."""

# ✅ GOOD: Clear about what happens
"""Resolve population for the location. If the config specifies a fixed value,
return a constant array. If it specifies a PopulationSpec, generate a time
series. Raise ValueError if any value is non-finite."""
```

### ❌ DON'T: Sacrifice precision for brevity

```python
# BAD: Loses important detail
"""Generate rainfall."""

# ✅ GOOD: Precise
"""Generate a monthly rainfall time series using the seasonal_smooth generator
with mean and amplitude from the config."""
```

### Summary
- Use direct, active sentences
- Assume intermediate programming knowledge
- Show concrete data structures and values in comments
- Explain WHY and WHAT, not just HOW
- List steps for complex processes
- Use technical terms correctly; explain non-standard choices
- Make edge cases and constraints explicit

---

## 2. Naming Conventions

Use consistent, descriptive names. Reserve single letters for loop indices and type variables only.

| Type | Convention | Example |
|------|-----------|---------|
| Modules | `lower_with_under` | `disease.py`, `engine.py` |
| Classes | `CapWords` | `ScenarioConfig`, `VariableSpec` |
| Exceptions | `CapWords` + "Error" | `ValueError`, `FileNotFoundError` |
| Functions/Methods | `lower_with_under()` | `_run_one_location()`, `build_disease_cases()` |
| Constants | `CAPS_WITH_UNDER` | `DEFAULT_SEED`, `MAX_PERIODS` |
| Variables | `lower_with_under` | `var_name`, `generated_data`, `shared_cache` |
| Internal (private) | prepend `_` | `_child_rng()`, `_private_var` |

### ✅ DO: Use descriptive names

```python
var_name = spec.name  # Clear what it is
generated_data = _generate_variable(...)  # Clear what it contains
drivers = {}  # Clear what this dict holds
disease = build_disease_cases(...)  # Clear what it returns
```

### ❌ DON'T: Use single letters (except in loops/indices)

```python
# BAD: d, s, g are meaningless
d = {}
for s in config.variables:
    g = _generate_variable(...)
    d[s.name] = g
```

### ✅ OKAY: Single letters in loops and for type variables

```python
# OK: standard loop patterns
for i in range(config.n_total):  # i is a standard index
    ...

for spec in config.variables:  # spec is meaningful in context
    ...

from typing import TypeVar
T = TypeVar('T')  # Standard type variable naming
```

### ❌ DON'T: Redundant type indicators in names

```python
# BAD: Name already says it's a dict; the _dict is redundant
id_to_name_dict = {}

# ✅ GOOD: Name is clear without the type suffix
id_to_name = {}
```

### ✅ DO: Use meaningful names for booleans

```python
is_valid = True  # Clear it's a boolean
has_error = False
should_continue = True
```

### ❌ DON'T: Use obscure abbreviations

```python
# BAD: What does cfg mean? What is spc?
cfg = config
spc = spec
dgen = _generate_variable(...)

# ✅ GOOD: Full names or standard abbreviations
config = config
spec = spec
generated = _generate_variable(...)
```

---

## 3. File Organization

Organize code within a file in this order: docstring → imports → constants → public API → helpers → classes. Prioritize readability: readers should see what the file does (public API) before implementation details.

### File Structure Order

```python
"""Module docstring: explain what this file does."""

# Standard library
import math
from pathlib import Path

# Third-party
import numpy as np

# Local
from dsl.core.config.schema import ScenarioConfig

# Constants and configuration
DEFAULT_SEED = 0
MAX_PERIODS = 1000

# PUBLIC API functions (entry points—what callers use)
def run(config: ScenarioConfig) -> pd.DataFrame:
    """Run the whole simulation.
    
    This is what users call. Implementation details are below.
    """
    shared_cache = {}
    for location in config.locations:
        df = _run_one_location(config, location, shared_cache)
    return df

# HELPER functions (implementation details)
def _run_one_location(config, location, shared_cache) -> pd.DataFrame:
    """Generate data for one location (internal)."""
    drivers = _generate_all_variables(config, location, shared_cache)
    return assemble_dataframe(drivers)

def _generate_all_variables(config, location, shared_cache) -> dict:
    """Generate all variables (internal)."""
    ...

# DATA MODELS (classes, if needed)
class Config(BaseModel):
    """Configuration schema."""
    ...
```

### Why This Order?

- **Public API first** — A reader sees what the module does immediately
- **Helpers after** — Implementation details come after the high-level view
- **Classes at the end** — Data models are usually referenced by functions; define functions first

### Ordering Helpers by Call Depth

Within the helpers block, order functions by call depth, following the actual
call chain. A function is placed immediately after the function that calls it,
not grouped by category or alphabetically. Leaf helpers (called by multiple
functions, calling nothing else in the module) sink to the bottom.

**Why?** Reading the file top-to-bottom traces the execution flow. You see what
the public API does, then dive into `_run_one_location()` (which it calls),
then into `_generate_variable()` (which it calls), etc. Each step shows what
the next layer does before revealing its own internal machinery. This matches
how a reader naturally traces the code.

#### ✅ GOOD: Ordered by call depth (top-to-bottom reads as execution flow)

```python
# PUBLIC API
def run(config):
    """Entry point."""
    shared_cache = {}
    for location in config.locations:
        df = _run_one_location(config, location, shared_cache)
    return df

# HELPERS (ordered by call chain)
def _run_one_location(config, location, shared_cache):
    """First helper called by run()."""
    drivers = _generate_variable(config, spec, location, shared_cache)
    population = _resolve_population(...)
    return assemble_dataframe(...)

def _generate_variable(config, spec, location, shared_cache):
    """Second helper, called by _run_one_location()."""
    params = dict(spec.params)
    generator = _build_generator(spec.generate, params, spec.name)
    own = generator.generate(config.n_total, config.period, var_rng)
    return own

# LEAF HELPERS (called by multiple functions, no internal calls)
def _child_rng(seed, *keys):
    """Create a seeded RNG (no calls to other helpers in module)."""
    entropy = [int(seed) & 0xFFFFFFFF]
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:4]
        entropy.append(int.from_bytes(digest, "big"))
    return np.random.default_rng(np.random.SeedSequence(entropy))

def _build_generator(name, params, variable=None):
    """Instantiate a generator (leaf helper)."""
    try:
        return get_generator(name)(**params)
    except TypeError as exc:
        where = f"variable '{variable}'" if variable else "population"
        raise ValueError(f"{where}: generator '{name}' got an invalid param ({exc}).") from exc
```

#### ❌ BAD: Helpers grouped by role or alphabetically (breaks call chain)

```python
# PUBLIC API
def run(config):
    """Entry point."""
    shared_cache = {}
    for location in config.locations:
        df = _run_one_location(config, location, shared_cache)  # defined 50 lines later!
    return df

# LEAF HELPERS (defined first for some reason)
def _build_generator(name, params, variable=None):
    """Instantiate a generator."""
    ...

def _child_rng(seed, *keys):
    """Create a seeded RNG."""
    ...

# Actual implementation (reader has to jump back and forth)
def _run_one_location(config, location, shared_cache):
    """Helper that _run() calls."""
    ...
```

Reader must jump between sections repeatedly; the file doesn't tell a story.

### Exception 1: Data-Driven Modules

If the file is primarily about data models (e.g., schema.py), put classes near the top:

```python
"""Validation schemas."""

# Classes (the main point of this file)
class ScenarioConfig(BaseModel):
    ...

class VariableSpec(BaseModel):
    ...

# Public functions (use the classes)
def parse_config(data: dict) -> ScenarioConfig:
    ...

# Helpers
def _validate_cross_fields(...):
    ...
```

### Exception 2: CLI Entry Point (`main()`)

The `main()` function goes **at the bottom**, right before the `if __name__ == "__main__"` guard. This is Python convention:

```python
"""CLI for generating datasets."""

import sys

STARTER = "..."

def run_scenario(path: str) -> int:
    """Run a scenario (helper)."""
    ...

def main(argv=None) -> int:
    """Entry point (always at bottom)."""
    ...

if __name__ == "__main__":
    sys.exit(main())
```

**Why:** Readers of a CLI module expect `main()` to be at the bottom. It's the conventional Python pattern.

---

## 3. Function Structure

### 3.1 Docstrings

All non-trivial functions must have docstrings. One-line functions can have one-line docstrings.

#### ✅ DO: Complete docstrings with Args, Returns, Raises

```python
def _run_one_location(
    config: ScenarioConfig,
    location: str,
    shared_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Generate the full time series for a single location.
    
    Generate all variables (rainfall, humidity, etc.) for this location,
    resolve the population, build the disease incidence signal, and
    assemble everything into a DataFrame. The disease signal uses a 7-step
    model: baseline → drivers (weighted, lagged) → sigmoid squash → Poisson draw.
    
    Args:
        config: The validated scenario configuration.
        location: The location identifier (e.g., "north", "south").
        shared_cache: Dict caching shared (regional) variables across locations.
    
    Returns:
        A DataFrame with columns: time_period, location, all variables,
        disease_cases, population. One row per time period.
    
    Raises:
        ValueError: If disease signal computation fails or population is invalid.
    """
    ...
```

#### ✅ OKAY: One-line docstrings (rare, only for self-contained functions)

One-liners are acceptable only if the function is small and self-contained:

```python
def _child_rng(seed: int, *keys: str) -> np.random.Generator:
    """Create a seeded RNG from a seed and component keys."""
    ...
```

**Most functions need Args/Returns/Raises.** One-liners are the rare exception.

#### ❌ DON'T: Vague docstrings

```python
def _run_one_location(config, location, shared_cache):
    """Generate the full series for a single named location."""
    # What does "full series" mean? What columns come back?
    ...
```

#### ❌ DON'T: Docstrings that just repeat the code

```python
def _run_one_location(config, location, shared_cache):
    """Generates all variables, resolves population, builds disease, returns DataFrame."""
    # This is obvious from reading the code.
    ...
```

#### ❌ DON'T: Missing Args/Returns/Raises

```python
def _run_one_location(config, location, shared_cache):
    """Generate the full time series for a single location."""
    # Where are the Args, Returns, Raises? Caller must read the code to understand.
    ...
```

#### ✅ DO: Document errors that can occur (even if caught)

For functions that catch and handle errors internally, document what errors *can occur* as advisory information:

```python
def run_scenario(scenario_path: str, out_dir: str | None, plot: bool, plot_fmt: str) -> int:
    """Run a scenario from YAML and write output files.
    
    Execute the full pipeline: load YAML → parse config → generate data →
    write CSV, metadata, and optional plot.
    
    Args:
        scenario_path: Path to the YAML scenario file.
        out_dir: Output directory (defaults to out/<scenario_name>).
        plot: Whether to generate a plot.
        plot_fmt: Plot format (e.g., "html").
    
    Returns:
        Exit code: 0 if all steps succeed, 1 if any step fails.
    
    Errors Caught (logged to stderr):
        FileNotFoundError: If the scenario file doesn't exist.
        ValueError: If the scenario config is invalid.
        ValidationError: If Pydantic validation fails.
        KeyError: If required keys are missing during generation.
    """
    ...
```

**Why:** Even though these exceptions don't escape, callers need to know what can go wrong and why the function returns 1.

**Do not repeat:** Do not say "All errors are caught" in the docstring summary if you list them in the "Errors Caught" section—it is redundant. The section already makes clear that errors are handled.

#### ✅ DO: Include concrete examples in Returns (for complex data structures)

For functions returning arrays, dicts, or DataFrames with non-obvious structure, include a concrete example in the Returns section:

```python
def _generate_variable(...) -> np.ndarray:
    """Generate one variable's time series for one location.
    
    ...
    
    Returns:
        A numpy array of n_total float values (one per time period from YAML n_total).
        Example: array([45.2, 54.1, 63.8, 72.5, 78.3, 81.1, 79.5, 73.6, 64.2, 51.3, 36.7, 27.4, ...])
        with length 36 if n_total=36 in the scenario YAML.
    """
```

**When:** Use examples for arrays, dicts, DataFrames, or anything where shape/content/NaN placement matters. Skip for simple types like `int`, `str`, `bool`, `list[str]`.

**Why:** Readers understand concrete output better than abstract descriptions. Shows data type, shape, and realistic values in one glance.

#### ❌ DON'T: Skip examples for complex returns

```python
# BAD: abstract, unclear what the data looks like
def build_disease_cases(...) -> np.ndarray:
    """Build the disease signal.
    
    Returns:
        A float array of case counts.
    """
    # What does it look like? How many values? Are there NaN?
```

#### ✅ OKAY: No example needed for simple returns

```python
# OK: type is self-explanatory
def is_valid(config) -> bool:
    """Check if config is valid.
    
    Returns:
        True if valid, False otherwise.
    """
    # bool is clear; no example needed
```

### 3.2 Function Length

Keep functions under ~60 lines. Break up longer functions for readability and maintainability.

#### ✅ DO: Keep functions focused and short

```python
def _run_one_location(
    config: ScenarioConfig,
    location: str,
    shared_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Generate the full time series for a single location.
    
    Args:
        config: The validated scenario configuration.
        location: The name of the location.
        shared_cache: Cache for shared variables across locations.
    
    Returns:
        A DataFrame with time_period, location, variables, disease_cases, population.
    """
    # Step 1: Generate all variables
    drivers = {}
    for spec in config.variables:
        var_name = spec.name
        generated_data = _generate_variable(config, spec, location, shared_cache)
        drivers[var_name] = generated_data
    # Now drivers = {"rainfall": [50.5, 59.3, ...], "humidity": [65.2, 68.1, ...]}

    # Step 2: Resolve population
    population = _resolve_population(
        config.population_for(location),
        config.n_total,
        config.period,
        _child_rng(config.seed, location, "population"),
        config.start_period,
    )
    disease_spec = config.disease_cases.model_copy(update={"population": population})
    # Now population = [100000, 100000, 100000, ...] (one value per time period)

    # Step 3: Build disease signal using the drivers
    disease = build_disease_cases(
        drivers,
        disease_spec,
        _child_rng(config.seed, location, "disease"),
        config.n_total,
        config.period,
    )
    # Now disease = [nan, nan, 5.0, 8.0, 12.0, ...] (Poisson draws, first 2 are warm-up)

    # Step 4: Assemble into DataFrame
    if config.start_period is not None:
        start_year, offset = parse_period(config.start_period, config.period)
    else:
        start_year, offset = 2000, 0

    columns: dict[str, object] = {
        "time_period": [
            format_period(i + offset, config.period, start_year)
            for i in range(config.n_total)
        ],
        "location": location,
    }
    columns.update(drivers)
    columns["disease_cases"] = disease
    columns["population"] = population

    return pd.DataFrame(columns)
    # Returns DataFrame with columns: 
    # [time_period, location, rainfall, humidity, disease_cases, population]
    # with 36 rows (one per month)

# Total: ~60 lines. Each step is clear and focused.
```

**Benefits:**
- Function reads like a story with clear steps
- Each step handles one concern
- Reader understands the flow without reading inside each step

#### ❌ DON'T: Write monolithic 200+ line functions

```python
# BAD: All logic in one place, no structure
def do_everything_at_once(config, location, shared_cache):
    # 200 lines of tangled logic
    # No clear separation of concerns
    # Hard to understand, debug, or test
    ...
```

#### ✅ DO: Extract helper functions when a step gets complex

If a step in your function grows beyond a few lines, extract it to a helper:

```python
def _run_one_location(config, location, shared_cache) -> pd.DataFrame:
    # Step 1: Generate all variables
    drivers = _generate_all_variables(config, location, shared_cache)
    
    # Step 2: Resolve population
    population = _resolve_population_for_location(config, location)
    
    # Step 3: Build disease signal
    disease = _build_disease_signal(config, location, drivers)
    
    # Step 4: Assemble into DataFrame
    return _assemble_dataframe(config, location, drivers, disease, population)
```

**Benefits:**
- Main function reads like pseudocode
- Complex logic lives in focused helper functions
- Each helper can be tested independently

### 3.3 Step-by-Step Comments

Each step should include a comment showing **what the data structure looks like after that step completes**. This makes it crystal clear what each step produces.

```python
# Step 1: Build a dict mapping each item to its processed value
result = {}

for item in collection:
    key = item.name
    value = process(item)
    result[key] = value
# Now result = {"item1": processed_val, "item2": processed_val, ...}

# Step 2: Resolve population
population = _resolve_population(...)
# Now population = [100000, 100000, 100000, ...] (one value per time period)

# Step 3: Build disease signal
disease = build_disease_cases(...)
# Now disease = [nan, nan, 5.0, 8.0, 12.0, ...] (Poisson draws, first 2 are warm-up)

# Step 4: Assemble into DataFrame
columns = {...}
return pd.DataFrame(columns)
# Returns DataFrame with columns: [time_period, location, rainfall, humidity, disease_cases, population]
```

**Why this works:**
- Each step shows input → processing → output
- Reader sees concrete data structures (not just "generate variables")
- Lag warm-up, data types, and array shapes are all visible
- Makes debugging easier (you know what should be where)

### When to Skip the "Now X = ..." Trace Comment

A `# Now x = ...` comment earns its place only when it shows a concrete
shape or value the reader couldn't already get from the line above it and
the function's own `Returns:` docstring. If it just restates one of those
in English — no numbers, no example, no new fact — cut it.

This applies doubly to a trace comment placed **after** a `return`
statement: it's dead code (unreachable), and if it repeats the `Returns:`
section instead of adding to it, it's pure duplication with a spot for the
two to drift out of sync. Put the concrete example in the docstring's
`Returns:` section instead — that's the one place a reader will actually
look for it.

```python
# ❌ DON'T: restates the Returns: docstring after an unreachable return
def build_disease_cases(...) -> np.ndarray:
    """...
    Returns:
        A float array of length n_periods with Poisson-drawn case counts.
    """
    ...
    return counts
    # Returns a float array of length n_periods; NaN marks warm-up and
    # missing periods, everything else is a non-negative case count.

# ✅ DO: the mid-function trace comments stay (they show real intermediate
# state); nothing repeats the docstring after the return.
def _run_one_location(...) -> pd.DataFrame:
    drivers = {}
    for spec in config.variables:
        drivers[spec.name] = _generate_variable(...)
    # Now drivers = {"rainfall": [50.5, 59.3, ...], "humidity": [65.2, ...]}
    ...
    return pd.DataFrame(columns)
```

---

## 4. Comments

Explain **WHY**, not **WHAT**. Do not describe the code—assume the reader knows Python.

### ✅ DO: Step-by-step block comments (the story)

Break complex functions into numbered steps. Each step explains the next action in the pipeline:

```python
def _run_one_location(config, location, shared_cache):
    # Step 1: Generate all variables
    drivers = {}
    for spec in config.variables:
        var_name = spec.name
        generated_data = _generate_variable(config, spec, location, shared_cache)
        drivers[var_name] = generated_data

    # Step 2: Resolve population (its override or the default)
    population = _resolve_population(...)

    # Step 3: Build disease signal using the drivers
    disease = build_disease_cases(...)

    # Step 4: Assemble into DataFrame
    columns = {...}
    return pd.DataFrame(columns)
```

**Why this works:** Each step tells a clear story. A reader understands the big picture immediately without reading inside each step.

### ✅ DO: Inline comments explaining WHY

```python
# We need to handle this edge case because the database returns
# partial rows when a concurrent update is in flight. Retry once.
retry_count = 0
```

### ❌ DON'T: Comments that just state the code

```python
# BAD: This explains what the code does, but the code already says it
var_name = spec.name  # Get the variable name
generated_data = _generate_variable(...)  # Generate the data
drivers[var_name] = generated_data  # Store it in the dict
```

### ❌ DON'T: Over-commenting obvious code

```python
# BAD: Too many comments on simple statements
x = x + 1  # Increment x
y = y * 2  # Double y
```

### ✅ BETTER: Comments only where logic is non-obvious

```python
# Only comment when the logic is subtle or the intent is unclear
x = x + 1  # Compensate for 1-indexed API (not obvious why we add 1)
```

### Block Comment Length: 3-Line Cap

A comment block explaining WHY should fit in 1-3 lines: one line for
something that just needs a label, two or three lines for a real "why"
that isn't obvious from the code.

If an explanation needs more than 3 lines, that's a signal, not a license to
keep writing. Either the comment is restating what the code below it already
shows (cut the restatement), or the logic itself deserves a named helper
function with its own docstring (the docstring is where a longer explanation
belongs — Args/Returns/Errors already give it room).

#### ❌ DON'T: A 7-line comment justifying one `if` branch

```python
# `generator` above is bound to THIS location's own
# source_location (auto-matched per-location when the variable
# set none) — reusing it would make the "shared" component just
# this location's own data again. A shared series needs its OWN
# location-independent source: use the variable's explicit
# source_location if it set one, else this is ambiguous on a
# multi-location file.
if "source_location" not in dict(spec.params):
    ...
```

#### ✅ DO: The same WHY, trimmed to what the code below doesn't already show

```python
# `generator` is bound to THIS location's own source_location;
# reusing it would make "shared" just this location's data again,
# so the shared series needs its own explicit source_location.
if "source_location" not in dict(spec.params):
    ...
```

---

## 5. Comprehensions & Generator Expressions

Use comprehensions for simple cases only. **Prefer explicit loops with step comments** for most code in this DSL.

### ✅ PREFERRED: Explicit loops with comments

```python
# Step 1: Build a dict mapping each item to its processed value
result = {}

for item in collection:
    key = item.name
    value = process(item)
    result[key] = value
# Now result = {"item1": processed_val, "item2": processed_val, ...}
```

### ✅ OKAY: Simple one-line comprehensions (if super readable)

```python
# Simple transformations—readable at a glance
squared = [x ** 2 for x in numbers]

# Simple filters
evens = [x for x in numbers if x % 2 == 0]

# Simple set operations
unique_names = {user.name for user in users if user is not None}
```

### ❌ DON'T: Complex or multi-line comprehensions

```python
# Not readable—expand to a loop
result = {item.name: process(item) for item in collection}

# Multiple for clauses—always forbidden
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

# Multiple lines—expand to a loop instead
result = [
    transform({'key': key, 'value': value}, color='black')
    for key, value in generate_iterable(some_input)
    if complicated_condition_is_met(key, value)
]
```

**Rule of thumb:** If you hesitate reading it, expand it to a loop with comments.

---

## 6. Exceptions

Raise exceptions for error conditions using built-in exceptions. Make error messages clear and actionable.

### ✅ DO: Raise specific built-in exceptions with clear messages

```python
def _resolve_population(source, n_periods, period, rng, start_period=None) -> np.ndarray:
    """Convert a population source to a length-n_periods integer array.
    
    If source is an int, return a constant array. If source is a PopulationSpec,
    generate a time series and round to non-negative integers.
    
    Raises:
        ValueError: If generated population contains NaN or Inf values.
    """
    if not np.all(np.isfinite(series)):
        raise ValueError(
            f"population generator '{source.generate}' produced non-finite values; "
            f"population must be finite at every period."
        )
    return np.maximum(np.round(series), 0).astype(int)
```

**Common exceptions to use:**
- `ValueError` — precondition violation, invalid arguments
- `FileNotFoundError` — file doesn't exist
- `KeyError` — key not found in dict
- `TypeError` — wrong type passed
- `RuntimeError` — something went wrong at runtime

### ✅ DO: Catch specific exceptions only

```python
try:
    return get_generator(name)(**params)
except TypeError as exc:
    where = f"variable '{variable}'" if variable else "population"
    raise ValueError(
        f"{where}: generator '{name}' got an invalid param ({exc})."
    ) from exc
```

### ✅ DO: Make error messages clear and actionable

```python
# ✅ GOOD: Says exactly what went wrong and where
raise ValueError(
    f"variable '{variable}': generator '{name}' got an invalid param ({exc})."
)

# ❌ BAD: Cryptic message
raise ValueError("Invalid input")
```

### ❌ DON'T: Use bare `except:` without re-raising

```python
# BAD: Silently swallows all errors including KeyboardInterrupt
try:
    result = process_data()
except:
    return None
```

### ❌ DON'T: Catch Exception or StandardError without a reason

```python
# BAD: Too broad, masks real errors
try:
    result = process_data()
except Exception:
    return None

# ✅ GOOD: Catch only what you expect
try:
    result = process_data()
except ValueError:
    return None
```

### ✅ DO: Use `from exc` to preserve the exception chain

```python
try:
    generator = get_generator(name)(**params)
except TypeError as exc:
    raise ValueError(f"Invalid generator '{name}'") from exc
```

Preserve the exception chain so the original error is available for debugging.

---

## 8. Instance Methods in Classes

When a class has methods (functions that operate on `self`), document them the same way as standalone functions: complete Args, Returns, and Errors Caught sections.

### ✅ DO: Document instance methods with complete docstrings

```python
class ScenarioConfig(BaseModel):
    """Validated scenario configuration."""
    
    period: str
    n_total: int
    variables: list[VariableSpec]

    def population_for(self, location: str) -> int | PopulationSpec:
        """Resolve the population for a location.

        Returns the population source (either a fixed int or a PopulationSpec
        generator) for the given location.

        Args:
            location: The location name.

        Returns:
            The population (int or PopulationSpec) from disease_cases.population.
        """
        return self.disease_cases.population
```

**Why:** Methods are public API just like functions. Readers need to know what they do, what they need (Args), and what they return.

### ❌ DON'T: Assume readers understand what `self` means without explanation

```python
# BAD: No explanation of what the method does or returns
class Config:
    def validate(self):
        """Validate the config."""  # Too vague
        ...
```

---

## 9. Early Returns

Use early returns to exit functions immediately when a simple condition is true. Add an inline comment explaining why you exit early.

### ✅ DO: Early return with inline comment

```python
def _resolve_population(source, n_periods, ...):
    """Resolve a population source to an integer array."""
    # Early return if source is a fixed population (not a generator spec).
    if isinstance(source, int):
        return np.full(n_periods, source, dtype=int)
    
    # Continue with more complex generator logic...
    params = dict(source.params)
    series = _build_generator(source.generate, params).generate(...)
    return np.maximum(np.round(series), 0).astype(int)
```

**Why:** Early returns make the simple case obvious and tell readers "if this condition, we're done." No need to nest the rest of the logic.

### ❌ DON'T: Silent early return without explanation

```python
# BAD: Reader has to guess why this returns early
if isinstance(source, int):
    return np.full(n_periods, source, dtype=int)
# Rest of function continues below...
```

---

## 11. Decorated Functions

For functions decorated with `@register_generator`, `@register_transform`, or similar registry decorators, place the docstring right after the decorator, before the function signature:

### ✅ DO: Docstring after decorator

```python
@register_generator("seasonal_smooth")
def seasonal_smooth_gen(
    n_periods: int, 
    period: str, 
    rng: np.random.Generator,
    mean: float = 50,
    amplitude: float = 20,
) -> np.ndarray:
    """Generate a seasonal time series with smooth sine waves.
    
    Produces a repeating sine wave pattern, one full cycle per year.
    Registered as "seasonal_smooth" in the generator registry.
    
    Args:
        n_periods: Number of time periods.
        period: Period type (e.g., "monthly", "daily").
        rng: Seeded random generator for reproducibility.
        mean: Center value of the oscillation (default 50).
        amplitude: Height of the sine wave from center (default 20).
    
    Returns:
        A numpy array of length n_periods with seasonal pattern.
        Example: array([50.0, 59.3, 68.2, 75.1, 80.2, 82.5, ...])
    """
    ppy = periods_per_year(period)
    t = np.arange(n_periods)
    return mean + amplitude * np.sin(2 * np.pi * (t % ppy) / ppy)
```

**Why:** The decorator is part of the public API. Readers see the decorator first, then expect documentation immediately after (before the function name).

### ✅ DO: Mention the decorator in the docstring

Briefly note what the decorator registers the function as:

```python
@register_transform("lag")
def lag_transform(series: np.ndarray, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Shift a series backward by n periods to model delayed effects.
    
    Registered as "lag" in the transform registry. Introduces NaN at the
    start for lag warm-up (first n periods cannot be computed).
    
    Args:
        series: The input time series.
        rng: Seeded random generator (required by API, unused for lag).
        n: Number of periods to lag (default 1).
    
    Returns:
        The lagged series with the first n elements as NaN.
    """
    lagged = np.full_like(series, np.nan, dtype=float)
    lagged[n:] = series[:-n]
    return lagged
```

### Class-Based Generators and Transforms

Most generators and transforms in this codebase are classes (`__init__` +
`generate`/`apply`), not the plain decorated function shown above. The
docstring balance across the three spots is deliberate:

- **Class docstring**: registry note plus the concrete return-shape
  example (what `generate()`/`apply()` produces, with sample values) — the
  first thing a reader sees when they open the class, so it should tell
  them what the generator actually outputs, not just restate the field
  names from `__init__`'s signature.
- **`__init__` docstring**: stays minimal. No `Args:` section restating
  what a parameter name and type already say (`level: The constant value
  the series sits at` tells the reader nothing `level: float` didn't).
  Keep only `Errors Caught` for real validation. A parameter whose
  behavior genuinely isn't obvious from its name (e.g. does `clamp_min`
  floor or clip both ways?) is worth a plain sentence in the class
  docstring instead of an `Args:` entry here — a docstring with one
  `Args:` line for three parameters reads as if the other two were
  forgotten, not deliberately skipped.
- **`generate`/`apply` docstring**: keeps its full `Args:`/`Returns:`,
  including the same concrete example as the class docstring. Yes, this
  duplicates the class docstring's example — that's intentional: it's the
  method someone actually looks up when they want to know what calling it
  returns, so it should be self-contained without requiring a scroll back
  to the class docstring.

```python
@register_generator("flat")  # this string is what you write in YAML
class FlatGenerator(VariableGenerator):
    """A constant level with optional Gaussian noise — no seasonality.

    Registered as "flat" in the generator registry. generate() returns
    level plus noise, floored at clamp_min if set.
    Example: array([50.3, 49.1, 50.8, 49.6, ...]) for level=50, noise=1.
    """

    def __init__(
        self,
        level: float = 0.0,
        noise: float = 1.0,
        clamp_min: float | None = None,
    ):
        """Store the YAML params: for this variable.

        Errors Caught (raised to caller):
            ValueError: If noise < 0.
        """
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.level = level
        self.noise = noise
        self.clamp_min = clamp_min

    def generate(
        self, n_periods: int, period: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate the flat series.

        Args:
            n_periods: Number of time periods.
            period: Period type (e.g., "monthly", "daily"); unused — flat
                has no seasonal shape to align.
            rng: Seeded random generator for reproducibility.

        Returns:
            A numpy array of length n_periods, holding self.level plus
            optional noise, floored at clamp_min if set.
            Example: array([50.3, 49.1, 50.8, 49.6, ...]) for level=50,
            noise=1.
        """
        series = np.full(n_periods, float(self.level))
        if self.noise > 0:
            series = series + rng.normal(0.0, self.noise, size=n_periods)
        if self.clamp_min is not None:
            series = np.maximum(series, self.clamp_min)
        return series
```

---

## 12. Imports

Organize imports in three groups: standard library, third-party, then local. Use one import per line (except `typing` and `collections.abc` can share lines).

### ✅ DO: Organize imports in groups

```python
# Standard library
import hashlib
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# Local
from dsl.core.config.schema import ScenarioConfig
from dsl.core.pipeline.disease import build_disease_cases
from dsl.core.pipeline.periods import format_period, parse_period
```

**Within each group:** Sort alphabetically for consistency (optional but helpful).

### ✅ DO: Use full package paths (no relative imports)

```python
# ✅ GOOD: Full path, always clear where it comes from
from dsl.core.config.schema import ScenarioConfig
from dsl.core.pipeline.engine import run as run_engine

# ❌ BAD: Relative import, unclear and fragile
from . import config
```

### ✅ DO: Use `from x import y as z` when needed for clarity

```python
# Alias for clarity (common or long names)
from dsl.core.pipeline.engine import run as run_engine
from dsl.core.config.schema import ScenarioConfig as Config  # if shortening helps
```

### ✅ DO: Use standard abbreviations for common libraries

```python
import numpy as np
import pandas as pd
```

### ❌ DON'T: Use `import *`

```python
# BAD: Pollutes namespace, unclear where things come from
from dsl.core.pipeline import *
```

You cannot tell which names came from which module.

### ❌ DON'T: Mix import styles for the same module

```python
# BAD: Inconsistent
import numpy
import numpy as np  # Don't do both

# ✅ GOOD: Stick to one style
import numpy as np
```

### ✅ DO: Import modules/packages, not individual items (general rule)

```python
# ✅ GOOD: Import the module
import dsl.core.config

# ✅ GOOD: Import specific items when they're part of the public API
from dsl.core.config.schema import ScenarioConfig, VariableSpec
```

### ✅ OKAY: Exception for `typing` module

```python
# Can import multiple typing items on one line
from typing import Any, Dict, List, Optional, TypeVar
```
