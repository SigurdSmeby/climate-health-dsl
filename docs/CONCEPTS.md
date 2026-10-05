# Concepts

Why the tool works the way it does. Understanding-oriented: not a field lookup
(that's the [reference](REFERENCE.md)) and not a walkthrough (the
[tutorial](TUTORIAL.md)).

Four ideas, in the order they bite.

## 1. One list, and what depends on what

A scenario declares one `series:` list. Climate and disease are not different
kinds of thing — each is a named series, and a series becomes a disease signal
only by carrying a `counts:` block.

The point is that **any series can be built from any other**. Rain fills a soil
moisture store, heat drains it, the disease responds to the store:

```yaml
series:
  - name: rainfall
    generate: seasonal_spike
  - name: soil_moisture              # no generator: its parents give it shape
    depends_on:
      - { series: rainfall, lag: 1, weight: 1.0 }
      - { series: temperature, lag: 1, weight: -0.4 }
  - name: disease_cases
    counts: {}
    depends_on:
      - { series: soil_moisture, lag: 2 }
```

`soil_moisture` is a real column in the output. The chain you planted is
visible in the data, not buried inside a weight — so a model's claim about it
can be checked.

### Why cycles are rejected

The list is a directed graph, and the engine generates in dependency order:
parents before children. That requires no loops. A chain of any length is fine
in one direction — `wind → temp → rainfall` — but adding `wind ← rainfall`
closes a three-series loop, and the run stops before generating anything:

```
series dependencies must not form a cycle: rain -> wind -> temp -> rain.
Give the relationship one direction, or use a shared event if these series
should move together in the same period.
```

The check walks the whole graph, so a loop of **any** length is caught — not
just a direct `A ↔ B` pair. That matters because the three-series case is easy
to build by accident and impossible to spot by eye in a long scenario.

A loop is rejected even with lags. Resolving `rain[t] ← temp[t-1]` *and*
`temp[t] ← rain[t-1]` would need the engine rebuilt as a per-timestep loop, and
the case people actually want — two series moving together — is an event, not
mutual causation. Which brings us to:

## 2. Events are not generators

A generator makes **one** series and cannot see the others. So
`seasonal_spike` on rainfall can never make temperature move in the same month,
however the parameters are tuned.

An event can. It fires in specific periods and strikes every series that opts
in — in the *same* period, in whichever direction each one needs:

```yaml
events:
  storm: { at: [5] }

series:
  - name: rainfall
    generate: flat
    params: { level: 10.0, noise: 0.0 }
    events: { storm: { multiplier: 2.5 } }   # proportional, up
  - name: temperature
    generate: flat
    params: { level: 20.0, noise: 0.0 }
    events: { storm: { add: -6.0 } }         # absolute, down
```

Period 5: rainfall goes 10 → 25 while temperature goes 20 → 14. One cause, two
series, same month, opposite directions.

### When to use which

|  | generator | event |
|---|---|---|
| Timing | every year, same point in the cycle | specific periods (`at`) or randomly (`rate`) |
| Shape | a curve spread over periods | applies to the periods it fires on |
| Reach | **one** series | **several** series at once |
| Scales with resolution | yes | no — indices are raw periods |

They compose, and a realistic scenario uses both: rainfall has a yearly wet
season *and* occasional storms; temperature has no wet season but shares the
storms.

Two notes on the effects. `multiplier` is proportional, so it grows with the
series' own level and can never move a series sitting at zero, nor push a
positive series negative. `add` shifts by a fixed amount regardless of level —
that is the one to use when the effect is absolute, like a storm dropping the
temperature by six degrees.

Events are **regional**: drawn once and shared by every location, so a storm
hits the whole region in the same periods. That is what makes two provinces
move together, and the correlation has a name you can point at rather than an
anonymous mixing weight.

## 3. How a disease signal becomes case counts

A `counts:` block runs a population-relative incidence model, not a weighted
sum. In order:

1. Start from the series' own generator, or flat 0 when it has only parents.
2. For each `depends_on` entry: delay the parent by `lag` (causally — the
   warm-up becomes NaN, and values never wrap round from the end), apply any
   `transforms`, standardize to a z-score, multiply by `weight`, add.
3. Apply any `events` this series reacts to, before anything downstream reads it.
4. If `autoregressive`, add an AR(1) process: `x[t] = phi·x[t-1] + noise`.
   Unlike a random walk it stays bounded, and `phi` is itself a ground truth a
   model can be scored on recovering.
5. Squash through a sigmoid shifted so a typical period lands near
   `median_rate`, then scale: `sigmoid × population × max_rate`.
6. Draw whole counts from that rate (Poisson, or overdispersed negative
   binomial), capped at `population`.
7. Blank every period with no valid input — the lag warm-up, plus rows where a
   parent was itself missing — then apply `missing_rate` last.

**Standardizing in step 2 is why weights are comparable.** Each driver becomes
a z-score before its weight applies, so `weight: 2.0` means the same whether
the driver is millimetres of rain or degrees of temperature.

**`max_rate` caps the rate, not each draw.** With `max_rate: 0.02` over a
population of 1000 the expected count is 20 — but Poisson draws scatter around
20, and in 4000 periods the largest was **39**. Only `population` is a hard cap
on a drawn value. Worth stating plainly if you describe the model in writing.

**There is no built-in seasonal term.** A disease is seasonal because its
drivers are. That way the seasonal amplitude a model recovers is the one the
scenario planted, rather than that plus an invisible constant.

## 4. Why folds need a report

Splitting data in two is easy. The hard question is whether each half still
contains what you are testing for.

A fold whose test half holds **no outbreak** will score well on outbreak
recovery — there was nothing to recover. The number looks like success. So
every run with a `split:` writes a report counting each planted feature on both
sides of every fold:

```
fold,series,kind,train,test
0,rainfall,seasonal_spike,2,1
0,rainfall,storm,1,0
0,heatwaves,outbreak,1,1
```

and warns where a fold cannot measure something:

```
fold 0: no 'storm' in the test half of 'rainfall' (1 in train); this fold
cannot measure how well a model recovers it.
```

**The counts are exact, not detected.** Each generator reports the periods it
placed a feature at, so two overlapping outbreak shocks are still counted as
two — which no threshold detector could recover from the output.

### Expanding, not shuffled

The default time split trains each fold only on the periods *before* its test
block. Shuffling periods, or the `blocked` alternative, lets a fold train on
data from after what it predicts — which inflates a forecaster's score for a
reason that has nothing to do with the model. `blocked` exists because it is
the textbook k-fold and reviewers ask for it; the run warns when you choose it.

## See also

- **[Reference](REFERENCE.md)** — the exact fields these ideas map to.
- **[Tutorial](TUTORIAL.md)** — see the lag and the warm-up live.
- **[How-to](HOW_TO.md)** — add a generator, transform or emitter.
