# Buy or Wait? — AI Financial Decision Agent

An AI-assisted financial decision engine that answers a deceptively simple question:

> **“Can I afford this?”**

The system does not make that decision from the current account balance alone. It reconstructs a user's financial state, resolves evidence from transactions, messages and images, converts currencies, detects recurring financial commitments, forecasts cash flow for 90 days, evaluates payment strategies, and deterministically validates the final recommendation.

The result is a structured decision for every request:

- pay in full
- pay partially
- use installments
- wait
- or do not proceed

The key design principle is **deterministic financial reasoning with bounded LLM assistance**. The LLM can help interpret messy evidence and rewrite explanations, but it is deliberately kept outside the financial decision core.

---

## Table of Contents

- [Problem](#problem)
- [Solution Overview](#solution-overview)
- [Core Design Principles](#core-design-principles)
- [Architecture](#architecture)
- [Decision Pipeline](#decision-pipeline)
- [Financial State Reconstruction](#financial-state-reconstruction)
- [Evidence and LLM Layer](#evidence-and-llm-layer)
- [90-Day Forecasting](#90-day-forecasting)
- [Decision Engine](#decision-engine)
- [Output Validation](#output-validation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Evaluation and Observability](#evaluation-and-observability)

---

## Problem

A purchase can look affordable today and still be unsafe.

For example, a user may have enough money in their account to buy a laptop, but upcoming rent, utilities, debt repayments, subscriptions, pending card transactions, or a delayed salary could make the purchase unsafe.

The challenge therefore requires the system to reason over:

- current available balance
- minimum balance the user wants to preserve
- recurring expenses
- one-time expenses
- pending transactions
- confirmed income
- settlement dates
- flexible spending
- user priorities
- payment preferences
- installment options
- dated foreign-exchange rates
- messages containing financial evidence
- images such as payroll letters, bills, receipts and statements

The system must produce one validated prediction for every request in `dataset/requests.csv`.

---

# Solution Overview

The implementation is organized as a deterministic financial pipeline:

```text
                    ┌─────────────────────┐
                    │      Dataset        │
                    │ CSV + Images +      │
                    │ Messages + Options  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Loading     │
                    │ Schema validation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Normalization    │
                    │ Dates / amounts /   │
                    │ currencies / types  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐           ┌─────────────────────┐
   │ Evidence Resolution │           │ Currency Conversion │
   │ Messages + Images   │           │ Fixed dated FX     │
   └──────────┬──────────┘           └──────────┬──────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ State Reconstruction│
                    │ Cash roles +        │
                    │ recurring series    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 90-Day Forecast     │
                    │ Income / expenses / │
                    │ reserves / changes  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Decision Engine     │
                    │ Full / partial /    │
                    │ installments / wait │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Candidate Ranking   │
                    │ Deadline → changes  │
                    │ → cost → timing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Output Validation   │
                    │ Hard constraints    │
                    └──────────┬──────────┘
                               │
                               ▼
                         ┌───────────┐
                         │ output.csv│
                         └───────────┘
```

---

# Core Design Principles

### 1. The LLM is not the financial authority

The LLM is intentionally outside the financial decision core.

It may:

1. classify previously-unhandled financial messages into an allow-listed evidence schema;
2. extract an amount from an image when deterministic OCR/cache resolution is insufficient;
3. rewrite an already-determined decision explanation.

It does **not**:

- calculate balances;
- perform the financial forecast;
- choose a payment method;
- choose spending changes;
- rank financial candidates;
- override validation rules.

This separation keeps the financially important path reproducible and auditable.

### 2. Missing money is never silently treated as zero

If a financial event has a missing amount, the system attempts to resolve it from linked image evidence.

If it still cannot resolve the amount, it remains unresolved rather than being silently converted into `0`.

### 3. Safety is evaluated across time

A recommendation is only considered safe when the complete payment plan can be simulated while preserving the user's required minimum balance throughout the relevant forecast horizon.

### 4. User preferences are constraints

The engine respects:

- payment methods the user accepts;
- maximum installment duration;
- categories the user wants protected;
- categories the user is willing to reduce;
- categories the user is willing to stop.

### 5. Final output is independently validated

The decision engine and the output validator are separate concerns. The generated CSV is checked against the required schema and financial-plan constraints before the run is considered successful.

---

# Architecture

The code is divided into focused stages.

| Module | Responsibility |
|---|---|
| `main.py` | Orchestrates the complete pipeline |
| `data_loader.py` | Loads and validates dataset CSVs |
| `models.py` | Raw dataset data structures |
| `normalization.py` | Converts raw values into typed normalized objects |
| `currency.py` | Applies dated FX rates |
| `evidence.py` | Resolves image/message evidence and extracts financial amounts |
| `reconstruction.py` | Reconstructs user financial state and recurring series |
| `forecast.py` | Builds and simulates the 90-day cash forecast |
| `planner.py` | Generates, evaluates and ranks payment candidates |
| `decision_engine.py` | Converts the winning candidate into an output decision |
| `explanations.py` | Produces deterministic explanations and optional LLM rewrites |
| `money.py` | Decimal-safe money formatting |
| `output_writer.py` | Writes the final `output.csv` |
| `validate_output.py` | Validates output structure and decision constraints |
| `llm_agent.py` | Bounded Groq/OpenAI-compatible LLM integration and usage tracking |

---

# Decision Pipeline

The runtime starts in `code/main.py`.

The orchestration is:

```text
load_dotenv()
      │
      ▼
LLM client + usage tracker
      │
      ▼
load_dataset()
      │
      ▼
normalize_dataset()
      │
      ▼
reconstruct_dataset()
      │
      ▼
decide_requests()
      │
      ▼
write_output()
      │
      ▼
validate_output_file()
      │
      ▼
usage_report.md
llm_runtime_transcript.jsonl
```

The entry point also prints dataset, normalization and reconstruction summaries during execution.

---

# Financial State Reconstruction

The reconstruction layer turns raw transactions into a usable financial model.

Each reconstructed event contains information such as:

- event ID
- user ID
- event type
- category
- direction
- original amount/currency
- home-currency amount
- event date
- settlement date
- status
- flexibility
- minimum allowed amount
- cash role
- cadence
- recurring-series ID
- amount source
- provenance

The reconstructed user state additionally contains:

- current available balance
- protected minimum balance
- reconstructed events
- recurring series
- evidence overlays
- event lookup indexes
- exchange-rate lookup data

## Cash-role classification

Events are assigned one of:

```text
include
reserve
ignore
```

Examples include:

- cancelled transactions → ignored
- failed transactions → generally ignored
- unrealized/non-cash events → ignored
- pending debits → reserved
- pending credits → not immediately counted
- confirmed salary → included according to its settlement date
- settled/scheduled cash events → included where appropriate
- unconfirmed windfalls → ignored

This prevents optimistic assumptions from contaminating the forecast.

---

# Currency Normalization

The dataset contains multiple currencies:

```text
EUR
IDR
INR
USD
ZAR
```

Amounts are converted into each user's `home_currency` using the dated exchange-rate table.

The conversion date uses the event settlement date when available, otherwise the event date.

The engine also preserves FX provenance, for example:

```text
fx:USD->INR@2026-09-15
```

Money calculations use Python `Decimal` rather than floating-point arithmetic.

---

# Evidence and LLM Layer

Financial evidence can appear outside the transaction row itself.

The implementation handles:

### Messages

Messages can contain additional financial information such as:

- payment confirmations
- income information
- changed amounts
- other event-specific evidence

The evidence layer parses supported message overlays and feeds them into reconstruction.

### Images

Images may contain:

- payroll information
- statements
- receipts
- bills
- other financial documents

The system first attempts deterministic extraction.

The image pipeline includes:

1. cached verified image amounts;
2. OCR using multiple page-segmentation configurations;
3. bounded vision-LLM extraction when deterministic extraction cannot resolve the amount.

Image cache entries are hash-checked before use, helping ensure the cached amount corresponds to the actual image.

---

# 90-Day Forecasting

The forecast horizon is:

```text
request_date → request_date + 90 days
```

The forecast includes future cash flows from reconstructed events and recurring series.

Each forecast flow tracks:

- date
- signed amount
- category
- description
- recurring-series ID
- change event ID
- flexibility
- minimum allowed amount
- source

Credits increase the balance.

Debits decrease it.

The forecast also respects pending transactions and recurring obligations rather than looking only at today's balance.

## Recurring-series detection

The reconstruction layer detects recurring patterns from historical events.

Different categories receive different treatment.

Examples include:

- salary streams
- rent
- utilities
- insurance
- debt repayment
- education
- housing
- family support
- subscriptions
- variable spending categories

The implementation distinguishes recurring, variable and one-time events.

For recurring series, the engine models:

- recurrence period
- typical amount
- last event date
- flexibility
- minimum allowed amount
- occurrence count
- member event IDs

---

# Decision Engine

For each request, the planner first calculates:

```text
safe amount today
earliest date full amount becomes safe
```

It then evaluates permitted payment strategies.

## 1. Full payment

A full-payment candidate is created when paying the entire requested amount on the request date can be simulated safely.

```text
request_date
     │
     └── requested_amount
```

## 2. Wait

If the amount is not safe today but becomes safe later and the date is within the requested deadline, the engine can create:

```text
earliest_safe_date
     │
     └── requested_amount
```

## 3. Partial payment

Partial payment is considered only when:

- the request permits partial payment;
- the user accepts partial payment;
- today's safe amount is greater than zero;
- today's safe amount is less than the requested amount;
- the full amount can become safe later;
- the later date is within the deadline.

The resulting plan contains exactly two payments:

```text
request_date: safe_today
earliest_safe_date: requested_amount - safe_today
```

The two payments must sum exactly to the requested amount.

## 4. Installments

Installment candidates are generated only from payment options supplied by the dataset.

The engine:

1. builds the exact schedule from the option;
2. rejects schedules beyond the user's maximum installment duration;
3. simulates the complete schedule;
4. rejects any schedule that violates the minimum balance.

This means the engine does not invent installment terms.

## 5. Spending changes

The system can model up to three permitted recurring expense changes.

Eligible changes are constrained by:

- category protection preferences;
- user willingness to reduce/stop the category;
- event flexibility;
- minimum allowed amounts.

Supported changes are:

```text
stop:<event_id>
```

or:

```text
reduce_to:<event_id>:<amount>
```

The planner enumerates combinations rather than relying on a simple “largest savings first” heuristic. This matters because the safest valid solution can involve a combination of smaller expenses.

---

# Output Validation

The final CSV is independently validated before completion.

Required columns are:

| Column |
|---|
| `request_id` |
| `amount_safe_to_pay` |
| `affordability_status` |
| `recommended_payment_method` |
| `payment_plan` |
| `earliest_date_for_full_payment` |
| `spending_changes_needed` |
| `decision_explanation` |

The validator checks, among other constraints:

### Amount bounds

```text
0 <= amount_safe_to_pay <= requested_amount
```

### Request coverage

There must be exactly one output row for every expected request ID.

### Full payment

A full-payment plan must:

- contain exactly one payment;
- equal the requested amount;
- occur on or after the request date;
- be accepted by the user's payment preferences.

### Partial payment

A partial-payment plan must:

- be allowed by the request;
- be accepted by the user;
- contain exactly two payments;
- start on the request date;
- use `amount_safe_to_pay` as the first payment;
- sum exactly to the requested amount;
- finish by the requested deadline.

### Installments

An installment plan must exactly match one of the supplied payment options.

The user's maximum installment duration is also enforced.

### Waiting

A wait plan must:

- have `affordable_later` status;
- have a valid earliest full-payment date;
- contain exactly one payment;
- pay the requested amount;
- pay on the earliest safe date;
- finish by the requested deadline.

### Spending changes

Every spending change is checked against:

- the correct user;
- debit direction;
- protected categories;
- user willingness;
- event flexibility;
- minimum allowed amount;
- maximum of three changes;
- duplicate-event restrictions.

---

# Dataset

The challenge provides multiple CSV sources.

| File | Purpose |
|---|---|
| `requests.csv` | Target requests requiring predictions |
| `sample_requests.csv` | Solved examples used for validation |
| `financial_profiles.csv` | User balances, minimums, priorities and preferences |
| `financial_events.csv` | Historical and future financial events |
| `exchange_rates.csv` | Dated currency conversion rates |
| `request_payment_options.csv` | Valid payment/installment options |
| `messages.csv` | Supporting user/event/request messages |
| `images.csv` | Links between financial events and images |
| `media/images/` | Financial document images |

Only `requests.csv` requires predictions.

All other datasets provide context for the decision.

---

# Configuration

LLM configuration is environment-based.

Create a local `.env` from the provided example:

```env
GROQ_API_KEY=YOUR_KEY
LLM_ENABLED=1
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_VISION_MODEL=qwen/qwen3.6-27b
```

Never commit:

```text
.env
API keys
secrets
```

The financial engine can also operate with the deterministic fallbacks when LLM functionality is unavailable or disabled.

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

## Linux / macOS

From the repository root:

```bash
PYTHONPATH=code python code/main.py
```

## Windows PowerShell

```powershell
$env:PYTHONPATH="code"
python code/main.py
```

## Optional limited run

The entry point supports a `--limit` argument for processing a smaller number of requests during development:

```bash
PYTHONPATH=code python code/main.py --limit 10
```

The normal submission run should use the full dataset without `--limit`.

---

# Evaluation and Observability

The challenge evaluates the generated output on:

- `amount_safe_to_pay`
- `affordability_status`
- `recommended_payment_method`
- `payment_plan`
- `earliest_date_for_full_payment`
- `spending_changes_needed`
- `decision_explanation`

The implementation also provides internal summaries for:

- loaded dataset counts
- normalized records
- reconstructed users/events
- cash-included events
- reserved events
- ignored events
- recurring/variable/one-time events
- recurring series
- evidence overlays
- image-resolved amounts
- unresolved amounts

This makes the pipeline easier to inspect when a prediction looks wrong.


