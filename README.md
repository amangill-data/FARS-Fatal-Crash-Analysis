# FARS Fatal Crash Analysis

*(See repository files: `P1_EDA_FARS23.ipynb` and `P2_Modeling_FARS23.ipynb` for full implementation. The encoding glossary documents all FARS variable codes. The project presentation deck provides a narrative walkthrough of findings.)*

Binary classification analysis of the **NHTSA FARS 2023** dataset — identifying key predictors of alcohol-impaired and speeding-related fatal crashes across the United States using Logistic Regression, Random Forest, and XGBoost, with SHAP-based explainability.

**Conducted by Aman Gill & Musa Sayeed**

---

## Overview

The leading causes of fatal motor vehicle crashes in the U.S. are alcohol impairment and speeding. This project uses NHTSA's Fatality Analysis Reporting System (FARS) — a comprehensive national census of every fatal crash occurring on public roadways — to model the conditions that predict these two outcomes at the driver level.

Rather than relying on pre-aggregated summaries, we work directly from the relational FARS database: merging the accident, vehicle, and person files with their corresponding NHTSA auxiliary files to build a driver-level analytic dataset of 57,939 records spanning all 50 states in 2023.

---

## Research Questions

| # | Target | Question |
|---|--------|----------|
| RQ1 | Alcohol | How do driver demographic and behavioral characteristics (age, sex, prior DWI history, vehicle type, speed limit) interact with alcohol impairment in fatal crashes? |
| RQ2 | Alcohol | Which roadway, environmental, and temporal characteristics (rural/urban, road function, weather, lighting, day of week, hour) are most strongly associated with alcohol-impaired drivers? |
| RQ3 | Speeding | Are roadway design and environmental conditions (road classification, trafficway type, intersection type, weather, surface condition, speed limit, lighting) associated with speeding-related fatal crashes? |
| RQ4 | Speeding | How do speeding-related fatal crash risk factors vary across states and roadway classifications in 2023, and what do these patterns suggest about geographic disparities in traffic safety? |

---

## Data

**Source:** [NHTSA FARS 2023](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars) — a federally mandated census of all fatal motor vehicle crashes on U.S. public roadways.

Six files are used, joined as a relational database:

| File | Grain | Rows | Purpose |
|------|-------|------|---------|
| `accident.csv` | 1 per crash | 37,654 | Crash context: time, location, weather, road type |
| `person.csv` | 1 per person | 92,400 | Driver demographics, BAC results, injury severity |
| `vehicle.csv` | 1 per vehicle | 58,319 | Speeding flag, speed limit, road geometry, prior violations |
| `acc_aux.csv` | 1 per crash | 37,654 | NHTSA pre-computed crash-level analytical flags |
| `per_aux.csv` | 1 per person | 92,400 | NHTSA pre-computed person-level flags |
| `veh_aux.csv` | 1 per vehicle | 58,319 | NHTSA pre-computed vehicle-level flags |

**Join keys:** `ST_CASE` (crash) · `ST_CASE + VEH_NO` (vehicle/person) · `ST_CASE + VEH_NO + PER_NO` (aux files)

### Target Variable Derivation

**`IS_ALCOHOL_IMPAIRED`** — derived from `ALC_RES` (person file):
- BAC stored as integer: `8` = 0.08 g/dL, `15` = 0.15 g/dL, etc.
- Codes 995–999 (not tested / refused / unknown) set to `NaN`
- `1` if BAC ≥ 8 · `0` if below threshold · `NaN` if untested
- ~64% of drivers have no valid BAC test; all alcohol rates are computed on **tested drivers only** (n = 20,955)

**`IS_SPEEDING`** — derived from `SPEEDREL` (vehicle file):
- `1` for codes 2–5 (Too Fast for Conditions, Racing, Exceeded Limit, Too Fast Other)
- `0` for code 0 (Not Speed Related)
- `NaN` for codes 8–9 (Not Reported / Unknown)
- Valid speeding records: n = 54,564 · **19.7% speeding-involved**

---

## Pipeline

### Phase 1 | Exploratory Data Analysis (`P1_EDA_FARS23.ipynb`)

- Filter person file to drivers only (`PER_TYP == 1`): 57,939 of 92,400 persons
- Derive binary targets and validate against NHTSA auxiliary flags
- Analyze missing data patterns and their real-world causes
- EDA across temporal, demographic, environmental, and roadway dimensions
- Cross-target analysis: alcohol × speeding co-occurrence (speeding crashes are **1.9× more likely** to also involve an impaired driver)

**Key EDA Findings:**

| Variable | Alcohol Finding | Speeding Finding |
|----------|----------------|-----------------|
| Hour of day | Peaks 1–3 AM (~73% impairment); lowest 10–11 AM | Elevated late-night; mirrors alcohol curve |
| Day of week | Sunday highest (~49%); Saturday second | Friday–Sunday cluster |
| Age | 20–29 highest (~45%); declines with age | — |
| Sex | Male ~36% vs. Female ~26% | — |
| Road surface | — | Ice/snow highest (~35–47%) |
| Speed limit | — | Lower-limit paradox: 25–35 mph zones > 65–70 mph highways |

### Phase 2 | Modeling (`P2_Modeling_FARS23.ipynb`)

Three models trained per target with `class_weight='balanced'` / `scale_pos_weight` to handle class imbalance. All categorical variables one-hot encoded. Numeric unknowns (FARS codes 97–99) imputed with column median.

**Alcohol model features (131 after OHE):** driver demographics, prior record, vehicle type, temporal variables, environmental/road context

**Speeding model features (241 after OHE):** roadway design & geometry, environmental conditions, work zone flag, driver behavioral history, temporal variables, state-level geography

SHAP TreeExplainer run on both XGBoost models for feature-level explainability. RQ4 geographic analysis maps mean predicted speeding probability across all 50 states and road function classes.

---

## Results

### Alcohol Impairment Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.711 | 0.572 | 0.721 | 0.638 | 0.785 |
| Random Forest | 0.726 | 0.592 | 0.722 | 0.650 | 0.801 |
| **XGBoost** | **0.747** | **0.618** | **0.739** | **0.673** | **0.824** |

### Speeding Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.745 | 0.416 | 0.723 | 0.528 | 0.811 |
| Random Forest | 0.761 | 0.432 | 0.675 | 0.526 | 0.804 |
| **XGBoost** | **0.770** | **0.446** | **0.692** | **0.543** | **0.820** |

XGBoost achieved the best ROC-AUC on both targets. SHAP values were computed on the XGBoost models to identify top predictors per research question.

---

## Technical Stack

| Layer | Tool |
|-------|------|
| Data Source | NHTSA FARS 2023 (CSV / XLSX) |
| Data Manipulation | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Preprocessing | scikit-learn `ColumnTransformer`, `OneHotEncoder` |
| Models | scikit-learn `LogisticRegression`, `RandomForestClassifier` · XGBoost `XGBClassifier` |
| Explainability | SHAP `TreeExplainer` |
| Environment | Python 3.x, Jupyter Notebook / Google Colab |

---

## Repository Contents

| File | Description |
|------|-------------|
| `P1_EDA_FARS23.ipynb` | Data loading, merging, target derivation, missing value analysis, and full EDA across Sections 0–6 |
| `P2_Modeling_FARS23.ipynb` | Preprocessing pipeline, Logistic Regression / Random Forest / XGBoost models, SHAP explainability, and RQ4 geographic analysis |
| `[Presentation].pptx` | Project presentation deck — narrative walkthrough of methodology and findings |
| `[Encoding Glossary]` | FARS variable code reference — maps raw integer codes to their labeled meanings for all variables used in this analysis |
