
<div align="center">

# VitaCheck — AI Health Risk Classifier

**Biomedical risk stratification using ensemble machine learning**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/KarthikaKrishna123/vitacheck-ml)
[![GitHub](https://img.shields.io/badge/GitHub-vitacheck--ml-181717?logo=github&logoColor=white)](https://github.com/KARTHIKAKRISHNA123/vitacheck-ml)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Classifies individuals as **Healthy** or **At Risk** based on 22 clinical and lifestyle features,
> trained on 9,549 patient records with **99.30% recall** using a tuned Random Forest ensemble.

[Live Demo](https://huggingface.co/spaces/KarthikaKrishna123/vitacheck-ml) · [Report Issue](https://github.com/KARTHIKAKRISHNA123/vitacheck-ml/issues)

</div>

---

## Problem Statement

Biomedical research institutes recruit thousands of volunteers annually for clinical studies and longitudinal health analysis. Manually reviewing each participant's health profile to determine eligibility or risk level is time-consuming, inconsistent, and does not scale. Researchers lack a reliable automated system to stratify populations into healthy and at-risk groups based on multi-domain health indicators.

**VitaCheck** addresses this by providing a machine learning pipeline that ingests 22 physiological, lifestyle, and demographic features and produces a calibrated binary risk prediction — healthy (0) or at risk (1) — with a personalised explainability dashboard for each individual.

---

## Solution Overview

VitaCheck implements a complete supervised ML lifecycle:

- **EDA** — class balance analysis, correlation heatmaps, feature distribution plots, and boxplot comparisons across health outcomes
- **Preprocessing** — boolean normalization, stratified train-test split, StandardScaler fitting on training data only
- **Ensemble modeling** — five classifiers compared: Logistic Regression, KNN, Random Forest, Gradient Boosting, Voting Classifier
- **Threshold tuning** — decision threshold lowered from 0.50 to 0.25 to maximize recall
- **Explainability** — global feature importance from Random Forest, personalized risk factor display per user
- **Deployment** — interactive Streamlit frontend deployed on Hugging Face Spaces

---

## Key Features

| Feature | Description |
|---|---|
| 22-feature health input | Sliders and dropdowns covering physical, lab, lifestyle, and demographic data |
| Tuned decision threshold | 0.25 threshold increases recall from 95.88% to 99.30% |
| Personalised risk chart | Red/green bar chart showing which of the user's values fall outside healthy ranges |
| Global feature importance | Ranked bar chart from Random Forest |
| Complete health summary | Per-feature table with value, healthy range, status, and importance score |
| Risk probability gauge | Visual progress bar showing raw at-risk probability |
| Dark professional UI | Full dark theme with custom CSS |
| Model persistence | All artifacts saved: model, scaler, feature importance, threshold |

---

## Model Performance

| Model | Recall | Accuracy |
|---|---|---|
| Logistic Regression | 82.83% | 81.41% |
| KNN (k=5) | 88.35% | 88.32% |
| Gradient Boosting | 94.98% | 93.04% |
| Voting Classifier | 93.07% | 91.62% |
| **Random Forest** | **95.88%** | **93.77%** |
| **Random Forest (tuned, threshold=0.25)** | **99.30%** | **87.91%** |

> Recall is the primary metric. In medical risk classification, a false negative is more dangerous than a false positive. The tuned model misses only 7 out of 996 at-risk individuals.

---

## Overall Architecture

```mermaid
flowchart TD
    A[novagen_dataset.csv] --> B[EDA Notebook]
    B --> C[Preprocessing]
    C --> D[Model Training - 5 classifiers]
    D --> E[Random Forest - Best recall 95.88pct]
    E --> F[Threshold Tuning - 0.50 to 0.25 - Recall 99.30pct]
    F --> G[Model Artifacts]
    G --> H[app.py - Streamlit Frontend]
    H --> I[Hugging Face Spaces]
    H --> J[GitHub - KARTHIKAKRISHNA123]
```

---

## System Architecture

```mermaid
flowchart LR
    subgraph InputLayer["User Input Layer"]
        UI[Streamlit Sidebar - 22 feature widgets]
    end

    subgraph AppLayer["Application Layer - app.py"]
        LOAD[load artifacts - st.cache_resource]
        PREPROCESS[Input DataFrame - column order aligned]
        PREDICT[predict proba - threshold applied]
        EXPLAIN[Risk flag detection - healthy range comparison]
    end

    subgraph ArtifactLayer["Model Artifacts - model/"]
        M[vitacheck_model.pkl]
        S[scaler.pkl]
        FI[feature_importance.csv]
        TH[threshold.json - 0.25]
    end

    subgraph OutputLayer["Output Layer"]
        R[Result card - AT RISK or HEALTHY]
        P[Risk probability gauge]
        PRC[Personalised chart - red-green bars]
        TABLE[Health summary - 22-row HTML table]
    end

    UI --> PREPROCESS
    LOAD --> M & S & FI & TH
    PREPROCESS --> PREDICT
    M & TH --> PREDICT
    PREDICT --> R & P
    FI & EXPLAIN --> PRC & TABLE
```

---

## Technology Stack

| Technology | Version | Category | Purpose | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| Python | 3.13 | Runtime | Core language | Industry standard for ML | f-strings, venv, type hints |
| pandas | 2.2.3 | Data Processing | DataFrame operations, CSV loading | Best-in-class tabular data tool | read_csv, DataFrame, drop, isnull, describe |
| numpy | 2.1.3 | Numerical Computing | Array operations, threshold application | Foundational ML dependency | arange, astype, array slicing |
| scikit-learn | 1.6.1 | Machine Learning | All 5 classifiers, preprocessing, metrics | Complete ML lifecycle in one library | RandomForestClassifier, KNeighborsClassifier, LogisticRegression, GradientBoostingClassifier, VotingClassifier, StandardScaler, train_test_split, recall_score |
| matplotlib | 3.10.0 | Visualization | All charts in notebook and app | Fine-grained plot control | barh, subplots, patches, tight_layout, facecolor |
| seaborn | — | Visualization | Correlation heatmap, boxplots in EDA | Statistical plot presets | heatmap, boxplot, set_palette |
| joblib | 1.4.2 | Model Persistence | Save and load trained model and scaler | scikit-learn native serialization | dump, load |
| streamlit | 1.45.1 | Web Framework | Complete interactive frontend | Python-native web apps | set_page_config, cache_resource, sidebar, columns, progress, pyplot |
| json | stdlib | Config | Save and load tuned decision threshold | Lightweight human-readable config | json.dump, json.load |
| Git LFS | — | Version Control | Store large binary model files | HF Spaces rejects files over 10MB | lfs migrate import, lfs track |

---

## ML Pipeline

```mermaid
flowchart TD
    START([Start]) --> LOAD[Load novagen_dataset.csv]
    LOAD --> EDA[Perform EDA]
    EDA --> FIXBOOL[Fix boolean columns - True/False strings to 0/1]
    FIXBOOL --> SPLIT[Stratified train-test split 80pct 20pct]
    SPLIT --> SCALE[Fit StandardScaler on train only]
    SCALE --> TRAIN5[Train 5 classifiers]
    TRAIN5 --> EVAL[Evaluate all on recall and accuracy]
    EVAL --> BEST{Best recall?}
    BEST -->|Random Forest 95.88pct| TUNE[Threshold tuning sweep 0.10 to 0.90]
    TUNE --> SELECT[Select threshold 0.25 - Recall 99.30pct]
    SELECT --> SAVE[Save model, scaler, feature importance, threshold]
    SAVE --> END([End])
```

---

## Request Lifecycle

### Landing Page Load

```
1. Browser opens HF Space URL
2. app.py executes — st.set_page_config, CSS injected
3. load_artifacts() — joblib loads model.pkl, scaler.pkl
   pd.read_csv loads feature_importance.csv
   json.load reads threshold.json (0.25)
   All cached via @st.cache_resource
4. 22 sidebar widgets rendered inside st.form
5. Global feature importance chart rendered as base64 inline image
```

### Prediction Submission

```
1. User fills inputs, clicks Run Health Risk Assessment (form submit)
2. ONE Python rerun triggered — the only rerun in the session
3. input_data dict built — 22 features in exact training column order
4. model.predict_proba(input_df)[0][1] returns at-risk probability
5. prediction = int(proba >= 0.25) applies tuned threshold
6. Loop over 22 features vs HEALTHY_RANGES — builds risk_flags list
7. All HTML pre-computed and stored in st.session_state.result_cache
8. Result card, probability bar, personalised chart, summary table rendered
9. All post-result interactions handled in browser — zero further reruns
```

---

## Data Flow

```
novagen_dataset.csv
  → pandas DataFrame (9549 x 23)
  → Boolean fix (True/False strings to 0/1)
  → X = drop Target (9549 x 22), y = Target (9549,)
  → Stratified split: X_train (7639 x 22), X_test (1910 x 22)
  → StandardScaler.fit_transform(X_train) — learns mean and std
  → StandardScaler.transform(X_test) — applies same stats (no leakage)
  → 5 models trained and evaluated
  → Random Forest selected (best recall)
  → Threshold sweep: predict_proba compared at 16 thresholds
  → Best threshold 0.25 saved to threshold.json

Streamlit Runtime
  → User input (22 values via st.form) → pd.DataFrame (1 x 22)
  → model.predict_proba() → at-risk probability
  → probability >= 0.25 → AT RISK or HEALTHY
  → 22 values compared against HEALTHY_RANGES
  → All HTML pre-built → stored in result_cache
  → Personalised chart encoded as base64 → zero layout shift
```

---

## UML Diagrams

<details>
<summary>View all 9 UML diagrams</summary>

### 1. Use Case Diagram

```mermaid
flowchart TD
    USER([Researcher or Patient])
    SYSTEM[[VitaCheck System]]
    USER -->|enters 22 health features| UC1(Run Health Risk Assessment)
    USER -->|views| UC2(View Global Feature Importance)
    USER -->|reviews| UC3(View Personal Risk Factors)
    USER -->|reads| UC4(View Complete Health Summary)
    USER -->|interprets| UC5(View Risk Probability Score)
    UC1 --> SYSTEM
    UC2 --> SYSTEM
    UC3 --> SYSTEM
    UC4 --> SYSTEM
    UC5 --> SYSTEM
```

### 2. Class Diagram

```mermaid
classDiagram
    class VitaCheckApp {
        +model: RandomForestClassifier
        +scaler: StandardScaler
        +importance_df: DataFrame
        +THRESHOLD: float
        +HEALTHY_RANGES: dict
        +load_artifacts()
        +build_input_df()
        +predict()
        +compute_risk_flags()
        +render_landing()
        +render_prediction()
    }
    class RandomForestClassifier {
        +n_estimators: int
        +max_depth: None
        +random_state: int
        +feature_importances_: ndarray
        +fit(X, y)
        +predict(X)
        +predict_proba(X)
    }
    class StandardScaler {
        +mean_: ndarray
        +scale_: ndarray
        +fit_transform(X)
        +transform(X)
    }
    class HealthRecord {
        +Age: float
        +BMI: float
        +Blood_Pressure: float
        +Cholesterol: float
        +Glucose_Level: float
        +Heart_Rate: float
        +Sleep_Hours: float
        +Exercise_Hours: float
        +Water_Intake: float
        +Stress_Level: int
        +Target: int
    }
    VitaCheckApp --> RandomForestClassifier
    VitaCheckApp --> StandardScaler
    VitaCheckApp --> HealthRecord
```

### 3. Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant Sidebar
    participant App as app.py
    participant Model as vitacheck_model.pkl
    participant Renderer as Output Renderer

    User->>Sidebar: Fill 22 health feature inputs inside st.form
    User->>Sidebar: Click Run Health Risk Assessment
    Sidebar->>App: ONE rerun triggered by form submit
    App->>App: Build input_data dict
    App->>App: pd.DataFrame drop Target column
    App->>Model: predict_proba(input_df)
    Model-->>App: probability array
    App->>App: apply threshold 0.25
    App->>App: compare values vs HEALTHY_RANGES
    App->>App: pre-build all HTML into result_cache
    App->>Renderer: render from result_cache
    Renderer-->>User: Result card
    Renderer-->>User: CSS progress bar
    Renderer-->>User: Base64 inline chart
    Renderer-->>User: 22-row HTML summary table
```

### 4. Activity Diagram

```mermaid
flowchart TD
    S([Start]) --> L[Load dataset]
    L --> E[Perform EDA]
    E --> FB[Fix boolean columns]
    FB --> SP[Stratified train-test split]
    SP --> SC[Fit StandardScaler on train]
    SC --> T[Train 5 classifiers]
    T --> EV[Evaluate all on recall]
    EV --> B{Best recall?}
    B -->|Random Forest| TN[Threshold tuning sweep]
    TN --> SEL[Select threshold 0.25]
    SEL --> SV[Save artifacts]
    SV --> EN([End])
```

### 5. State Diagram

```mermaid
stateDiagram-v2
    [*] --> Loading : App starts
    Loading --> LandingPage : Artifacts cached
    LandingPage --> LandingPage : User fills form inputs (zero reruns)
    LandingPage --> Predicting : User clicks submit
    Predicting --> ResultPage : ONE rerun completes
    ResultPage --> ResultPage : User interacts via browser only
    ResultPage --> Predicting : User resubmits form
```

### 6. Component Diagram

```mermaid
flowchart TD
    subgraph FrontendLayer["Frontend - app.py"]
        CMP1[Sidebar st.form - 22 inputs]
        CMP2[Landing Page - metric cards and global chart]
        CMP3[Result Card - AT RISK or HEALTHY]
        CMP4[Risk Factor List - personalised flags]
        CMP5[Personalised Chart - base64 inline]
        CMP6[Summary Table - 22-row HTML]
    end
    subgraph MLLayer["ML Layer - model/"]
        CMP7[vitacheck_model.pkl]
        CMP8[scaler.pkl]
        CMP9[feature_importance.csv]
        CMP10[threshold.json]
    end
    CMP1 --> CMP3
    CMP7 --> CMP3
    CMP10 --> CMP3
    CMP9 --> CMP5
    CMP9 --> CMP6
    CMP4 --> CMP5
```

### 7. Deployment Diagram

```mermaid
flowchart TD
    subgraph DevMachine["Developer Machine"]
        NB[Jupyter Notebooks]
        LOCAL[Streamlit localhost 8501]
        GIT[Git with Git LFS]
    end
    subgraph GitHub["GitHub - KARTHIKAKRISHNA123"]
        REPO[vitacheck-ml main branch]
        PARENT[AIML_Knowledge_Base submodule]
    end
    subgraph HF["Hugging Face Spaces"]
        SPACE[vitacheck-ml Space - Streamlit SDK]
        LFS[Git LFS Storage - pkl files]
        CONTAINER[Docker Container - Python 3.13]
    end
    NB --> LOCAL
    LOCAL --> REPO
    REPO --> SPACE
    SPACE --> LFS
    SPACE --> CONTAINER
    REPO --> PARENT
```

### 8. Package Diagram

```mermaid
flowchart TD
    subgraph AppPkg["app package"]
        apppy[app.py]
    end
    subgraph ModelPkg["model package"]
        pkl[vitacheck_model.pkl]
        scaler[scaler.pkl]
        fimp[feature_importance.csv]
        thresh[threshold.json]
    end
    subgraph NotebookPkg["notebooks package"]
        eda[vitacheck_eda.ipynb]
        ml[vitacheck-ml.ipynb]
    end
    subgraph DataPkg["data package"]
        csv[novagen_dataset.csv]
    end
    apppy --> pkl
    apppy --> scaler
    apppy --> fimp
    apppy --> thresh
    ml --> csv
    ml --> pkl
```

### 9. Object Diagram

```mermaid
flowchart TD
    subgraph RuntimeObjects["App Runtime Objects"]
        OBJ1["model: RandomForestClassifier - n_estimators=200 - random_state=42"]
        OBJ2["scaler: StandardScaler - mean_ and scale_ per feature"]
        OBJ3["importance_df: DataFrame - 22 rows - Feature and Importance columns"]
        OBJ4["THRESHOLD: float = 0.25"]
        OBJ5["input_df: DataFrame - 1 row x 22 cols"]
        OBJ6["proba: float = predict_proba result"]
        OBJ7["risk_flags: list of tuples"]
    end
    OBJ1 --> OBJ6
    OBJ5 --> OBJ6
    OBJ4 --> OBJ6
    OBJ6 --> OBJ7
    OBJ3 --> OBJ7
```

</details>

---

## Data Flow Diagrams

<details>
<summary>View DFD Level 0 and Level 1</summary>

### DFD Level 0 — Context Diagram

```mermaid
flowchart LR
    E1[Researcher or Patient] -->|22 health feature values| P1(("0 - VitaCheck System"))
    P1 -->|Health risk classification and personalised report| E1
    E2[NovaGen Dataset CSV] -->|9549 patient records| P1
```

### DFD Level 1 — System Decomposition

```mermaid
flowchart TD
    E1[Researcher or Patient]
    E2[NovaGen Dataset CSV]

    E2 -->|Raw health records| P1(("1.0 - Load and Validate Data"))
    P1 -->|Clean DataFrame| D1[(D1 - Training Dataset)]

    D1 -->|Features and labels| P2(("2.0 - Train and Select Model"))
    P2 -->|Best model artifacts| D2[(D2 - Model Store)]

    E1 -->|22 health feature inputs| P3(("3.0 - Preprocess User Input"))
    D2 -->|scaler.pkl| P3
    P3 -->|Scaled input DataFrame| P4(("4.0 - Run Inference"))
    D2 -->|model.pkl and threshold| P4

    P4 -->|Risk probability and prediction| P5(("5.0 - Generate Explanation"))
    D2 -->|feature_importance.csv| P5

    P5 -->|Risk classification and personalised chart| E1
```

</details>

---

## Folder Structure

```
vitacheck-ml/
│
├── app.py                          ← Streamlit entry point — full dark UI
├── requirements.txt                ← HF Spaces dependency manifest
├── .gitattributes                  ← Git LFS tracking rules
│
├── model/                          ← Saved ML artifacts (Git LFS tracked)
│   ├── vitacheck_model.pkl         ← Trained RandomForestClassifier (200 trees)
│   ├── scaler.pkl                  ← Fitted StandardScaler
│   ├── feature_importance.csv      ← 22 features ranked by importance
│   └── threshold.json              ← Tuned threshold: 0.25
│
├── notebooks/
│   ├── vitacheck_eda.ipynb         ← EDA: null checks, heatmap, distributions
│   ├── vitacheck-ml.ipynb          ← Full ML pipeline: preprocessing to saving
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── boxplots_by_target.png
│   ├── feature_importances.png
│   ├── threshold_tuning.png
│   └── confusion_matrix.png
│
└── data/
    └── novagen_dataset.csv         ← Source dataset: 9549 records, 23 columns
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/KARTHIKAKRISHNA123/vitacheck-ml.git
cd vitacheck-ml

# Pull LFS files
git lfs pull

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Feature Reference

| Feature | Type | Description |
|---|---|---|
| Age | Continuous | Age in years |
| BMI | Continuous | Body Mass Index |
| Blood_Pressure | Continuous | Systolic blood pressure (mmHg) |
| Cholesterol | Continuous | Cholesterol level (mg/dL) |
| Glucose_Level | Continuous | Blood glucose (mg/dL) |
| Heart_Rate | Continuous | Resting heart rate (bpm) |
| Sleep_Hours | Continuous | Average sleep hours per day |
| Exercise_Hours | Continuous | Exercise hours per day |
| Water_Intake | Continuous | Daily water intake (litres) |
| Stress_Level | Ordinal | Self-reported stress (1-10) |
| Smoking | Ordinal | 0=Non-smoker, 1=Smoker, 2=Ex-smoker |
| Alcohol | Ordinal | 0=None, 1=Occasional, 2=Regular |
| Diet | Ordinal | 0=Standard, 1=Vegetarian, 2=Vegan |
| MentalHealth | Ordinal | 0=Good, 1=Moderate, 2=Poor |
| PhysicalActivity | Ordinal | 0=Sedentary to 3=Active |
| MedicalHistory | Ordinal | 0=None, 1=Minor, 2=Major |
| Allergies | Binary | Presence of known allergies |
| Diet_Type__Vegan | Binary | One-hot encoded from Diet |
| Diet_Type__Vegetarian | Binary | One-hot encoded from Diet |
| Blood_Group_AB | Binary | One-hot encoded blood type |
| Blood_Group_B | Binary | One-hot encoded blood type |
| Blood_Group_O | Binary | One-hot encoded blood type |

---

## Engineering Decisions

| Decision | Choice | Alternative | Rationale |
|---|---|---|---|
| Primary metric | Recall | Accuracy | Missing an at-risk person is medically more dangerous |
| Decision threshold | 0.25 | 0.50 default | Increases recall from 95.88% to 99.30% |
| Best model | Random Forest | Gradient Boosting | Higher recall, no scaling needed, faster inference |
| Feature scaling | LR and KNN only | Universal scaling | Tree-based models are scale-invariant |
| Table rendering | Pure HTML divs | pandas Styler | Streamlit dark theme makes Styler colors invisible |
| Model serialization | joblib | pickle | Better handling of large numpy arrays |
| Large file storage | Git LFS | Regular git | HF Spaces rejects files over 10MB in git history |
| Sidebar inputs | st.form | Individual widgets | Form suppresses reruns on every input — only one rerun on submit |
| Chart rendering | Base64 inline img with explicit dimensions | st.pyplot | Eliminates image-load layout shift; browser pre-allocates space before PNG decodes |
| Progress bar | CSS div with gradient | st.progress | st.progress re-renders its DOM on rerun causing flicker |
| Expandable section | HTML details/summary | st.expander | st.expander triggers full Python rerun on click; native HTML is browser-only |

---

## Fixing the HF Spaces Page Shake

<details>
<summary>Root cause analysis and resolution</summary>

### Why It Happens

HF Spaces embeds Streamlit apps inside an `<iframe>` with `height: auto`. The iframe listens to the Streamlit app via `postMessage` and resizes to match content height. Streamlit reruns the entire Python script on every widget interaction, which causes:

```
widget click → Python rerun → DOM cleared → DOM repainted → iframe resizes → visible jump
```

### What Was Tried and Why Each Failed

| Attempt | Why it failed |
|---|---|
| `@st.cache_data` on charts | Prevented recomputation but HTML elements still re-rendered |
| `st.empty()` slot | Slot still patches its content on every rerun |
| `st.session_state` snapshot | Identical HTML still caused DOM patches and flicker |
| Pre-computed `result_cache` | Python reruns still caused iframe resize |
| CSS `height: 100vh` lock | Applied too late; Streamlit's initial paint overrode it |
| Sliders to `number_input` | `+`/`-` buttons still triggered continuous reruns |
| `@st.fragment` on sidebar | `st.sidebar.*` not supported inside a fragment |
| `with st.sidebar:` inside fragment | Streamlit 1.45.1 blocks this — fragment and sidebar contexts conflict |

### The Actual Fix — Four Compounding Causes

**Cause 1 — Continuous reruns on every widget interaction**

Fixed with `st.form()`. All sidebar inputs wrapped in a single form. Widget interactions inside a form produce zero reruns. Only the submit button triggers one deliberate rerun.

**Cause 2 — `st.pyplot` image-load layout shift**

`st.pyplot` inserts an `<img>` with no declared dimensions. The browser renders the page, reports height to HF Spaces, then the PNG loads, the image expands, and a second resize message fires. Fixed by encoding charts as inline base64 with explicit pixel dimensions:

```python
f'<img src="data:image/png;base64,{b64}" width="{pw}" height="{ph}" style="width:100%;height:auto;">'
```

**Cause 3 — DOM branch swap on state transition + iframe resize**

When landing page transitions to results page, all DOM nodes are replaced. Fixed with CSS:

```css
.block-container { min-height: 100vh !important; }
.stApp { height: 100vh !important; overflow: hidden !important; }
[data-testid="stMain"] { height: 100vh !important; overflow-y: auto !important; }
```

**Cause 4 — `st.expander` and `st.progress` rerenders**

`st.expander` click triggers a full Python rerun. `st.progress` re-renders its DOM on rerun. Fixed by replacing both with pure HTML equivalents — CSS gradient div for progress, native `<details>`/`<summary>` for expandable sections.

### Final State

The app triggers **exactly one** Python rerun per session — when the user clicks submit. Every other interaction is handled entirely in the browser. The iframe height is locked and never changes.

</details>

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| FileNotFoundError: model/vitacheck_model.pkl | LFS not pulled or wrong directory | Run `git lfs pull` then `streamlit run app.py` from project root |
| FileNotFoundError in notebook | Notebook is in notebooks/ subdirectory | Use `../data/novagen_dataset.csv` |
| HF push rejected over 10MB | Model committed without LFS | Run `git lfs migrate import --include="*.pkl" --everything` |
| HF push rejected binary files | PNG files not in LFS | Run `git lfs migrate import --include="*.png" --everything` |
| KeyError RdY1Gn colormap | Typo — 1 instead of lowercase l | Change to `cmap="RdYlGn"` |
| Page shaking on HF Spaces | iframe auto-resize on every Streamlit rerun | Use `st.form` for all inputs; render charts as base64 with explicit dimensions; replace `st.expander` and `st.progress` with HTML equivalents |
| `st.fragment` sidebar error | `st.sidebar.*` not supported inside fragment | Use `st.form` instead — simpler and more effective for this use case |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Credits

- Dataset: NovaGen Research Labs observational health study (9,549 records)
- Built as part of Anna University CSE supervised ML curriculum
- Developed by **Karthika Krishna M** — [GitHub](https://github.com/KARTHIKAKRISHNA123)

---

<div align="center">

**VitaCheck** · Random Forest + Streamlit · For research purposes only · Not a substitute for clinical diagnosis

</div>
