flowchart TB

%% =======================
%% PREPROCESSING SECTION
%% =======================

subgraph INPUT["Input Data"]
    RC[Raw Counts]
    MG[(User Marker Genes?<br/>If missing → Run DE)]
end

subgraph OPTIONAL_DE["Optional Differential Expression (DE)"]
    direction LR
    RC -->|If no marker genes| DE[DE → Marker Gene Discovery]
end

MG --> NORM[Log(TPM) Normalization]
DE --> NORM

%% =======================
%% OPTIONAL BATCH CORRECTION
%% =======================

subgraph OPTIONAL_BC["Optional Batch Correction"]
    direction TB
    NORM --> BC[Batch Correction<br/>States:<br/>1. batch-only<br/>2. batch+design<br/>3. covariates<br/>4. covariates+design<br/>5. PCs<br/>6. PCs+design]
end
BC --> PREP[Preprocessed Matrix]
NORM -->|If disabled| PREP

%% =======================
%% HOLDOUT + LABELED DATA
%% =======================

subgraph LABELED["Labeled Training Data"]
    PREP --> H[Holdout Split (optional)]
    H --> IMB[Imbalance Assessment (mild/severe)]
end

%% =======================
%% MODEL ZOO
%% =======================

subgraph TRAIN["Model Training Pipeline"]
    
    IMB --> GSO[Gene-set Optimization<br/>(CV & surface-marker bootstrap)]
    GSO --> HPO[Hyperparameter Optimization<br/>(Nested CV)]
    
    subgraph MODELS["Explicit Model Variants (severity-ordered)"]
        direction TB
        M0[Random Baseline]
        M1[Random Forest]
        M2[RF Class-Weighted]
        M3[RF + SafeSMOTE]
        M4[RF + SafeBorderlineSMOTE]
        M5[RF + SafeSVMSMOTE]
        M6[RF + SafeADASYN]
        M7[XGBoost]
        M8[LightGBM]
        M9[Stacked Ensemble]
    end
    
    HPO --> M0 --> F
    HPO --> M1 --> F
    HPO --> M2 --> F
    HPO --> M3 --> F
    HPO --> M4 --> F
    HPO --> M5 --> F
    HPO --> M6 --> F
    HPO --> M7 --> F
    HPO --> M8 --> F
    HPO --> M9 --> F
    
    F[Final Calibrated Training]
end

F --> TOP[Top-Performing Model]

%% =======================
%% PREDICTION PHASE
%% =======================

subgraph PRED["Prediction Phase"]
    U[New Unlabeled Data] -->|Optional batch correction| PREP2[Preprocess New Data]
    TOP --> PREDOUT[Predictions + Probabilities]
    PREP2 --> PREDOUT
end

%% =======================
%% SURVIVAL ANALYSIS
%% =======================

subgraph SURV["Survival Analysis (Optional)"]
    PREDOUT --> KM[Kaplan–Meier Curves]
    PREDOUT --> COX[Cox PH Model]
    PREDOUT --> FP[Forest Plot]
    PREDOUT --> CH[Confidence Histogram]
    PREDOUT --> VP[Violin Plot per Class]
end
