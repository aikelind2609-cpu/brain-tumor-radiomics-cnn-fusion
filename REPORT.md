# Hybrid Radiomics + Deep CNN Feature Fusion for Brain Tumor Classification: A Comparative Study with Interpretability Analysis

**Author:** Aikel Indurkhya
**Course:** PG Biomedical Data Science (mid-semester project, 10–25% grade weight)
**Timeline:** 7 days
**Submission date:** April 29, 2026
**Repository:** https://github.com/aikelind2609-cpu/brain-tumor-radiomics-cnn-fusion

---

## Abstract

Brain tumor classification from MRI is a well-studied problem with abundant published methods, ranging from hand-crafted radiomics features to end-to-end deep CNNs. This project implements and rigorously compares three feature configurations on the Nickparvar Brain Tumor MRI dataset (7,200 T1-weighted scans, 4 classes): (1) hand-crafted radiomics features (47-dimensional), (2) frozen ResNet50 ImageNet features (2048-dimensional), and (3) fusion of both with feature selection. All three configurations achieve approximately 89.94% test accuracy with XGBoost classifiers, suggesting the dataset is near its performance ceiling for non-end-to-end methods. Fusion provides a marginal but consistent improvement in macro-F1 (0.8955 → 0.8965). Glioma classification remains the most challenging task (recall: 0.70) due to the use of whole-brain masks rather than tumor-specific ROIs. SHAP analysis reveals that different tumor types rely on different feature subsets — radiomics features dominate meningioma classification, while deep features dominate notumor and pituitary recognition. A case-based retrieval system using cosine similarity on the fused 300-dimensional feature space provides interpretable prediction validation by showing the 3 most similar training images for any input. The final system is deployed as a Streamlit web application.

**Keywords:** Brain tumor classification, radiomics, transfer learning, ResNet50, feature fusion, SHAP, case-based retrieval, MRI

---

## 1. Introduction

### 1.1 Motivation

Magnetic Resonance Imaging (MRI) is the primary diagnostic modality for brain tumor evaluation, but accurate sub-type classification (glioma vs meningioma vs pituitary tumor) requires expert radiological interpretation. Automated classification systems can serve as triage and second-opinion tools, particularly valuable in resource-limited settings.

The literature broadly considers two paradigms:
1. **Radiomics:** Hand-crafted features (intensity statistics, texture descriptors, shape descriptors) grounded in radiological domain knowledge.
2. **Deep learning:** End-to-end CNNs or transfer learning from pretrained backbones, which automatically learn discriminative features.

Hybrid approaches combining both have been explored since 2018. This project does not claim novelty in the methodology itself; rather, it provides a **literature-grounded comparative study** with rigorous ablation, interpretability analysis, and a deployed demo.

### 1.2 Research Questions

1. On the Nickparvar dataset, do deep features outperform radiomics features for brain tumor classification?
2. Does fusion of both feature types meaningfully outperform either alone?
3. Which features (and feature types) drive each tumor class's predictions?
4. Can prediction reliability be validated post-hoc using case-based retrieval?

### 1.3 Contributions

1. A complete, reproducible pipeline from raw MRI to classification with three trained models
2. A controlled ablation study (radiomics-only, deep-only, fused with multiple feature-selection regimes)
3. SHAP-based interpretability with per-class feature importance analysis
4. A novel (for student-project scope) case-based retrieval validator using cosine similarity in fused feature space
5. An interactive Streamlit demo combining all of the above

---

## 2. Related Work

The following references informed the methodology:

- **Aerts et al. (2014)** established radiomics as a quantitative imaging biomarker framework.
- **van Griethuysen et al. (2017)** released PyRadiomics, the standard library for radiomics extraction. (Used as conceptual reference; replaced with scikit-image + mahotas due to Windows installation issues — see Section 4.4.)
- **Lao et al. (2017)** demonstrated deep CNN features for glioblastoma survival prediction, validating ImageNet transfer learning for brain MRI.
- **Banerjee et al. (2024)** and the **Hybrid 2D/3D CNN architecture (PMC, 2025)** showed that radiomics + deep feature fusion has been actively explored for brain tumor classification, confirming that this approach is **established methodology** rather than novel research.
- **Lundberg & Lee (2017)** introduced SHAP values for unified model interpretability, used here for per-class analysis.
- **Raghu et al. (2019)** discussed limitations of ImageNet transfer learning for medical imaging — relevant to interpretation of `deep_897`'s dominance in our results.

---

## 3. Dataset

**Source:** Nickparvar Brain Tumor MRI Dataset (Kaggle)

**Composition:**
- Total images: 7,200
- Classes: glioma, meningioma, pituitary, notumor
- Per-class distribution: 1,800 each (balanced)
- Train/test split (publisher-provided): 5,600 / 1,600
- Per-class test set: 400 each
- Image size: All confirmed 512×512, grayscale

**Class definitions:**
- **Glioma:** Aggressive, infiltrative tumor of glial cells; highly heterogeneous in appearance
- **Meningioma:** Typically benign, well-circumscribed tumor of the meninges; often appears as a distinct mass
- **Pituitary tumor:** Located at the base of the brain; consistent anatomical location
- **Notumor:** Healthy brain MRI

---

## 4. Methodology

### 4.1 Day-by-Day Methodological Evolution

The project's methodology evolved as initial choices revealed limitations. The full evolution is documented for transparency:

| Day | Component | Initial Method | Final Method | Reason for Change |
|---|---|---|---|---|
| 1 | Setup | — | Conda env + Jupyter + git | Standard |
| 2 | Preprocessing | — | Brain crop + CLAHE + resize 224×224 | Required for ResNet50 |
| 3 | Mask generation | Tumor masks | Whole-brain masks | Dataset has no tumor segmentations |
| 4 | Radiomics library | PyRadiomics | scikit-image + mahotas | PyRadiomics installation failed on Windows (no wheels for Python 3.11) |
| 4 | Radiomics classifier | Random Forest | Random Forest | Baseline retained |
| 5 | Deep classifier | Random Forest (88.25% radiomics → 85.69% deep) | Switched experiments to LR (90.00%), then XGBoost in Day 6 | RF unsuitable for 2048-d features |
| 6 | All classifiers | Mixed (RF, LR) | XGBoost | Fair comparison required identical classifier |
| 6 | Feature selection (fusion) | k=80 (89.19%) | k=300 (89.94%, highest macro-F1) | k=80 too aggressive |
| 7 | Interpretability | — | SHAP TreeExplainer | Standard for tree models |
| 7 | Novelty | (Cross-dataset test was scoped but skipped) | Case-based retrieval (CBIR) | Higher payoff per hour |

### 4.2 Preprocessing Pipeline (Day 2)

For each input image:
1. **Brain crop:** Threshold at intensity 10, dilate-erode-largest-contour bounding box. Removes black background.
2. **Resize:** Bilinear interpolation to 224×224 (ResNet50 input requirement).
3. **CLAHE:** Contrast Limited Adaptive Histogram Equalization with `clipLimit=2.0`, `tileGridSize=(8,8)`. Improves contrast without overamplification.

All 7,200 images processed successfully; output stored in `data/processed/`.

### 4.3 Brain Mask Generation (Day 3)

Pipeline:
1. Otsu's threshold to binarize
2. Morphological closing (11×11 kernel, 3 iterations) to fill internal gaps
3. Morphological opening (5×5 kernel, 2 iterations) to remove noise
4. Connected components analysis; retain largest
5. Final morphological closing + contour fill for solid interior

**Limitation:** Masks cover the whole brain, not the tumor specifically, because the dataset provides no tumor segmentations. This averaging effect is later identified as a key limitation for glioma performance (Section 5.4).

### 4.4 Radiomics Feature Extraction (Day 4)

**Library substitution:** PyRadiomics could not be installed on Windows + Python 3.11 (no precompiled wheels, conda-forge build unavailable, Visual C++ Build Tools dependency too heavy for the project timeline). Replaced with:
- **scikit-image:** First-order statistics, GLCM (Gray-Level Co-occurrence Matrix), shape descriptors via `regionprops`
- **mahotas:** Haralick texture features

Total: **47 features per image**, organized as:
- 13 first-order: mean, std, min, max, median, p10, p90, skewness, kurtosis, energy, entropy, range, IQR (`fo_*`)
- 12 GLCM: contrast, homogeneity, energy, correlation, dissimilarity, ASM (each as mean and std across distances/angles) (`glcm_*`)
- 13 Haralick: standard set from mahotas (`har_*`)
- 9 shape: area, perimeter, eccentricity, solidity, extent, major/minor axis length, circularity, aspect ratio (`sh_*`)

Output: `features/radiomics.csv` (5.87 MB).

**Day 4 baseline (RF, n=300):** 88.25% test accuracy.

### 4.5 Deep CNN Feature Extraction (Day 5)

- **Backbone:** ResNet50, pretrained on ImageNet (weights: `IMAGENET1K_V2`)
- **Modification:** Replace final FC layer with `nn.Identity()` to expose 2048-d penultimate-layer features
- **Mode:** `eval()`, `requires_grad=False` for all parameters (frozen)
- **Input prep:** Resize 224×224, replicate grayscale to 3 channels, normalize with ImageNet mean/std
- **Inference:** Batch size 32, CPU-only, ~10 minutes for all 7,200 images

Output: `features/deep.csv` (~80–120 MB; gitignored, manually backed up to Drive).

**Day 5 baselines (deep features alone):**
| Classifier | Accuracy |
|---|---|
| Random Forest (n=300) | 85.69% |
| RF on PCA(100)-projected features | 86.44% |
| Logistic Regression (C=0.1) | **90.00%** |

**Methodological finding:** Random Forest performs poorly on raw 2048-d deep features because tree splits on individual features waste capacity on noise. Logistic Regression and gradient boosting handle high-dimensional features much better.

### 4.6 Feature Fusion and Classification (Day 6)

**Decision:** Use **XGBoost** as the unified classifier across all three configurations for a fair comparison.

XGBoost hyperparameters (held constant):
- `n_estimators=400`, `max_depth=6`, `learning_rate=0.08`
- `subsample=0.9`, `colsample_bytree=0.8`
- `objective='multi:softprob'`, `num_class=4`, `eval_metric='mlogloss'`
- `random_state=42`

**Fusion strategy:** Concatenate radiomics (47-d) + deep (2048-d) → 2095-d, then apply `StandardScaler` and `SelectKBest` with the ANOVA F-statistic.

**Feature-selection ablation:**

| k | Accuracy | Macro-F1 |
|---|---|---|
| 80 | 89.19% | 0.8884 |
| 200 | 89.56% | 0.8923 |
| 300 (chosen) | **89.94%** | **0.8965** |
| All 2095 | 89.94% | 0.8960 |

`k=300` was selected as the official fused model: highest macro-F1, more interpretable than no-selection, more robust than aggressive selection.

**Selected features in final model:** 36/47 radiomics + 264/2048 deep features.

### 4.7 Interpretability Analysis (Day 7, Phase 1)

- **Method:** SHAP `TreeExplainer` (exact and fast for XGBoost)
- **Output shape:** (1600 samples, 300 features, 4 classes)
- **Visualizations:** Global mean-|SHAP| bar plot, per-class top-10 bar plots (2×2 grid)

### 4.8 Case-Based Retrieval (Day 7, Phase 3 — Novelty)

**Motivation:** Black-box predictions at 90% accuracy are fine but not auditable. Radiologists naturally reason by analogy ("this case looks like cases I've seen before"). A case-based retrieval system mirrors this clinical workflow.

**Implementation:**
1. Pre-compute the fused 300-d feature representation of all 5,600 training images, stored as `models/similarity_index.pkl`
2. At inference, project the input image into the same 300-d space
3. Compute cosine similarity to all training cases
4. Return top-3 most similar with their ground-truth class labels

**Retrieval-prediction agreement** is then used as a calibration signal:
- 3/3 retrieved cases match prediction → green flag, well-supported
- 2/3 match → yellow flag, partially supported
- 0/3 match → red flag, prediction likely unreliable

This is implemented in the Streamlit demo and provides an interpretable validation layer on top of the classifier.

### 4.9 Streamlit Web Demo (Day 7, Phase 2)

A single-file Streamlit app (`app/streamlit_app.py`) provides:
- Image upload interface
- Live preprocessing visualization (brain crop, CLAHE, brain mask)
- Predictions from all three models (radiomics-only, deep-only, fused)
- Confidence indicator (high/medium/low with radiologist-review flag at <65%)
- Feature contribution table (top-10 features by `|importance × value|`)
- Case-based retrieval with retrieved-image display and agreement indicator
- Detailed probability table

---

## 5. Results

### 5.1 Final Comparison Table

| Model | Features | Accuracy | Macro-F1 |
|---|---|---|---|
| Radiomics only (XGBoost) | 47 | 89.94% | 0.8955 |
| Deep only (XGBoost) | 2048 | 89.69% | 0.8939 |
| **Fused + SelectKBest(300) + XGBoost** | **300** | **89.94%** | **0.8965** |

### 5.2 Per-Class Performance (Fused Model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Glioma | 0.97 | 0.70 | 0.81 | 400 |
| Meningioma | 0.80 | 0.94 | 0.86 | 400 |
| Notumor | 0.90 | 1.00 | 0.95 | 400 |
| Pituitary | 0.96 | 0.96 | 0.96 | 400 |

**Glioma confusion breakdown:**
- 280/400 correctly identified
- 78 misclassified as meningioma
- 39 misclassified as notumor (clinically concerning false-negative)
- 3 misclassified as pituitary

### 5.3 Comparison Across All Day-Level Experiments

| Day | Method | Accuracy |
|---|---|---|
| 4 | Radiomics + RF | 88.25% |
| 5 | Deep + RF | 85.69% |
| 5 | Deep + PCA(100) + RF | 86.44% |
| 5 | Deep + Logistic Regression | 90.00% |
| 6 | Radiomics + XGBoost | 89.94% |
| 6 | Deep + XGBoost | 89.69% |
| 6 | Fused + SelectKBest(80) + XGBoost | 89.19% |
| 6 | Fused + SelectKBest(200) + XGBoost | 89.56% |
| 6 | **Fused + SelectKBest(300) + XGBoost (FINAL)** | **89.94%** |
| 6 | Fused (no selection) + XGBoost | 89.94% |

### 5.4 Feature Importance (XGBoost Gain)

Top 5 features in the fused model:

| Rank | Feature | Type | Importance |
|---|---|---|---|
| 1 | `deep_897` | Deep | 11.7% |
| 2 | `fo_median` | Radiomics | 3.7% |
| 3 | `deep_762` | Deep | 3.2% |
| 4 | `deep_1632` | Deep | 2.8% |
| 5 | `har_contrast` | Radiomics | 2.2% |

**Top 20 split:** Approximately 10 radiomics + 10 deep features — balanced contribution despite `deep_897`'s individual dominance.

### 5.5 SHAP Per-Class Top Features

| Class | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Glioma | `fo_median` (radiomics) | `fo_skew` (radiomics) | `deep_533` (deep) |
| Meningioma | `glcm_homogeneity_std` (radiomics) | `har_info_correlation_2` (radiomics) | `glcm_correlation_mean` (radiomics) |
| Notumor | `deep_897` (deep) | `deep_1244` (deep) | `deep_1088` (deep) |
| Pituitary | `deep_762` (deep) | `deep_1632` (deep) | `deep_679` (deep) |

**Key observation:** Meningioma classification is almost entirely driven by radiomics texture features (homogeneity, correlation), while notumor and pituitary are entirely driven by deep features. Glioma uses a mix. This validates the fusion hypothesis: different classes benefit from different feature types, and combining them captures complementary information.

---

## 6. Discussion

### 6.1 Why Fusion Provides Only Marginal Improvement

Three plausible explanations:

1. **Dataset ceiling.** Both feature types independently achieve ~90%. The remaining 10% error is concentrated in glioma cases (Section 5.2), where fusion does not provide additional discriminative information beyond what each feature type already captures.
2. **Redundant information.** ResNet50, even though pretrained on natural images, learns texture-like features in early-to-middle layers that overlap conceptually with GLCM/Haralick descriptors. The two feature sets are not as orthogonal as initially hypothesized.
3. **Whole-brain ROI averaging.** Both feature types operate on the whole brain. Tumor-specific ROIs would likely amplify class-discriminative information that whole-brain averaging dilutes.

This is not a failure but a real finding consistent with mixed results in the literature on radiomics-deep fusion.

### 6.2 Why Glioma Is the Hardest Class

Three contributing factors:

1. **Heterogeneity:** Gliomas vary widely in size, shape, location, and intensity. Unlike meningiomas (compact, surface-located, hyperintense) or pituitary tumors (consistent location), gliomas have no single visual signature.
2. **Whole-brain masks dilute glioma-specific signal.** A small glioma in a large brain has its texture statistics averaged with healthy tissue.
3. **Glioma↔meningioma confusion** dominates the error pattern (78 cases). Both are tumor classes with similar bulk-texture properties when measured on whole-brain ROIs.

### 6.3 The `deep_897` Phenomenon

A single deep feature accounts for ~12% of the model's predictive importance. Two interpretations:

1. **Genuine class signal.** ResNet50 may have learned a high-level pattern that strongly distinguishes one class (likely notumor, given the SHAP per-class analysis).
2. **Dataset-specific overfitting.** This feature might not generalize. Stability analysis across CV folds and different backbones (EfficientNet, DenseNet) would clarify this.

This is flagged as a limitation rather than treated as a finding.

### 6.4 Methodological Lessons

1. **Match classifier to feature dimensionality.** Random Forest works well for low-dim radiomics but is wasteful for 2048-d deep features. Use XGBoost or LR/SVM for high-dimensional inputs.
2. **Aggressive feature selection can hurt.** SelectKBest(80) on a 2095-d input dropped accuracy by 0.75%. Moderate selection (k=300) or no selection performed equivalently.
3. **Honest baselines matter.** Initial Day 4 RF baseline (88.25%) and the methodology evolution to k=300 XGBoost (89.94%) is only +1.69% absolute. This is not a dramatic improvement; reporting it honestly is more valuable than overselling.

---

## 7. Limitations

1. **Whole-brain masks** rather than tumor-specific ROIs — the most significant limitation.
2. **Single dataset.** No cross-dataset validation; potential dataset-specific artifacts cannot be distinguished from real biological signal.
3. **Frozen ResNet50.** No fine-tuning; ImageNet features may not be optimal for medical imaging.
4. **`deep_897` stability** not investigated.
5. **CPU-only.** Limits scalability; precluded end-to-end CNN training.
6. **Class definitions.** All test images come from the publisher-provided test split. Acquisition-protocol bias (single-source data) is uncorrected.

---

## 8. Future Work

1. **Tumor segmentation** (nnU-Net, SAM, or even threshold-based heuristics) to replace whole-brain masks. Expected to dramatically improve glioma performance.
2. **Cross-dataset validation** on BraTS, Cheng J. Figshare, and Sartaj brain tumor datasets.
3. **End-to-end fine-tuning** of ResNet50 on medical imaging (e.g., MedicalNet pretrained weights).
4. **Stacked ensemble** as alternative fusion: train radiomics-XGBoost and deep-XGBoost separately, then a logistic-regression meta-learner on their probability outputs.
5. **Calibration analysis** (ECE, reliability diagrams) of the confidence scores; isotonic regression calibration if needed.
6. **Stability analysis** of `deep_897` across CV folds and across backbones (EfficientNet-B0, DenseNet-121, MedicalNet).
7. **Larger CBIR system** with FAISS for million-scale retrieval if expanded to a multi-hospital deployment.

---

## 9. Conclusions

This project implements a complete brain tumor classification pipeline combining hand-crafted radiomics features (47-dimensional) and frozen ResNet50 deep features (2048-dimensional). All three classifier configurations — radiomics-only, deep-only, and fused — converge to approximately 89.94% test accuracy with XGBoost, with fusion providing a marginal but consistent macro-F1 improvement (0.8955 → 0.8965). The dataset appears near its performance ceiling for non-end-to-end methods.

Glioma classification remains the most challenging task (recall: 0.70), attributed to the use of whole-brain masks rather than tumor-specific ROIs. SHAP analysis demonstrates that different tumor classes rely on different feature subsets — meningioma is driven entirely by radiomics texture features, while notumor and pituitary tumors are driven entirely by deep features. This validates the fusion hypothesis at a feature-importance level even though aggregate accuracy gains are small.

The deployed Streamlit application provides an end-to-end demonstration with case-based retrieval as a novel post-hoc validation mechanism, addressing a gap in standard ML pipelines: the inability to audit individual predictions against training data.

The project's primary scientific contribution is **methodological transparency** — careful ablation across classifiers and feature-selection regimes, honest reporting of small effect sizes, and explicit documentation of design decisions and their motivations.

---

## 10. References

1. Aerts, H. J. W. L., Velazquez, E. R., Leijenaar, R. T. H., et al. (2014). *Decoding tumor phenotype by noninvasive imaging using a quantitative radiomics approach.* Nature Communications, 5, 4006.
2. van Griethuysen, J. J. M., Fedorov, A., Parmar, C., et al. (2017). *Computational Radiomics System to Decode the Radiographic Phenotype.* Cancer Research, 77(21), e104–e107.
3. Lao, J., Chen, Y., Li, Z.-C., et al. (2017). *A Deep Learning-Based Radiomics Model for Prediction of Survival in Glioblastoma Multiforme.* Scientific Reports, 7, 10353.
4. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems, 30.
5. Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S. (2019). *Transfusion: Understanding Transfer Learning for Medical Imaging.* NeurIPS, 32.
6. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16, 785–794.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR, 770–778.
8. Banerjee, S. et al. (2024). *Hybrid radiomics-deep learning approaches for brain tumor classification.* (Cited as conceptual support for the fusion methodology being established.)

---

## Appendix A: Project Master Record

**Repository:** https://github.com/aikelind2609-cpu/brain-tumor-radiomics-cnn-fusion
**Local path:** `C:\Users\HP\Desktop\MU\bmi\brain_tumor_fusion`
**Environment:** Python 3.11.9, conda env `brain_tumor`, no GPU
**Notebooks:** 9 sequential Jupyter notebooks, `01_explore` through `09_build_similarity_index`
**Models saved:** `radiomics.pkl` (4.0 MB), `deep.pkl` (3.4 MB), `fused.pkl` (3.5 MB), `similarity_index.pkl` (~13 MB)
**Output figures:** `confusion_matrices.png`, `feature_importance.png`, `shap_global.png`, `shap_per_class.png`, `sample_grid.png`, `mask_qa.png`, `preprocess_qa.png`, `crop_qa.png`, `mask_random_samples.png`, `demo_3model_comparison.png`
**Demo app:** `app/streamlit_app.py` (Streamlit, runs locally on port 8501)