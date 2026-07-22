#!/usr/bin/env python3
"""Build the source-verified SynTrustBench v0.1.0 annotation CSV.

The row definitions below are the adjudication-ready source layer. The emitted
CSV is the public interchange artifact consumed by score_evidence.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


COLUMNS = [
    "report_id", "citation_key", "title", "first_author", "year", "venue",
    "publication_type", "peer_reviewed", "primary_source_url", "doi_or_identifier",
    "modality", "task_scope", "evaluated_systems", "architecture_family", "model_name",
    "dataset_names", "dataset_versions", "data_access", "code_available", "code_url",
    "fidelity_evaluated", "fidelity_metrics", "clinical_validity_evaluated",
    "clinical_validity_method", "utility_evaluated", "utility_metrics",
    "held_out_real_test", "privacy_evaluated", "privacy_test_type", "threat_model",
    "membership_inference", "attribute_inference", "reidentification_or_reconstruction",
    "dp_claimed", "formal_dp_verified", "dp_evidence_status", "dp_mechanism",
    "epsilon_reported", "epsilon_values", "delta_reported", "delta_values",
    "fairness_evaluated", "protected_attributes", "subgroup_metrics",
    "robustness_evaluated", "external_validation", "seeds_or_repeats",
    "uncertainty_reported", "fidelity_maturity", "utility_maturity", "privacy_maturity",
    "equity_maturity", "robustness_maturity", "evaluability_gate", "verification_status",
    "verifier", "verified_date", "evidence_notes", "ambiguity_flag", "eligibility_status",
    "exclusion_reason", "mimic_iii_used",
]

BINARY_COLUMNS = {
    "fidelity_evaluated", "clinical_validity_evaluated", "utility_evaluated",
    "privacy_evaluated", "membership_inference", "attribute_inference",
    "reidentification_or_reconstruction", "dp_claimed", "formal_dp_verified",
    "epsilon_reported", "delta_reported", "fairness_evaluated",
    "robustness_evaluated", "external_validation", "uncertainty_reported",
    "ambiguity_flag", "mimic_iii_used",
}

ENUM_COLUMNS = {
    "publication_type": ["Journal article", "Conference paper", "Workshop paper", "Preprint", "Thesis", "Other"],
    "peer_reviewed": ["Yes", "No", "Unclear"],
    "modality": ["Tabular EHR", "Clinical time series", "Medical imaging", "Clinical text", "Multimodal/other"],
    "task_scope": ["Generator", "Generator plus benchmark", "Benchmark/evaluation framework", "Augmentation", "Translation/reconstruction", "Other"],
    "architecture_family": ["GAN", "Diffusion", "LLM", "VAE", "Hybrid/other", "Kernel/marginal/other", "Evaluation framework"],
    "data_access": ["Public", "Credentialed", "Private", "Mixed", "NR"],
    "code_available": ["Yes", "No", "NR"],
    "held_out_real_test": ["Yes", "No", "Unclear"],
    "dp_evidence_status": ["Complete", "Incomplete", "Not a DP claim"],
    "evaluability_gate": ["Pass", "Conditional", "Fail"],
    "eligibility_status": ["Included", "Context-only", "Excluded"],
}


def annotation_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        column: {"type": "string", "minLength": 1} for column in COLUMNS
    }
    for column in BINARY_COLUMNS:
        properties[column] = {"enum": ["Yes", "No"]}
    for column, values in ENUM_COLUMNS.items():
        properties[column] = {"enum": values}
    properties["report_id"] = {"type": "string", "pattern": "^STB-[0-9]{3}$"}
    properties["citation_key"] = {"type": "string", "pattern": "^[a-z0-9]+$"}
    properties["year"] = {"type": "integer", "minimum": 2018, "maximum": 2026}
    properties["primary_source_url"] = {"type": "string", "format": "uri"}
    properties["verified_date"] = {"type": "string", "format": "date"}
    for column in (
        "fidelity_maturity", "utility_maturity", "privacy_maturity",
        "equity_maturity", "robustness_maturity",
    ):
        properties[column] = {"type": "integer", "minimum": 0, "maximum": 4}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SynTrustBench v0.1.0 study annotation row",
        "description": "One source-audited report in the frozen pilot corpus.",
        "type": "object",
        "required": COLUMNS,
        "properties": properties,
        "additionalProperties": False,
    }

COMMON = {
    "dataset_versions": "NR",
    "code_available": "No",
    "code_url": "NA",
    "clinical_validity_evaluated": "No",
    "clinical_validity_method": "NR",
    "privacy_evaluated": "No",
    "privacy_test_type": "None",
    "threat_model": "NR",
    "membership_inference": "No",
    "attribute_inference": "No",
    "reidentification_or_reconstruction": "No",
    "dp_claimed": "No",
    "formal_dp_verified": "No",
    "dp_evidence_status": "Not a DP claim",
    "dp_mechanism": "NA",
    "epsilon_reported": "No",
    "epsilon_values": "NA",
    "delta_reported": "No",
    "delta_values": "NA",
    "fairness_evaluated": "No",
    "protected_attributes": "NR",
    "subgroup_metrics": "NR",
    "robustness_evaluated": "No",
    "external_validation": "No",
    "seeds_or_repeats": "NR",
    "uncertainty_reported": "No",
    "equity_maturity": 0,
    "robustness_maturity": 0,
    "verification_status": "Primary source verified; human double-code pending",
    "verifier": "N.S.H. original extraction; Codex source check",
    "verified_date": "2026-07-21",
    "ambiguity_flag": "No",
    "eligibility_status": "Included",
    "exclusion_reason": "NA",
    "mimic_iii_used": "No",
}

rows: list[dict[str, object]] = []


def add(**values: object) -> None:
    row = {**COMMON, **values}
    missing = [column for column in COLUMNS if column not in row]
    if missing:
        raise ValueError(f"{row.get('report_id', '<unknown>')}: missing {missing}")
    rows.append({column: row[column] for column in COLUMNS})


add(
    report_id="STB-001", citation_key="jordon2019pategan",
    title="PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees",
    first_author="Jordon", year=2019, venue="ICLR 2019", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://openreview.net/forum?id=S1zk9iRqF7",
    doi_or_identifier="OpenReview S1zk9iRqF7", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="PATE-GAN", architecture_family="GAN", model_name="PATE-GAN",
    dataset_names="Kaggle Credit Card Fraud; MAGGIC heart failure; UNOS transplant; Kaggle Cervical Cancer; UCI ISOLET; UCI Epileptic Seizure Recognition",
    data_access="Mixed", code_available="Yes",
    code_url="https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/pategan",
    fidelity_evaluated="Yes", fidelity_metrics="Synthetic Ranking Agreement; variable-importance ranking preservation",
    utility_evaluated="Yes", utility_metrics="TSTR AUROC; AUPRC across 12 classifiers",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Formal differential privacy",
    threat_model="PATE teacher partitions with noisy aggregation; record-level adjacency described in method",
    dp_claimed="Yes", formal_dp_verified="Yes", dp_evidence_status="Complete",
    dp_mechanism="PATE", epsilon_reported="Yes", epsilon_values="1 primary; 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50 sensitivity",
    delta_reported="Yes", delta_values="1e-5", robustness_evaluated="Yes",
    seeds_or_repeats="12 downstream classifiers; epsilon sensitivity analysis", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=4, robustness_maturity=2,
    evaluability_gate="Conditional",
    evidence_notes="Primary evaluation uses six datasets, not Adult/Health/MNIST. Formal (epsilon=1, delta=1e-5)-DP comparison and epsilon sweep; no empirical attack or equity analysis."
)

add(
    report_id="STB-002", citation_key="torfi2020corgan",
    title="CorGAN: Correlation-Capturing Convolutional Generative Adversarial Networks for Generating Synthetic Healthcare Records",
    first_author="Torfi", year=2020, venue="FLAIRS-33", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://cdn.aaai.org/ocs/18458/18458-79398-1-PB.pdf",
    doi_or_identifier="FLAIRS-33 proceedings", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="CorGAN", architecture_family="GAN", model_name="CorGAN",
    dataset_names="MIMIC-III; UCI Epileptic Seizure Recognition",
    dataset_versions="MIMIC-III version NR; UCI Epileptic 11,500 segments/178 features",
    data_access="Mixed", code_available="Yes", code_url="https://github.com/astorfi/cor-gan",
    fidelity_evaluated="Yes", fidelity_metrics="Dimension-wise Bernoulli probabilities; dimension-wise prediction",
    utility_evaluated="Yes", utility_metrics="TSTR seizure AUROC; AUPRC; F1-related target-variable experiments",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Cosine-similarity membership/record-matching experiment",
    threat_model="Attacker knows synthetic records and compromised subsets of real train/test records; similarity thresholds varied",
    membership_inference="Yes", seeds_or_repeats="Repeated target-variable experiments", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=2, evaluability_gate="Conditional",
    mimic_iii_used="Yes",
    evidence_notes="Critical correction: no formal DP, epsilon, delta, MMD, MIT-BIH, or PTB. Privacy evidence is an empirical cosine-similarity attack."
)

add(
    report_id="STB-003", citation_key="beyki2021scorgan",
    title="Synthetic Electronic Medical Record Generation using Generative Adversarial Networks",
    first_author="Beyki", year=2021, venue="Virginia Tech", publication_type="Thesis",
    peer_reviewed="No", primary_source_url="http://hdl.handle.net/10919/104642",
    doi_or_identifier="Virginia Tech handle 10919/104642", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="SCorGAN; Improved Correlation Capturing Wasserstein GAN", architecture_family="GAN",
    model_name="SCorGAN", dataset_names="MIMIC-III; UCI Epileptic Seizure Recognition",
    data_access="Mixed", code_available="Yes", code_url="https://github.com/mohibeyki/SCorGAN",
    fidelity_evaluated="Yes", fidelity_metrics="Dimension-wise probabilities; kernel MMD",
    utility_evaluated="Yes", utility_metrics="TSTR AUC; precision; recall across six classifiers",
    held_out_real_test="Yes", privacy_evaluated="No", privacy_test_type="None",
    seeds_or_repeats="Five MMD runs", uncertainty_reported="Yes",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=0, evaluability_gate="Conditional",
    mimic_iii_used="Yes",
    evidence_notes="The thesis reproduces CorGAN privacy material in an earlier chapter, but the novel SCorGAN experiment has no separate privacy audit. It is not peer reviewed."
)

add(
    report_id="STB-004", citation_key="harder2021dpmerf",
    title="DP-MERF: Differentially Private Mean Embeddings with Random Features for Practical Privacy-Preserving Data Generation",
    first_author="Harder", year=2021, venue="AISTATS 2021; PMLR 130", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://proceedings.mlr.press/v130/harder21a.html",
    doi_or_identifier="PMLR 130:1819-1827", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="DP-MERF", architecture_family="Kernel/marginal/other", model_name="DP-MERF",
    dataset_names="ISOLET; Covertype; Epileptic; Credit; Cervical; Census; Adult; Intrusion; MNIST; Fashion-MNIST; toy Gaussian mixture",
    data_access="Public", code_available="Yes", code_url="https://github.com/ParkLabML/DP-MERF",
    fidelity_evaluated="Yes", fidelity_metrics="Random-feature kernel MMD; mode coverage; negative log-likelihood; qualitative images",
    utility_evaluated="Yes", utility_metrics="TSTR ROC-AUC; PR-AUC; accuracy; F1; image classification",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Formal differential privacy",
    threat_model="Gaussian-mechanism release of a bounded random-feature mean followed by post-processing",
    dp_claimed="Yes", formal_dp_verified="Yes", dp_evidence_status="Complete", dp_mechanism="Gaussian mechanism",
    epsilon_reported="Yes", epsilon_values="1 primary; 0.2 for image sensitivity; some comparisons near 9.6 or 10",
    delta_reported="Yes", delta_values="1e-5", robustness_evaluated="Yes",
    seeds_or_repeats="Five independent runs; epsilon sensitivity", uncertainty_reported="Yes",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=4, robustness_maturity=2,
    evaluability_gate="Conditional",
    evidence_notes="Non-GAN method. Privacy is guaranteed by the privatized mean embedding; no empirical disclosure attack or equity analysis."
)

add(
    report_id="STB-005", citation_key="ceritli2023tabddpmehr",
    title="Synthesizing Mixed-type Electronic Health Records using Diffusion Models",
    first_author="Ceritli", year=2023, venue="arXiv", publication_type="Preprint",
    peer_reviewed="No", primary_source_url="https://arxiv.org/abs/2302.14679",
    doi_or_identifier="arXiv:2302.14679", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="TabDDPM; GAN/VAE baselines", architecture_family="Diffusion", model_name="TabDDPM EHR",
    dataset_names="MIMIC-III; Pima Indians Diabetes; ILPD; Kaggle Stroke",
    data_access="Mixed", code_available="No", code_url="NA",
    fidelity_evaluated="Yes", fidelity_metrics="Dimension-wise probability; target-variable F1; kernel MMD",
    utility_evaluated="Yes", utility_metrics="TSTR and augmentation accuracy; AUROC; AUPRC; F1 across seven classifiers",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Distance to closest record; threshold membership inference",
    threat_model="Euclidean distance to nearest synthetic record with F1-based member classification; MIMIC-focused",
    membership_inference="Yes", seeds_or_repeats="Five generation seeds; ten classifier seeds", uncertainty_reported="Yes",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=2, evaluability_gate="Conditional",
    mimic_iii_used="Yes",
    evidence_notes="Preprint, not peer-reviewed in the coded version. Code was promised on acceptance but no matching release was verified."
)

add(
    report_id="STB-006", citation_key="chen2025synthehrella",
    title="Generating synthetic electronic health record data: a methodological scoping review with benchmarking on phenotype data and open-source software",
    first_author="Chen", year=2025, venue="Journal of the American Medical Informatics Association 32(7):1227-1240",
    publication_type="Journal article", peer_reviewed="Yes",
    primary_source_url="https://doi.org/10.1093/jamia/ocaf082", doi_or_identifier="10.1093/jamia/ocaf082",
    modality="Tabular EHR", task_scope="Benchmark/evaluation framework",
    evaluated_systems="Resample/PBR; Plasmode; Synthea; MedGAN; CorGAN; VAE; PromptEHR; EHRDiff; baseline",
    architecture_family="Evaluation framework", model_name="SynthEHRella",
    dataset_names="MIMIC-III; MIMIC-IV", data_access="Credentialed", code_available="Yes",
    code_url="https://github.com/chenxran/synthEHRella",
    fidelity_evaluated="Yes", fidelity_metrics="Maximum marginal discrepancy; RMSPE; MAPE; correlation Frobenius distance; discriminator AUC/accuracy",
    utility_evaluated="Yes", utility_metrics="Coefficient/95% CI agreement; TRTR; TSTR; TSRTR predictive AUC and accuracy",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Membership risk; exact matches; attribute inference",
    threat_model="Nearest Euclidean distance/exact matches; 1-NN attribute inference from partially known fields",
    membership_inference="Yes", attribute_inference="Yes", robustness_evaluated="Yes", external_validation="Yes",
    seeds_or_repeats="Five-fold discriminator CV; MIMIC-III to MIMIC-IV transport", uncertainty_reported="Yes",
    fidelity_maturity=4, utility_maturity=4, privacy_maturity=2, robustness_maturity=3,
    evaluability_gate="Conditional", mimic_iii_used="Yes",
    evidence_notes="Peer-reviewed 2025 JAMIA version supersedes the preprint coding. This is a scoping review plus benchmark, not a GAN generator."
)

add(
    report_id="STB-007", citation_key="ramachandranpillai2024btgan",
    title="Bt-GAN: Generating Fair Synthetic Healthdata via Bias-transforming Generative Adversarial Networks",
    first_author="Ramachandranpillai", year=2024, venue="Journal of Artificial Intelligence Research 79:1313-1341",
    publication_type="Journal article", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1613/jair.1.15317",
    doi_or_identifier="10.1613/jair.1.15317", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="Bt-GAN; GAN baselines", architecture_family="GAN", model_name="Bt-GAN",
    dataset_names="MIMIC-III", data_access="Credentialed",
    fidelity_evaluated="Yes", fidelity_metrics="Discriminative score; Jensen-Shannon divergence; alpha precision; beta recall; authenticity; Context FID",
    utility_evaluated="Yes", utility_metrics="TSTR AUROC; AUPRC; accuracy; F1 for mortality and length-of-stay tasks",
    held_out_real_test="Yes", fairness_evaluated="Yes", protected_attributes="Race/ethnicity (White versus Black)",
    subgroup_metrics="Parity gap; AUROC gap; statistical parity; representation fairness/LDS; bias amplification",
    seeds_or_repeats="Mean and standard deviation reported for main comparisons", uncertainty_reported="Yes",
    fidelity_maturity=3, utility_maturity=3, privacy_maturity=0, equity_maturity=3,
    evaluability_gate="Conditional", mimic_iii_used="Yes",
    evidence_notes="The paper's 'data/model leakage' measures ethnicity predictability and bias amplification, not patient privacy leakage. No MIA, AIA, DP, epsilon, or delta."
)

add(
    report_id="STB-008", citation_key="hao2024llmsyn",
    title="LLMSYN: Generating Synthetic Electronic Health Records Without Patient-Level Data",
    first_author="Hao", year=2024, venue="MLHC 2024; PMLR 252", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://proceedings.mlr.press/v252/hao24a.html",
    doi_or_identifier="PMLR 252", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="Eight LLMs and ablations", architecture_family="LLM", model_name="LLMSYN",
    dataset_names="MIMIC-III", data_access="Credentialed",
    fidelity_evaluated="Yes", fidelity_metrics="Feature-wise KS; joint-distribution MMD; mortality and diagnosis distributions",
    utility_evaluated="Yes", utility_metrics="Random-forest TSTR/augmentation; phenotype accuracy; mortality AUROC; F1",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="k-anonymity violation counts",
    threat_model="Ethnicity and ICD-9 quasi-identifiers at k=7 and k=15; authors call it a lower-bound assessment",
    seeds_or_repeats="Eight LLMs; six main synthetic sets; ablations", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=1, evaluability_gate="Conditional",
    mimic_iii_used="Yes",
    evidence_notes="Avoids patient-level prompting but does not prove privacy. k-anonymity counts are proxy evidence, not a membership attack or DP guarantee."
)

add(
    report_id="STB-009", citation_key="kotal2024contextgan",
    title="Differentially Private Synthetic Data Generation Using Context-Aware GANs",
    first_author="Kotal", year=2024, venue="IEEE International Conference on Big Data 2024",
    publication_type="Conference paper", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1109/BigData62323.2024.10826047",
    doi_or_identifier="10.1109/BigData62323.2024.10826047", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="ContextGAN; baselines", architecture_family="GAN", model_name="ContextGAN",
    dataset_names="PIMA Diabetes; UCI Heart Disease; UNSW-NB15; CICIDS2017; UCI Credit Default; German Credit",
    data_access="Public", fidelity_evaluated="Yes", fidelity_metrics="Earth Mover/Wasserstein distance; categorical L1 plus continuous L2 distance",
    utility_evaluated="Yes", utility_metrics="TSTR-like accuracy; precision; recall; F1 using RF, XGBoost, LR",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Re-identification matcher; model inversion/attribute inference; white-/black-box membership inference; claimed DP",
    threat_model="Partial-overlap re-identification; black-box model inversion; white- and black-box membership attacks",
    membership_inference="Yes", attribute_inference="Yes", reidentification_or_reconstruction="Yes",
    dp_claimed="Yes", formal_dp_verified="No", dp_evidence_status="Incomplete",
    dp_mechanism="Claimed DP-SGD-like discriminator training", epsilon_reported="No", epsilon_values="NR",
    delta_reported="No", delta_values="NR", seeds_or_repeats="Six datasets; seed protocol NR", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=2, evaluability_gate="Conditional",
    evidence_notes="The algorithm names epsilon, delta, clipping C, and noise sigma but gives no numeric parameters, sampling rate, accountant, or auditable per-example clipping. DP claim is not verified."
)

add(
    report_id="STB-010", citation_key="hu2025synqp",
    title="SynQP: A Framework and Metrics for Evaluating the Quality and Privacy Risk of Synthetic Data",
    first_author="Hu", year=2025, venue="IEEE Conference on Privacy, Security and Trust 2025",
    publication_type="Conference paper", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1109/PST65910.2025.11268831",
    doi_or_identifier="10.1109/PST65910.2025.11268831", modality="Tabular EHR",
    task_scope="Benchmark/evaluation framework", evaluated_systems="CTGAN; TVAE; GaussianCopula; custom noise variants",
    architecture_family="Evaluation framework", model_name="SynQP", dataset_names="Simulated pseudo-identifiable diabetes/BMI cohort informed by NIDDK",
    dataset_versions="N=10,000; 7,000 train; 3,000 holdout; 10,000 synthetic", data_access="Public",
    code_available="Yes", code_url="https://github.com/CAN-SYNH/SynQP",
    fidelity_evaluated="Yes", fidelity_metrics="Hellinger distance across columns",
    utility_evaluated="Yes", utility_metrics="Logistic-regression TSTR AUC", held_out_real_test="Yes",
    privacy_evaluated="Yes", privacy_test_type="SD-IDR; SD-MIA; invalid local-DP-style interpolation claim",
    threat_model="Released synthetic data plus 3,000 known population/holdout rows; output-only black-box risk",
    membership_inference="Yes", reidentification_or_reconstruction="Yes", dp_claimed="Yes",
    formal_dp_verified="No", dp_evidence_status="Incomplete", dp_mechanism="Interpolation with Laplace draw; not sensitivity-calibrated DP",
    epsilon_reported="No", epsilon_values="0.8 is a mixing coefficient, not a privacy budget", delta_reported="No", delta_values="NA",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=2, evaluability_gate="Pass",
    evidence_notes="Framework, not GAN. Its epsilon=0.8 is an interpolation/noise coefficient; no adjacency, sensitivity calibration, composition, accountant, or delta establishes DP."
)

add(
    report_id="STB-011", citation_key="kita2025multimorbidity",
    title="Innovative synthetic EHR data generation: diffusion models for enhanced privacy and clinical utility in multimorbidity clustering",
    first_author="Kita", year=2025, venue="Connection Science 37(1):2565163", publication_type="Journal article",
    peer_reviewed="Yes", primary_source_url="https://doi.org/10.1080/09540091.2025.2565163",
    doi_or_identifier="10.1080/09540091.2025.2565163", modality="Tabular EHR", task_scope="Generator plus benchmark",
    evaluated_systems="Customized DDPM; CTGAN; medGAN; TabVAE; beta-VAE; DPMM", architecture_family="Diffusion",
    model_name="Multimorbidity DDPM", dataset_names="Tanzania diabetic EHR; MIMIC-III; UK Biobank; NIMHANS; Kenya EHR",
    dataset_versions="MIMIC-III v1.4; cohort counts conflict between Methods and Results", data_access="Mixed",
    fidelity_evaluated="Yes", fidelity_metrics="Jensen-Shannon; Wasserstein; total variation; Pearson correlation; KS; joint distributions",
    clinical_validity_evaluated="No", utility_evaluated="Yes",
    utility_metrics="XGBoost/RF/LSTM accuracy; F1; precision; recall; DPMM silhouette; ARI; Davies-Bouldin; NMI",
    held_out_real_test="Unclear", privacy_evaluated="Yes", privacy_test_type="Membership risk; attribute risk; nearest-neighbor distance; KNN similarity",
    threat_model="Attack construction and attacker knowledge underspecified", membership_inference="Yes", attribute_inference="Yes",
    seeds_or_repeats="Five claimed datasets; training/transfer relationship unclear", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=2, evaluability_gate="Fail", mimic_iii_used="Yes",
    evidence_notes="Dataset counts and provenance conflict across sections; no code or formal DP. Five datasets are claimed but no clear site-shift or transfer protocol is reported."
)

add(
    report_id="STB-012", citation_key="fridadar2018livergan",
    title="GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification",
    first_author="Frid-Adar", year=2018, venue="Neurocomputing 321:321-331", publication_type="Journal article",
    peer_reviewed="Yes", primary_source_url="https://doi.org/10.1016/j.neucom.2018.09.013",
    doi_or_identifier="10.1016/j.neucom.2018.09.013", modality="Medical imaging", task_scope="Augmentation",
    evaluated_systems="Class-specific DCGAN; ACGAN comparison", architecture_family="GAN", model_name="Liver CT DCGAN",
    dataset_names="Private single-hospital liver CT lesion cohort", dataset_versions="182 lesions: 53 cysts, 64 metastases, 65 hemangiomas; 78 additional same-hospital cases",
    data_access="Private", fidelity_evaluated="Yes", fidelity_metrics="t-SNE; blinded two-radiologist real/fake and lesion-category study",
    clinical_validity_evaluated="Yes", clinical_validity_method="Two radiologists rated 182 real and 120 synthetic lesion images",
    utility_evaluated="Yes", utility_metrics="Patient-level three-fold CNN accuracy; sensitivity; specificity",
    held_out_real_test="Yes", seeds_or_repeats="Patient-level three-fold validation; 78 additional same-hospital cases", uncertainty_reported="Yes",
    fidelity_maturity=3, utility_maturity=3, privacy_maturity=0, evaluability_gate="Conditional",
    evidence_notes="Reports accuracy/sensitivity/specificity, not AUROC. De-identification and IRB status are not privacy tests; no external institution or code release."
)

add(
    report_id="STB-013", citation_key="costa2018retinal",
    title="End-to-End Adversarial Retinal Image Synthesis",
    first_author="Costa", year=2018, venue="IEEE Transactions on Medical Imaging 37(3):781-791",
    publication_type="Journal article", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1109/TMI.2017.2759102",
    doi_or_identifier="10.1109/TMI.2017.2759102", modality="Medical imaging", task_scope="Generator plus benchmark",
    evaluated_systems="Adversarial autoencoder vessel generator; conditional GAN/U-Net renderer", architecture_family="GAN",
    model_name="End-to-end retinal synthesis", dataset_names="Messidor-1; DRIVE",
    dataset_versions="Messidor grades 0-2: 946 pairs, split 614/155/177; DRIVE 20 train/20 test", data_access="Public",
    code_available="Yes", code_url="https://github.com/costapt/adversarial_retinal_synthesis",
    fidelity_evaluated="Yes", fidelity_metrics="Anatomical inspection; mutual-information nearest-training check; Image Structure Clustering quality; latent interpolation",
    clinical_validity_evaluated="Yes", clinical_validity_method="Anatomical vessel-structure assessment and vessel segmentation behavior",
    utility_evaluated="Yes", utility_metrics="Vessel-segmentation TSTR on DRIVE; AUROC over 11 independent models",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Nearest-training mutual-information memorization proxy",
    threat_model="No explicit membership/re-identification attacker; proxy checks closeness to training vessels",
    robustness_evaluated="Yes", external_validation="Yes", seeds_or_repeats="11 independent segmentation models; Messidor-to-DRIVE evaluation",
    uncertainty_reported="Yes", fidelity_maturity=3, utility_maturity=4, privacy_maturity=1, robustness_maturity=3,
    evaluability_gate="Pass",
    evidence_notes="Uses Messidor-1 and DRIVE, not STARE/CHASEDB1. No visual Turing test, vessel F1, or optic-disc accuracy. Memorization proxy is not formal privacy."
)

add(
    report_id="STB-014", citation_key="armanious2020medgan",
    title="MedGAN: Medical image translation using GANs",
    first_author="Armanious", year=2020, venue="Computerized Medical Imaging and Graphics 79:101684",
    publication_type="Journal article", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1016/j.compmedimag.2019.101684",
    doi_or_identifier="10.1016/j.compmedimag.2019.101684", modality="Medical imaging", task_scope="Translation/reconstruction",
    evaluated_systems="CasNet conditional GAN across three translation tasks", architecture_family="GAN", model_name="MedGAN",
    dataset_names="Private brain PET-to-CT; MR motion-correction; PET denoising cohorts",
    dataset_versions="PET-to-CT 46 patients; MR 11 volunteers; PET denoising 33 patients", data_access="Private",
    fidelity_evaluated="Yes", fidelity_metrics="SSIM; PSNR; MSE; VIF; UQI; LPIPS; radiologist realism study",
    clinical_validity_evaluated="Yes", clinical_validity_method="Five radiologists rated 60 triads per task",
    utility_evaluated="No", utility_metrics="NR; no downstream diagnostic or clinical-outcome task", held_out_real_test="No",
    fidelity_maturity=3, utility_maturity=0, privacy_maturity=0, evaluability_gate="Conditional",
    evidence_notes="No FID, privacy test, or downstream clinical-utility experiment. Reader study is fidelity/clinical plausibility, not diagnostic utility."
)

add(
    report_id="STB-015", citation_key="shin2020gandalf",
    title="GANDALF: Generative Adversarial Networks with Discriminator-Adaptive Loss Fine-Tuning for Alzheimer's Disease Diagnosis from MRI",
    first_author="Shin", year=2020, venue="MICCAI 2020; LNCS 12262:688-697", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://doi.org/10.1007/978-3-030-59713-9_66",
    doi_or_identifier="10.1007/978-3-030-59713-9_66", modality="Medical imaging", task_scope="Translation/reconstruction",
    evaluated_systems="Conditional MRI-to-PET GAN with Alzheimer classifier", architecture_family="GAN", model_name="GANDALF",
    dataset_names="ADNI", dataset_versions="1,033 participants; 1,525 MRI/AV45/FDG triplets; split 722/104/207",
    data_access="Credentialed", fidelity_evaluated="No", fidelity_metrics="NR; no quantitative image-fidelity or reader assessment",
    utility_evaluated="Yes", utility_metrics="Alzheimer classification accuracy; F2; precision; recall", held_out_real_test="Yes",
    fidelity_maturity=0, utility_maturity=2, privacy_maturity=0, evaluability_gate="Conditional",
    evidence_notes="Peer-reviewed MICCAI paper, not merely an arXiv preprint. Reports F2 rather than F1 and no image-fidelity or privacy evaluation."
)

add(
    report_id="STB-016", citation_key="pinaya2022brainldm",
    title="Brain Imaging Generation with Latent Diffusion Models",
    first_author="Pinaya", year=2022, venue="Deep Generative Models Workshop at MICCAI 2022; LNCS 13609",
    publication_type="Workshop paper", peer_reviewed="Yes", primary_source_url="https://arxiv.org/abs/2209.07162",
    doi_or_identifier="10.1007/978-3-031-18576-2_12", modality="Medical imaging", task_scope="Generator",
    evaluated_systems="Autoencoder plus latent diffusion; GAN baselines", architecture_family="Diffusion", model_name="Brain LDM",
    dataset_names="UK Biobank early release", dataset_versions="N=31,740 T1-weighted 3D MRI", data_access="Credentialed",
    fidelity_evaluated="Yes", fidelity_metrics="Med3D FID; MS-SSIM; 4-G-R-SSIM; ventricular-volume and brain-age conditioning correlations",
    clinical_validity_evaluated="Yes", clinical_validity_method="SynthSeg ventricular-volume and brain-age conditioning validity",
    utility_evaluated="No", utility_metrics="NR; no diagnostic TSTR task", held_out_real_test="No",
    robustness_evaluated="Yes", seeds_or_repeats="Covariate extrapolation and GAN comparisons", uncertainty_reported="No",
    fidelity_maturity=3, utility_maturity=0, privacy_maturity=0, robustness_maturity=2,
    evaluability_gate="Conditional",
    evidence_notes="PSNR and AUROC were incorrectly attributed in the old table. The paper asserts privacy but performs no privacy test; 100,000 images were released, code was not verified."
)

add(
    report_id="STB-017", citation_key="carrilloperez2024rnacdm",
    title="Generation of synthetic whole-slide image tiles of tumours from RNA-sequencing data via cascaded diffusion models",
    first_author="Carrillo-Perez", year=2024, venue="Nature Biomedical Engineering 9:320-332",
    publication_type="Journal article", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1038/s41551-024-01193-8",
    doi_or_identifier="10.1038/s41551-024-01193-8", modality="Medical imaging", task_scope="Translation/reconstruction",
    evaluated_systems="RNA VAE plus cascaded diffusion models", architecture_family="Diffusion", model_name="RNA-CDM",
    dataset_names="TCGA LUAD/KIRP/CESC/COAD/GBM; GEO GSE50760/GSE226069; PBTA; MSI dataset",
    data_access="Mixed", code_available="Yes", code_url="https://rna-cdm.stanford.edu",
    fidelity_evaluated="Yes", fidelity_metrics="FID; KID; Inception Score; HoverNet cell distributions; RNA deconvolution agreement",
    clinical_validity_evaluated="Yes", clinical_validity_method="Cell-type and transcriptomic consistency across cancer types",
    utility_evaluated="Yes", utility_metrics="Synthetic SimCLR pretraining; multicancer classification; MSI classification",
    held_out_real_test="Yes", robustness_evaluated="Yes", external_validation="Yes",
    seeds_or_repeats="Fivefold CV; five cancer types; external GEO/PBTA/MSI resources", uncertainty_reported="Yes",
    fidelity_maturity=3, utility_maturity=3, privacy_maturity=0, robustness_maturity=3, evaluability_gate="Conditional",
    evidence_notes="Old arXiv:2312.01119 is unrelated. Use the Nature DOI or bioRxiv 10.1101/2023.01.13.523899; coded year updated from 2023 to 2024 online."
)

add(
    report_id="STB-018", citation_key="zanier2024tomoray",
    title="TomoRay: Generating Synthetic Computed Tomography of the Spine From Biplanar Radiographs",
    first_author="Zanier", year=2024, venue="Neurospine 21(1):68-75", publication_type="Journal article",
    peer_reviewed="Yes", primary_source_url="https://doi.org/10.14245/ns.2347158.579",
    doi_or_identifier="10.14245/ns.2347158.579", modality="Medical imaging", task_scope="Translation/reconstruction",
    evaluated_systems="TomoRay/X2CT-GAN", architecture_family="GAN", model_name="TomoRay",
    dataset_names="VerSe2020; CTSpine1K", dataset_versions="VerSe2020 209 train/55 internal test; CTSpine1K external subset 56",
    data_access="Public", fidelity_evaluated="Yes", fidelity_metrics="PSNR; 2D SSIM; cosine similarity",
    utility_evaluated="No", utility_metrics="NR; no diagnostic task, reader study, or clinical-decision evaluation", held_out_real_test="No",
    robustness_evaluated="Yes", external_validation="Yes", seeds_or_repeats="External CTSpine1K test",
    uncertainty_reported="Yes", fidelity_maturity=2, utility_maturity=0, privacy_maturity=0, robustness_maturity=3,
    evaluability_gate="Conditional",
    evidence_notes="Paired 2D-to-3D reconstruction, not unconditional cohort generation. Clear external image test; no code or privacy evaluation."
)

add(
    report_id="STB-019", citation_key="alimozzaman2025imaging",
    title="Generative AI for Synthetic Medical Imaging to Address Data Scarcity",
    first_author="Alimozzaman", year=2025, venue="World Journal of Advanced Engineering Technology and Sciences 17(1):544-558",
    publication_type="Journal article", peer_reviewed="Unclear",
    primary_source_url="https://wjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-1415.pdf",
    doi_or_identifier="10.30574/wjaets.2025.17.1.1415", modality="Medical imaging", task_scope="Generator plus benchmark",
    evaluated_systems="Conditional U-Net/PatchGAN plus latent-diffusion refinement", architecture_family="Hybrid/other",
    model_name="Hybrid cGAN-LDM", dataset_names="Paper-stated Kaggle Brain MRI; NIH chest X-ray",
    dataset_versions="Paper states 3,064 MRI and 5,856 X-ray images; attribution is internally inconsistent", data_access="Public",
    fidelity_evaluated="Yes", fidelity_metrics="FID; SSIM; visual inspection; poorly specified radiologist plausibility",
    clinical_validity_evaluated="Yes", clinical_validity_method="Claimed radiologist plausibility; reader count/protocol not reported",
    utility_evaluated="Yes", utility_metrics="MRI and chest X-ray diagnostic classification accuracy gain", held_out_real_test="Unclear",
    privacy_evaluated="Yes", privacy_test_type="Inception-v3 cosine-similarity proxy",
    threat_model="Nearest-image/patient-mapping risk asserted with a threshold; no membership construction",
    seeds_or_repeats="Experiments stated to be repeated three times", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=1, evaluability_gate="Fail", ambiguity_flag="Yes",
    evidence_notes="Credibility flag: modality/dataset descriptions conflict, reader protocol is missing, code/data are absent. HIPAA/GDPR alignment is not a privacy test."
)

add(
    report_id="STB-020", citation_key="yoon2019timegan",
    title="Time-series Generative Adversarial Networks",
    first_author="Yoon", year=2019, venue="NeurIPS 2019", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html",
    doi_or_identifier="NeurIPS 2019 c9efe5f26cd17ba6216bbe2a7d26d490", modality="Clinical time series", task_scope="Generator",
    evaluated_systems="TimeGAN and sequence baselines", architecture_family="GAN", model_name="TimeGAN",
    dataset_names="Synthetic Sines; Google stocks; UCI Appliances Energy; private lung-cancer pathways",
    dataset_versions="Stocks 2004-2019; clinical event/timing cohort private", data_access="Mixed", code_available="Yes",
    code_url="https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/5c5a19b673a59a46720d3310172560d500ed721b/alg/timegan/",
    fidelity_evaluated="Yes", fidelity_metrics="t-SNE; PCA; discriminative score; predictive next-step score",
    utility_evaluated="Yes", utility_metrics="Generic next-step predictive TSTR MAE", held_out_real_test="Yes",
    seeds_or_repeats="Multiple domains and ablations", uncertainty_reported="Yes",
    fidelity_maturity=3, utility_maturity=1, privacy_maturity=0, evaluability_gate="Conditional",
    evidence_notes="Critical correction: clinical experiment uses a private lung-cancer pathway dataset, not MIMIC-III. No privacy, equity, or external clinical-site evaluation."
)

add(
    report_id="STB-021", citation_key="golany2019pgans",
    title="PGANs: Personalized Generative Adversarial Networks for ECG Synthesis to Improve Patient-Specific Deep ECG Classification",
    first_author="Golany", year=2019, venue="AAAI 2019", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://ojs.aaai.org/index.php/AAAI/article/view/3830",
    doi_or_identifier="10.1609/aaai.v33i01.3301557", modality="Clinical time series", task_scope="Augmentation",
    evaluated_systems="Personalized GAN plus LSTM classifier", architecture_family="GAN", model_name="PGANs",
    dataset_names="MIT-BIH Arrhythmia Database", dataset_versions="48 records; 109,492 annotated beats; PhysioNet host",
    data_access="Public", code_available="Yes", code_url="https://bitbucket.org/tomerGolany/ecg_dl",
    fidelity_evaluated="Yes", fidelity_metrics="Morphology-constrained P/Q/R/S/T loss; qualitative waveform inspection",
    clinical_validity_evaluated="Yes", clinical_validity_method="Patient-specific ECG morphology constraints",
    utility_evaluated="Yes", utility_metrics="Patient-specific arrhythmia ROC-AUC for F, S, and V classes",
    held_out_real_test="Yes", seeds_or_repeats="Patient-disjoint AAMI split; leave-one-patient-out evaluation",
    uncertainty_reported="No", fidelity_maturity=1, utility_maturity=3, privacy_maturity=0, evaluability_gate="Pass",
    evidence_notes="MIT-BIH and PhysioNet are one dataset and its host, not two datasets. Principal utility metric is ROC-AUC, not accuracy/F1."
)

add(
    report_id="STB-022", citation_key="biswal2021eva",
    title="EVA: Generating Longitudinal Electronic Health Records Using Conditional Variational Autoencoders",
    first_author="Biswal", year=2021, venue="MLHC 2021; PMLR 149:260-282", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://proceedings.mlr.press/v149/biswal21a.html",
    doi_or_identifier="PMLR 149:260-282", modality="Clinical time series", task_scope="Generator",
    evaluated_systems="Conditional Bayesian VAE", architecture_family="VAE", model_name="EVA",
    dataset_names="Sutter Health Palo Alto Medical Foundation EHR", dataset_versions="258,555 patients; 13.9 million visits",
    data_access="Private", fidelity_evaluated="Yes", fidelity_metrics="Unigram/bigram correlation; log likelihood; Jaccard diversity; unique-token ratio; clinician realism",
    clinical_validity_evaluated="Yes", clinical_validity_method="Clinician realism scores for longitudinal visit sequences",
    utility_evaluated="Yes", utility_metrics="Next-visit top-k recall; real-plus-synthetic augmentation; conditional heart-failure AUC",
    held_out_real_test="Yes", privacy_evaluated="Yes", privacy_test_type="Presence-disclosure record-matching attack",
    threat_model="Attacker knows selected patient records and tests training presence; prior membership probability 0.8",
    membership_inference="Yes", seeds_or_repeats="Held-out split and repeated prediction experiments", uncertainty_reported="No",
    fidelity_maturity=3, utility_maturity=3, privacy_maturity=2, evaluability_gate="Conditional",
    evidence_notes="Authors call the presence-disclosure assessment formal, but it is an empirical membership test, not formal differential privacy."
)

add(
    report_id="STB-023", citation_key="ashrafi2023ppgan",
    title="Protect and Extend -- Using GANs for Synthetic Data Generation of Time-Series Medical Records",
    first_author="Ashrafi", year=2023, venue="QoMEX 2023", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://arxiv.org/abs/2402.14042",
    doi_or_identifier="10.1109/QoMEX58391.2023.10178496", modality="Clinical time series", task_scope="Generator plus benchmark",
    evaluated_systems="simpleGAN; medGAN; DoppelGANger; DPGAN; PPGAN", architecture_family="GAN", model_name="PPGAN",
    dataset_names="PflegeTab", dataset_versions="81 users; 936 days; 193.72 hours; 33,377 completed tasks",
    data_access="Private", fidelity_evaluated="Yes", fidelity_metrics="Autocorrelation; sequence-length distributions; means/SD heatmaps; LSTM-based comparison",
    utility_evaluated="Yes", utility_metrics="Real-plus-synthetic and synthetic-only F1; RMSE", held_out_real_test="Unclear",
    privacy_evaluated="Yes", privacy_test_type="TensorFlow Privacy membership-inference attacks",
    threat_model="Logistic-regression and threshold membership attacks; target effectively reused as attack/shadow target",
    membership_inference="Yes", fidelity_maturity=2, utility_maturity=2, privacy_maturity=2,
    evaluability_gate="Conditional",
    evidence_notes="Critical correction: PPGAN does not report DP-SGD, epsilon, delta, accountant, or proof. DPGAN is a comparator, so remove the 'only formally DP time-series paper' claim."
)

add(
    report_id="STB-024", citation_key="naseer2023scoehr",
    title="ScoEHR: Generating Synthetic Electronic Health Records using Continuous-time Diffusion Models",
    first_author="Naseer", year=2023, venue="MLHC 2023; PMLR 219:489-508", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://proceedings.mlr.press/v219/naseer23a.html",
    doi_or_identifier="PMLR 219:489-508", modality="Tabular EHR", task_scope="Generator",
    evaluated_systems="Autoencoder plus continuous-time diffusion; medGAN family baselines", architecture_family="Diffusion", model_name="ScoEHR",
    dataset_names="MIMIC-III; Yale New Haven emergency-department records",
    dataset_versions="MIMIC-III 46,520 records/1,071 binary features; Yale 232,592 records/625 mixed features",
    data_access="Mixed", code_available="Yes", code_url="https://github.com/aanaseer/ScoEHR",
    fidelity_evaluated="Yes", fidelity_metrics="Marginals; pairwise correlation difference; log-cluster; synthetic-ranking agreement; clinician realism",
    clinical_validity_evaluated="Yes", clinical_validity_method="Three clinicians rated 100 real and 100 synthetic ED records",
    utility_evaluated="Yes", utility_metrics="Downstream AUROC ranking agreement", held_out_real_test="Yes",
    privacy_evaluated="Yes", privacy_test_type="Cosine-similarity threshold membership inference",
    threat_model="Adversary has partial records and tests generator-training membership; known-patient and synthetic-set sizes varied",
    membership_inference="Yes", robustness_evaluated="Yes", seeds_or_repeats="Five generated sets; repeated attacks across attacker/synthetic-set sizes",
    uncertainty_reported="Yes", fidelity_maturity=3, utility_maturity=2, privacy_maturity=2, robustness_maturity=2,
    evaluability_gate="Pass", mimic_iii_used="Yes",
    evidence_notes="Reclassify from time series to structured/tabular: patient-level mixed feature vectors are generated, not longitudinal sequences. Empirical privacy only."
)

add(
    report_id="STB-025", citation_key="yoon2023ehrsafe",
    title="EHR-Safe: generating high-fidelity and privacy-preserving synthetic electronic health records",
    first_author="Yoon", year=2023, venue="npj Digital Medicine 6:141", publication_type="Journal article",
    peer_reviewed="Yes", primary_source_url="https://doi.org/10.1038/s41746-023-00888-7",
    doi_or_identifier="10.1038/s41746-023-00888-7", modality="Clinical time series", task_scope="Generator",
    evaluated_systems="Sequential encoder-decoder plus latent WGAN-GP", architecture_family="GAN", model_name="EHR-Safe",
    dataset_names="MIMIC-III; eICU", dataset_versions="MIMIC-III N=19,946/90 features; eICU N=198,707",
    data_access="Credentialed", fidelity_evaluated="Yes", fidelity_metrics="CDF; means/SD/missingness; KS; t-SNE; propensity; feature importance",
    utility_evaluated="Yes", utility_metrics="Mortality TSTR AUROC with GBDT, RF, LR, GRU", held_out_real_test="Yes",
    privacy_evaluated="Yes", privacy_test_type="Membership inference; re-identification; attribute inference",
    threat_model="Training membership, nearest-record re-identification, and inference of gender/religion/marital status",
    membership_inference="Yes", attribute_inference="Yes", reidentification_or_reconstruction="Yes",
    fairness_evaluated="Yes", protected_attributes="Gender; religion", subgroup_metrics="Supplementary subgroup/algorithmic-fairness analyses; main-text parity metric NR",
    fidelity_maturity=3, utility_maturity=3, privacy_maturity=3, equity_maturity=2,
    evaluability_gate="Conditional", mimic_iii_used="Yes",
    evidence_notes="Authors explicitly state no theoretical DP guarantee. DP-SGD is only a future option. Correct fairness from No to Yes-limited based on supplementary analyses."
)

add(
    report_id="STB-026", citation_key="tian2024timediff",
    title="Reliable generation of privacy-preserving synthetic electronic health record time series via diffusion models",
    first_author="Tian", year=2024, venue="Journal of the American Medical Informatics Association 31(11):2529-2539",
    publication_type="Journal article", peer_reviewed="Yes", primary_source_url="https://doi.org/10.1093/jamia/ocae229",
    doi_or_identifier="10.1093/jamia/ocae229", modality="Clinical time series", task_scope="Generator",
    evaluated_systems="Mixed Gaussian/multinomial DDPM; baselines", architecture_family="Diffusion", model_name="TimeDiff",
    dataset_names="MIMIC-III; MIMIC-IV; eICU; Stocks; Energy", data_access="Mixed", code_available="Yes",
    code_url="https://github.com/MuhangTian/TimeDiff",
    fidelity_evaluated="Yes", fidelity_metrics="t-SNE; UMAP; GRU discriminative score; next-step predictive MAE",
    utility_evaluated="Yes", utility_metrics="TSTR and real-plus-synthetic mortality AUROC", held_out_real_test="Yes",
    privacy_evaluated="Yes", privacy_test_type="Nearest-neighbor adversarial accuracy; membership-inference risk/F1",
    threat_model="Distance-based membership risk to generator training records", membership_inference="Yes",
    robustness_evaluated="Yes", seeds_or_repeats="Ten repeats; three EHR and two nonclinical data sources", uncertainty_reported="Yes",
    fidelity_maturity=3, utility_maturity=3, privacy_maturity=2, robustness_maturity=2,
    evaluability_gate="Conditional", mimic_iii_used="Yes",
    evidence_notes="No formal DP. Paper is internally inconsistent about database counts: three named EHR databases and five total datasets are identifiable."
)

add(
    report_id="STB-027", citation_key="mawaldi2024llama2thesis",
    title="Synthetic Data Generation Using Large Language Models: Evaluating the Utility of Synthetic Clinical Text Generated via Fine-tuning Llama-2 on MIMIC-III Data When Used as Training Data for Clinical Named Entity Recognition",
    first_author="Mawaldi", year=2024, venue="Stockholm University", publication_type="Thesis",
    peer_reviewed="No", primary_source_url="https://su.diva-portal.org/smash/get/diva2%3A1955754/FULLTEXT01.pdf",
    doi_or_identifier="DiVA diva2:1955754", modality="Clinical text", task_scope="Generator",
    evaluated_systems="Llama-2 7B generator; BERT NER", architecture_family="LLM", model_name="Fine-tuned Llama-2 7B",
    dataset_names="MIMIC-III", dataset_versions="5,000 real/synthetic HPI train examples; 1,000 real test examples",
    data_access="Credentialed", fidelity_evaluated="Yes", fidelity_metrics="ROUGE-1/2/L/Lsum; length; vocabulary; entity coverage; diversity",
    utility_evaluated="Yes", utility_metrics="Clinical NER TSTR F1", held_out_real_test="Yes",
    fidelity_maturity=2, utility_maturity=2, privacy_maturity=0, evaluability_gate="Conditional", mimic_iii_used="Yes",
    evidence_notes="Critical identity correction: the old Tang et al. JAMIA citation is not this study. The matching source is a non-peer-reviewed 2024 master's thesis. Privacy was explicitly out of scope."
)

add(
    report_id="STB-028", citation_key="xu2024clingen",
    title="Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models",
    first_author="Xu", year=2024, venue="Findings of ACL 2024", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://aclanthology.org/2024.findings-acl.916/",
    doi_or_identifier="10.18653/v1/2024.findings-acl.916", modality="Clinical text", task_scope="Generator plus benchmark",
    evaluated_systems="ClinGen with GPT-3.5-turbo-0301; PubMedBERT downstream", architecture_family="LLM", model_name="ClinGen",
    dataset_names="18 biomedical/clinical NLP datasets across 8 task families", data_access="Public", code_available="Yes",
    code_url="https://github.com/ritaranx/ClinGen",
    fidelity_evaluated="Yes", fidelity_metrics="Central Moment Discrepancy; t-SNE; Sentence-BERT similarity/diversity; entity coverage; factuality review",
    clinical_validity_evaluated="Yes", clinical_validity_method="Medical-student factuality review",
    utility_evaluated="Yes", utility_metrics="PubMedBERT downstream performance across 18 datasets", held_out_real_test="Yes",
    robustness_evaluated="Yes", seeds_or_repeats="18 datasets; 8 task families; generator/classifier ablations",
    uncertainty_reported="No", fidelity_maturity=3, utility_maturity=3, privacy_maturity=0, robustness_maturity=3,
    evaluability_gate="Conditional",
    evidence_notes="Correct arXiv is 2311.00287; old 2311.01912 is unrelated. Procedural PHI avoidance is not a quantified privacy test."
)

add(
    report_id="STB-029", citation_key="hu2025clinicalie",
    title="Facilitating Clinical Information Extraction with Synthetic Data and Ontology using Large Language Models",
    first_author="Hu", year=2025, venue="AMIA Annual Symposium Proceedings:500-505", publication_type="Conference paper",
    peer_reviewed="Yes", primary_source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12919538/",
    doi_or_identifier="PMCID PMC12919538", modality="Clinical text", task_scope="Augmentation",
    evaluated_systems="GPT-4o-mini synthetic annotation; SNOMED-CT/DBSCAN filtering; QLoRA Llama-3-8B NER",
    architecture_family="LLM", model_name="Ontology-guided synthetic NER annotation",
    dataset_names="MTSamples; UTP; MIMIC-III; i2b2", data_access="Mixed",
    fidelity_evaluated="Yes", fidelity_metrics="Self-verification; ontology alignment; DBSCAN anomaly filtering; t-SNE",
    clinical_validity_evaluated="Yes", clinical_validity_method="SNOMED-CT concept alignment and anomaly filtering",
    utility_evaluated="Yes", utility_metrics="Exact NER F1 in-domain and across UTP/MIMIC-III/i2b2", held_out_real_test="Yes",
    robustness_evaluated="Yes", external_validation="Yes", seeds_or_repeats="Cross-institution evaluation on three external clinical corpora",
    uncertainty_reported="No", fidelity_maturity=1, utility_maturity=3, privacy_maturity=0, robustness_maturity=3,
    evaluability_gate="Conditional", ambiguity_flag="Yes", mimic_iii_used="Yes",
    evidence_notes="Scope ambiguity: synthesizes annotations for real sentences rather than de novo patient notes. Old arXiv:2406.07021 is unrelated; retained in frozen pilot with explicit flag."
)

add(
    report_id="STB-030", citation_key="miranda2024spanishdp",
    title="Evaluating Privacy Risks in Synthetic Clinical Text Generation in Spanish",
    first_author="Miranda", year=2024, venue="Latinx in AI at NeurIPS 2024", publication_type="Workshop paper",
    peer_reviewed="Yes", primary_source_url="https://openreview.net/forum?id=0hxBH0dEKu",
    doi_or_identifier="OpenReview 0hxBH0dEKu", modality="Clinical text", task_scope="Generator plus benchmark",
    evaluated_systems="LoRA Mistral-7B-v0.1; Meta-Llama-3.1-8B-Instruct with/without DP-SGD",
    architecture_family="LLM", model_name="DP fine-tuned Spanish clinical LLMs",
    dataset_names="MEDDOCAN", dataset_versions="1,000 reports; 750 train/250 validation; canary in 0/50/200 documents",
    data_access="Public", fidelity_evaluated="Yes", fidelity_metrics="MAUVE; perplexity",
    utility_evaluated="Yes", utility_metrics="MAUVE/perplexity privacy-utility degradation; no downstream clinical task", held_out_real_test="No",
    privacy_evaluated="Yes", privacy_test_type="Canary exposure/extraction; claimed DP-SGD",
    threat_model="Extraction of a linked surname and HIV diagnosis under repeated canary frequencies",
    reidentification_or_reconstruction="Yes", dp_claimed="Yes", formal_dp_verified="No", dp_evidence_status="Incomplete",
    dp_mechanism="Claimed DP-SGD", epsilon_reported="Yes", epsilon_values="8; 16; infinity",
    delta_reported="No", delta_values="NR", robustness_evaluated="Yes",
    seeds_or_repeats="Two LLMs; three canary frequencies; three privacy levels", uncertainty_reported="No",
    fidelity_maturity=2, utility_maturity=1, privacy_maturity=3, robustness_maturity=2,
    evaluability_gate="Conditional",
    evidence_notes="Old arXiv:2411.04081 is unrelated. Numeric epsilon is reported, but delta, accountant, clipping/noise settings, and patient-level adjacency are missing; formal guarantee is not verified."
)


def main() -> None:
    if len(rows) != 30:
        raise SystemExit(f"Expected 30 rows, found {len(rows)}")
    ids = [str(row["report_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate report_id")
    destination = Path(__file__).resolve().parents[1] / "data" / "study_annotations.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    schema_destination = destination.parent / "annotation_schema.json"
    schema_destination.write_text(
        json.dumps(annotation_schema(), indent=2) + "\n", encoding="utf-8"
    )
    jsonl_destination = destination.parent / "study_annotations.jsonl"
    jsonl_destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows and {len(COLUMNS)} columns to {destination}")


if __name__ == "__main__":
    main()
