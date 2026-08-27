PAIMANA-LIKE SYNTHETIC DATASET
=================================

Purpose
-------
This dataset is SYNTHETIC and is intended for a hackathon/demo prototype.
It is NOT official PAIMANA/OCMS data and must not be presented as such.

Dataset size
------------
Projects: 5000
Monthly observations per project: 12
Rows: 60,000

Files
-----
paimana_like_synthetic_monthly_dataset.csv : complete dataset
paimana_like_train.csv                    : 3,500 projects
paimana_like_test.csv                     : 1,000 projects
paimana_like_live_demo.csv                : 500 projects

Prediction targets
------------------
final_cost_overrun_flag:
    1 if final cost overrun > 10%, else 0

final_time_overrun_flag:
    1 if final time overrun > 10%, else 0

final_major_risk_flag:
    1 if either cost or time overrun flag is 1

IMPORTANT: final_* outcome fields are target/outcome fields.
Do NOT use them as model inputs.

Recommended model inputs
------------------------
Use only snapshot fields available at the reporting month, e.g.:
- original_cost_crore
- latest_revised_cost_crore
- cumulative_expenditure_crore
- planned_duration_months
- elapsed_duration_months
- schedule_consumption_pct
- expected_physical_progress_pct
- physical_progress_pct
- financial_progress_pct
- progress_gap_pct
- financial_physical_gap_pct
- milestones_total
- milestones_expected_to_date
- milestones_delayed
- milestone_delay_rate_pct
- land_acquisition_delay
- clearance_delay
- contractor_delay
- funding_issue
- expenditure_ratio_pct
- sector/ministry/state/agency

Recommended evaluation
----------------------
Because each project has monthly observations, use project-level or time-aware
validation. Do not randomly mix observations from the same project between
training and testing.

Suggested experiment
--------------------
Model A: CUF-like features only.
Model B: CUF-like + engineered/expanded features.
Compare ROC-AUC, PR-AUC, precision, recall, F1, calibration, and early-warning lead time.

Synthetic data generation
-------------------------
The data includes latent project difficulty and sector risk to create realistic
relationships between delays, progress, expenditure and eventual overruns.
Those latent variables are NOT included in the model feature list, preventing
direct leakage.

Disclaimer
----------
For presentation, say:
"Prototype evaluated using synthetic/representative project data because the
complete historical PAIMANA/OCMS training dataset is not assumed to be publicly
available to the team."
