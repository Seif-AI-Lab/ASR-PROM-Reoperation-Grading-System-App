# ASR PROM Reoperation Risk Analysis

This repository contains analysis code for PROM-based lumbar reoperation risk prediction, dynamic ODI modeling, and risk grading using American Spine Registry data.

## Contents

* `notebooks/Step_1_Analysis_and_Grading_System.ipynb`
  Step 1 preoperative PROM analysis, final model analysis, optional death-retained sensitivity analysis, grading-system construction, and survival validation.

* `notebooks/Step_2_Analysis.ipynb`
  Step 2 dynamic ODI analysis, final model analysis, sensitivity analyses, hospital analyses, and survival analysis.

* `scripts/`
  Python exports of the notebooks for code review and archival convenience.

* `docs/code_run_order.md`
  Suggested execution order and expected input files.

## Data availability

The original registry data are not included in this repository because of data-use restrictions and patient privacy requirements. The code expects the required input CSV files to be available locally or in the Google Colab `/content/` directory.

## Required input files

The primary analysis notebooks expect the following input files, depending on the analysis step:

```text
/content/PROM_ODI_Cohort.csv
/content/PROM_BackPain_Cohort.csv
/content/PROM_LegPain_Cohort.csv
/content/Step 2_ODI_Cohort.csv
```

Optional Step 1 death-retained sensitivity analyses, when run, expect the following additional input files:

```text
/content/PROM_ODI_Cohort_death_retained.csv
/content/PROM_BackPain_Cohort_death_retained.csv
/content/PROM_LegPain_Cohort_death_retained.csv
```

These optional death-retained files are not required for the primary Step 1 analysis, ASR-ODI grading-system analysis, or Step 2 dynamic ODI analysis.

## Running the code

The notebooks were prepared for execution in Google Colab or a compatible Python environment. Install the required packages with:

```bash
pip install -r requirements.txt
```

Then run the notebooks in the order listed in `docs/code_run_order.md`.

## Reproducibility note

Random seeds, train/calibration/test split rules, cross-validation settings, calibration procedures, and model-evaluation settings are specified inside the notebooks.

## Code archive

The analysis code is available on GitHub and has been archived on Zenodo.

```text
GitHub: https://github.com/Seif-AI-Lab/ASR-PROM-Reoperation-Grading-System-App
Zenodo: https://doi.org/10.5281/zenodo.21091849
```
