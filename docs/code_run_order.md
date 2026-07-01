# Code run order

## Step 1

Run:

```text
notebooks/Step_1_Analysis_and_Grading_System.ipynb
```

Required input files for the primary Step 1 analyses:

```text
/content/PROM_ODI_Cohort.csv
/content/PROM_BackPain_Cohort.csv
/content/PROM_LegPain_Cohort.csv
```

Optional input files for the Step 1 death-retained sensitivity analysis:

```text
/content/PROM_ODI_Cohort_death_retained.csv
/content/PROM_BackPain_Cohort_death_retained.csv
/content/PROM_LegPain_Cohort_death_retained.csv
```

The death-retained files are optional and are only needed if the death-retained Step 1 sensitivity analysis is run. They are not needed for the primary Step 1 analysis, ASR-ODI grading-system analysis, or Step 2 analysis.

This notebook contains the Step 1 preoperative PROM analyses and ASR-ODI grading-system analyses.

## Step 2

Run:

```text
notebooks/Step_2_Analysis.ipynb
```

Required input file:

```text
/content/Step 2_ODI_Cohort.csv
```

This notebook contains the Step 2 dynamic ODI analyses. Step 2 does not use death-retained input files.

## Important privacy note

Do not commit registry CSV files, generated output tables, model artifacts, or files containing patient-level data to the public repository.
