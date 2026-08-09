# Smartphone Usage Patterns

An exploratory data analysis project on smartphone usage, addiction, stress, and academic impact using a 7,500-row dataset.

## Overview

This repository contains a Python-based analysis of smartphone usage behavior. The project:

- cleans the raw dataset
- explores demographic distributions
- studies covariance and correlation
- detects outliers using IQR and Z-score methods
- compares usage patterns across addiction levels
- contrasts weekday and weekend screen time
- visualizes screen-time breakdown by activity

## Dataset

The analysis is based on **Smartphone Usage and Addiction Analysis (7,500 Records)**.

The dataset includes fields such as:

- age
- gender
- daily screen time
- social media hours
- gaming hours
- work/study hours
- sleep hours
- notifications per day
- app opens per day
- weekend screen time
- addiction level
- stress level
- academic/work impact

## What the script does

The Python script:

1. loads the CSV dataset
2. drops unused ID columns
3. fills missing addiction labels with `Unknown`
4. creates helper scores for analysis
5. prints summary statistics
6. generates multiple charts and saves them as PNG files
7. prints final conclusions from the analysis

## Key findings highlighted by the analysis

- higher screen time is strongly associated with stronger addiction patterns
- social media contributes heavily to non-work screen time
- sleep tends to move inversely with usage intensity
- weekend screen time is higher than weekday screen time
- outliers in notifications and app opens represent heavy usage behavior

## Files

- `1025250224_smartphone_analysis.py` — main analysis script
- `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` — dataset
- `README.md` — project documentation

## How to run

Install the dependencies:

```bash
pip install pandas numpy matplotlib
```

Run the script from the repo root:

```bash
python 1025250224_smartphone_analysis.py
```

## Output

The script saves figures in the project directory, including:

- demographic overview
- stress and academic impact charts
- outlier box plots
- addiction-level comparison charts
- weekend vs weekday screen time chart
- screen-time breakdown chart

## Notes

The script has been adjusted so it can load the CSV from the repository instead of a local Downloads path.

## Author

Lakshay Mohata
