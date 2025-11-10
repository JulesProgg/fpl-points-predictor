PROPOSAL.md

Fantasy Premier League Points Predictor
Predicting Real-World Player Performance

1.	Problem Statement and Motivation
Fantasy Premier League (FPL) is an online game played on mobile or desktop, where participants build the best possible team using real Premier League players. Each selected player earns points based on their actual performances — goals, assists, clean sheets, bonuses, and more.
Predicting these performances is a fascinating challenge, as it depends on a wide range of factors: player form, team momentum, opponent strength, match context, and potential injuries. FPL merges two fields I am passionate about — football and statistics. It is also an immensely popular game, with over ten million players participating every year.
The goal of this project is to develop a predictive model capable of estimating the expected number of points for each player in every Gameweek throughout a full season. This project combines data science, statistical modelling, and advanced programming within a sports analytics framework closely related to real-world financial data analysis.

2.	Planned Approach and Technologies
The project will follow a structured methodology. Data will be collected from Kaggle, where I have already identified several relevant datasets from past seasons. The data will be cleaned, normalized, and merged using Python libraries such as pandas and NumPy.
The modelling process will begin with moving averages and linear regressions, before being enhanced through gradient-based algorithms. The model will be trained on historical data from the 2021–2023 seasons and tested on the 2024 dataset (years are given as examples for now). Results will be visualized through an automatically generated dashboard (matplotlib/seaborn), accessible via a command-line interface built with Typer. All of this must be approached using methods covered in class.

3.	Expected Challenges and Proposed Solutions
Missing or noisy data will be handled using automated validation procedures and controlled imputations. Given the high variability of player performance, realistic smoothing will be applied through moving averages and aggregated evaluations by Gameweek. The project’s technical complexity may be challenging, so proper code modularization and extensive unit testing will be essential.

4.	Success Criteria
A complete and reproducible data pipeline. Predictive accuracy measured by an acceptable MAE on test data. Well-documented and PEP8-compliant code with over 70% test coverage. Clear visualizations and a concise 10-page final report.

5.	Additional Objectives
Automated team optimization via linear programming (incorporating FPL budget and constraints). A Streamlit web application to visualize forecasts. Quantile regressions to model uncertainty and confidence intervals.
An exploratory annex will compare model predictions to bookmaker-implied probabilities.
This will assess whether model-derived expectations outperform market odds, such conclusion would be very interesting...

6.	Alignment with the Course
This project aligns closely with the learning objectives of the Advanced Programming course. It integrates time series forecasting to predict FPL point trends, data pipeline development for automated multi-source ETL processes, and the design of a dashboard for the visualization of key indicators and predictions. It also involves statistical evaluation of models and the comparison of predictive strategies. Overall, it represents a comprehensive approach combining data engineering, predictive modelling, and applied analytical reasoning.

