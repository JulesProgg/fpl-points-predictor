
———————————————————————————————————————————————————TEMPLATE DOWN THERE

Advanced Programming 2025 - Final Project Report
# Abstract Provide a concise summary (150-200 words) of your project including: - The problem you're solving - Your approach/methodology - Key results/findings - Main contributions **Keywords:** data science, Python, machine learning, [add your keywords] \newpage # Table of Contents 1. [Introduction](#introduction) 2. [Literature Review](#literature-review) 3. [Methodology](#methodology) 4. [Results](#results) 5. [Discussion](#discussion) 6. [Conclusion](#conclusion) 7. [References](#references) 8. [Appendices](#appendices) \newpage 

# 1. Introduction Introduce your project and its context. This section should include: - **Background and motivation**: Why is this problem important? - **Problem statement**: What specific problem are you solving? - **Objectives and goals**: What do you aim to achieve? - **Report organization**: Brief overview of the report structure 

# 2. Literature Review Discuss relevant prior work, existing solutions, or theoretical background: - Previous approaches to similar problems - Relevant algorithms or methodologies - Datasets used in related studies - Gap in existing work that your project addresses 

# 3. Methodology 
## 3.1 Data Description Describe your dataset(s): - **Source**: Where did the data come from? - **Size**: Number of samples, features - **Characteristics**: Type of data, distribution - **Features**: Description of important variables - **Data quality**: Missing values, outliers, etc. 
## 3.2 Approach Detail your technical approach: - **Algorithms**: Which methods did you use and why? - **Preprocessing**: Data cleaning and transformation steps - **Model architecture**: If using ML/DL, describe the model - **Evaluation metrics**: How do you measure success? 
## 3.3 Implementation Discuss the implementation details: - **Languages and libraries**: Python packages used - **System architecture**: How components fit together - **Key code components**: Important functions/classes Example code snippet: ```python def preprocess_data(df): """ Preprocess the input dataframe. Args: df: Input pandas DataFrame Returns: Preprocessed DataFrame """ # Remove missing values df = df.dropna() # Normalize numerical features scaler = StandardScaler() df[numerical_cols] = scaler.fit_transform(df[numerical_cols]) return df ``` 

# 4. Results 
## 4.1 Experimental Setup Describe your experimental environment: - **Hardware**: CPU/GPU specifications - **Software**: Python version, key library versions - **Hyperparameters**: Learning rate, batch size, etc. - **Training details**: Number of epochs, cross-validation 
## 4.2 Performance Evaluation Present your results with tables and figures. | Model | Accuracy | Precision | Recall | F1-Score | |-------|----------|-----------|--------|----------| | Baseline | 0.75 | 0.72 | 0.78 | 0.75 | | Your Model | 0.85 | 0.83 | 0.87 | 0.85 | *Table 1: Model performance comparison* ## 4.3 Visualizations Include relevant plots and figures: - Learning curves - Confusion matrices - Feature importance plots - Results visualizations ![Example Results](path/to/figure.png) *Figure 1: Description of your results* 

# 5. Discussion Analyze and interpret your results: - **What worked well?** Successful aspects of your approach - **Challenges encountered**: Problems faced and how you solved them - **Comparison with expectations**: How do results compare to hypotheses? - **Limitations**: What are the constraints of your approach? - **Surprising findings**: Unexpected discoveries 

# 6. Conclusion 
## 6.1 Summary Summarize your key findings and contributions: - Main achievements - Project objectives met - Impact of your work 
## 6.2 Future Work Suggest potential improvements or extensions: - Methodological improvements - Additional experiments to try - Real-world applications - Scalability considerations 

# References 1. Author, A. (2024). *Title of Article*. Journal Name, 10(2), 123-145. 2. Smith, B. & Jones, C. (2023). *Book Title*. Publisher. 3. Dataset Source. (2024). Dataset Name. Available at: https://example.com 4. Library Documentation. (2024). *Library Name Documentation*. https://docs.example.com 

# Appendices 
## Appendix A: Additional Results Include supplementary figures or tables that support but aren't essential to the main narrative. 
## Appendix B: Code Repository **GitHub Repository:** https://github.com/yourusername/project-repo ### Repository Structure ``` project-repo/ ├── README.md ├── requirements.txt ├── data/ │ ├── raw/ │ └── processed/ ├── src/ │ ├── preprocessing.py │ ├── models.py │ └── evaluation.py ├── notebooks/ │ └── exploration.ipynb └── results/ └── figures/ ``` 

### Installation Instructions ```bash git clone https://github.com/yourusername/project-repo cd project-repo pip install -r requirements.txt ``` 

### Reproducing Results ```bash python src/main.py --config config.yaml ``` --- *Note: This report should be exactly 10 pages when rendered. Use the page count in your PDF viewer to verify.* --- 

## Conversion to PDF To convert this Markdown file to PDF, use pandoc: ```bash pandoc project_report.md -o project_report.pdf --pdf-engine=xelatex ``` Or with additional options: ```bash pandoc project_report.md \ -o project_report.pdf \ --pdf-engine=xelatex \ --highlight-style=pygments \ --toc \ --number-sections ```

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
Cite as: Scheidegger, S., & Smirnova, A. (2025). Data Science and Advanced Programming 2025. HEC Lausanne, University of Lausanne. View citation formats →

———————————————————————————————————————————————————MY PROJECT DOWN THERE

FPL-POINTS-PREDICTOR
Predicting Real-World Player Performance

# Abstract

Fantasy Premier League (FPL) is a widely played online game in which participants select real Premier League players and score points based on the actual on-field performances of the players they selected. Predicting these points is a challenging task due to the inherent variability of football outcomes and the influence of multiple contextual factors. This project aims to develop a data-driven framework to predict player-level FPL points for upcoming gameweeks using historical performance data.

The approach relies on multi-season FPL datasets structured at the player–gameweek level. Several predictive models are implemented and compared, ranging from simple baseline methods to supervised learning techniques, including linear regression and gradient boosting models. Feature engineering focuses on recent player form through lagged performance variables, while model evaluation is conducted using out-of-sample testing. Performance is assessed with metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), coefficient of determination (R²), and rank-based correlation measures.

Results show that machine learning models, particularly gradient boosting, outperform baseline approaches in terms of predictive accuracy and ranking consistency. The project delivers reproducible prediction pipelines, interpretable evaluation outputs, and automated reporting tools. Overall, this work demonstrates how historical FPL data can be effectively leveraged to support predictive decision-making in a sports analytics context while still remaining quite far from perfection, which seems difficult to achieve given the large variance over a gameweek. 

**Keywords:** sports analytics, athlete performance, Fantasy Premier League, machine learning, Python, predictive modeling

———————————————————————————————————————————————————

# 1. Introduction

Fantasy Premier League (FPL) is a popular online fantasy football game in which millions of participants select squads composed of real English Premier League players. Points are awarded to each player based on their actual match performances, including goals, assists, clean sheets, and other in-game events.  As a result, FPL represents a complex decision-making environment under uncertainty, where participants must continuously evaluate player performance, form, and risk when selecting their own 15 players squad. I personnaly am enjoying this game for several seasons now and to make the game even more exciting, for the last two seasons, I also created a mini-league in which each member (mainly friends) must commit to a certain amount, which will then be redistributed to the best players in the mini-league.


Predicting FPL points is challenging due to the stochastic nature of football outcomes and the multitude of factors influencing individual performances, such as player form, team strength, opposition quality, injuries, and tactical changes. Despite this uncertainty, large amounts of historical performance data are publicly available, making FPL an attractive setting for applying data-driven and machine learning techniques. Accurate predictions of player-level points can provide valuable insights for decision support in team selection and transfer strategies. It may be interesting to know that at the beginning of each season, a player's price is set. The people who set the prices most likely use the same method (i.e. prediction models) to determine them. The more promising a player is in terms of points scored, the more expensive he will be. 


This project addresses the question of whether it is possible to develop a predictive framework capable of estimating Fantasy Premier League points at the player–gameweek level using historical data. Several modelling approaches are implemented and compared, ranging from simple baseline methods to supervised machine learning models. The project focuses on evaluating predictive accuracy, consistency and interpretability, while ensuring reproducibility through a modular and well-structured Python codebase. An important and interessant aspect of the project is to compare whether these predictions can compete with the odds offered by bookmakers.


The remainder of this report is organised as follows. (Section 1 is the introduction) Section 2 reviews related work and existing approaches in sports performance prediction. Section 3 presents the data, modelling methodology, and evaluation strategy. Section 4 reports and analyses the empirical results. Section 5 discusses key findings and limitations, and Section 6 concludes with directions for future work. 

———————————————————————————————————————————————————

# 2. Literature Review

The prediction of player performance in professional sports has been an active area of research within sports analytics, driven by the increasing availability of detailed historical data. In football, predictive modelling has been applied to a wide range of tasks, including match outcome forecasting, player valuation, and performance evaluation. These approaches typically rely on historical match statistics, contextual variables, and increasingly, machine learning techniques. In my opinion, it is important to point out that the ‘eye test’ (qualitative assessment) should not be overlooked, i.e. a prediction based solely on statistics cannot be complete. 

Several studies and applied projects have focused on predicting football-related outcomes using regression-based models and tree-based algorithms. Prior studies typically rely on publicly available match-level or player-level historical datasets, including practitioner-curated Fantasy Premier League records and large-scale football match databases. As football is the most popular sport, it is now very easy to find statistics for almost all professional matches. The English Premier League, due to its reputation, is perhaps the one where the most statistics are publicly available.

Simple statistical models, such as moving averages or linear regression, are often used as baselines due to their interpretability and robustness. More advanced machine learning methods, including ensemble techniques such as gradient boosting, have been shown to capture non-linear relationships between player characteristics and future performance, leading to improved predictive accuracy in many settings.
Hubáček, Šourek and Železný (2019) demonstrate that feature-based gradient boosted tree models achieve superior predictive performance in football outcome prediction by modeling non-linear dependencies and higher-order interactions between team strength, recent form, and contextual variables. This work may be a little limited in terms of the features used. 

Within the context of Fantasy Premier League, prior work has largely emerged from practitioner communities, such as data science blogs and Kaggle competitions. These contributions commonly exploit player-level historical features, including minutes played, recent points, and team-level indicators, to forecast future FPL scores.
The literature discussed above highlights the effectiveness of gradient boosting models when combined with rich, carefully engineered feature sets in football prediction tasks. However, many existing approaches focus either on ad-hoc heuristics or on single-model implementations, with limited emphasis on systematic evaluation and reproducibility. Berrar et al. (2019) emphasize that many machine learning applications in sports analytics lack standardized evaluation protocols, often focusing on single-model implementations without sufficient reproducibility guarantees. This project places particular emphasis on reproducibility and systematic evaluation.

This project builds on existing sports analytics methodologies by implementing and comparing multiple predictive approaches within a unified and reproducible framework. By evaluating baseline methods alongside supervised machine learning models using consistent metrics, the project aims to assess the extent to which historical FPL data can provide reliable player-level point predictions and how these predictions compare to alternative benchmarks, such as bookmaker-based expectations. 

———————————————————————————————————————————————————

# 3. Methodology

This section describes the data sources, modelling approach, and evaluation strategy used to predict Fantasy Premier League points at the player–gameweek level. The methodology is designed to ensure comparability between models, robustness of results, and full reproducibility.

## 3.1 Data Description


Data source and access
Two publicly available datasets hosted on Kaggle are used in this project: one providing historical Fantasy Premier League player data, and one providing football match data with bookmaker odds.
Player-level Fantasy Premier League data are obtained from the Kaggle dataset Fantasy Premier League Player Data (2016–2024) by Reeve Barreto
(https://www.kaggle.com/datasets/reevebarreto/fantasy-premier-league-player-data-2016-2024).
This dataset consists of a single CSV file containing player–gameweek–level observations. In this project, only seasons from 2016/17 to 2022/23 are retained; the 2023/24 season is excluded due to incomplete observations. The raw file is used to construct cleaned and processed player–gameweek datasets, which form the basis for all predictive models.
Bookmaker data are obtained from the Kaggle dataset Club Football Match Data (2000–2025) by Adam Gbor
(https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025).
From this dataset, only the file matches.csv is used. The raw bookmaker odds are extracted and subsequently transformed into normalized implied probabilities (home win, draw, away win), which are then used as an external benchmark for model comparison.

It covers multiple Premier League seasons and is structured at the player–gameweek level, where each observation corresponds to the performance of a single player in a given gameweek.
The final dataset contains approximately several hundred thousand player–gameweek observations, with each observation described by dozens of features. The data consist primarily of numerical variables (e.g. minutes played, points scored, lagged statistics) and a small number of categorical identifiers (e.g. player ID, team, playing position), which are encoded consistently across seasons.

Key variables include player identifiers, team information, playing position, minutes played, and Fantasy Premier League points scored. To capture short-term performance dynamics and player form, lagged features are constructed based on historical values of points and playing time from previous gameweeks. These lag variables act as proxies for recent form and availability, which are widely recognised as important determinants of future performance in football.

Data quality checks are applied prior to modelling. Observations with missing or inconsistent key variables (e.g. missing minutes or points) are filtered out. Remaining missing values in auxiliary features are handled conservatively to avoid introducing bias. Outliers are not explicitly removed, as extreme performances are an inherent characteristic of football data and are relevant to the Fantasy Premier League setting.
To reflect real-world forecasting conditions and avoid look-ahead bias, the dataset is split into training and testing sets using a time-based split, ensuring that models are trained only on information available prior to the prediction period.

## 3.2 Modelling Approach

Several predictive approaches are implemented and compared in order to assess the trade-offs between simplicity, interpretability, and predictive performance.
As a baseline, simple statistical models based on historical averages are used to establish a reference level of performance. These baseline methods provide an interpretable benchmark and help contextualise the performance gains achieved by more complex models.

More advanced supervised learning models are then applied, including linear regression and gradient boosting. Linear regression serves as a transparent and easily interpretable model, allowing direct assessment of the relationship between historical player features and future FPL points. Gradient boosting models are employed to capture non-linear relationships and higher-order interactions between variables, which are common in football performance data and are not well represented by linear models.
All models are trained using the same set of input features and evaluated on identical out-of-sample test sets to ensure a fair and consistent comparison. Hyperparameters for machine learning models are selected using reasonable default values rather than extensive tuning, in order to prioritise robustness and reproducibility over marginal performance gains.

Model performance is assessed using multiple complementary evaluation metrics. Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to measure prediction accuracy in absolute terms, with RMSE placing greater weight on large errors. The coefficient of determination (R²) is reported to quantify the proportion of variance explained relative to a naïve baseline. In addition, rank-based correlation metrics are used to evaluate how well models preserve the relative ordering of players within a gameweek, which is particularly relevant in the Fantasy Premier League decision-making context.

## 3.3 Evaluation Strategy

The project is implemented entirely in Python, using a modular and reproducible codebase. Core libraries include pandas and NumPy for data manipulation, scikit-learn for model implementation and evaluation, and matplotlib for visualisation and reporting.

The system architecture follows a clear separation of concerns. Data loading and preprocessing are handled in dedicated modules, ensuring that raw data are transformed consistently across experiments. Modelling components are implemented as reusable classes and functions, allowing multiple algorithms to be evaluated using the same data pipeline. Evaluation and reporting modules compute performance metrics and generate tables and figures in a standardised format.

This modular structure allows the full experimental pipeline—from raw data to final results—to be executed in a reproducible manner. All steps, including preprocessing, model training, prediction, and evaluation, can be reproduced using the same configuration and codebase, facilitating transparency and comparability across modelling approaches.

———————————————————————————————————————————————————

# 4. Results

This section presents the empirical results obtained from the evaluation of the different predictive models. All results are reported on out-of-sample test data and are based on a consistent experimental setup across models to ensure fair comparison.

## 4.1 Experimental Setup

All experiments are conducted on a CPU-based environment without GPU acceleration. The computations are performed on a standard personal computing setup, which is sufficient given the moderate size of the dataset and the choice of classical machine learning models.

The project is implemented in Python (version 3.10). The main libraries used include pandas and NumPy for data manipulation, scikit-learn for model implementation and evaluation, and matplotlib for result visualisation. All experiments are executed within the same software environment to ensure consistency across runs.

Model training follows a time-based train–test split, as described in the methodology section. No cross-validation is performed, as the primary objective is to replicate a realistic forecasting scenario where predictions are made on future gameweeks using only past information. For this reason, models are trained once per experimental configuration and evaluated on held-out future observations.

For machine learning models, hyperparameters are set to reasonable default values provided by the scikit-learn library. This choice is intentional and aims to prioritise robustness and reproducibility over aggressive hyperparameter optimisation. In particular, the gradient boosting model uses a fixed number of trees and a limited tree depth to balance model flexibility and overfitting risk.

## 4.2 Quantitative Performance Evaluation

Model performance is evaluated using multiple complementary metrics, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the coefficient of determination (R²), and Spearman rank correlation, computed on a hold-out test season (2022/23). Additional diagnostics such as prediction bias, median absolute error (MedAE), and the 90th percentile of absolute error (P90_AE) are reported to provide a more complete view of error distribution and robustness.

Table 1 reports the out-of-sample performance of four models: a seasonal gradient boosting model, a seasonal linear regression model, and two linear regression models based on rolling lag features. Models are ranked according to MAE, which serves as the primary accuracy criterion.

![Figure 1: Predictive Performance across models via metrics.](metrics/gw_metrics_detailed.png)

The gradient boosting seasonal model achieves the best overall performance across nearly all metrics, with an MAE of 1.122 and an RMSE of 2.025, indicating a substantial reduction in prediction error compared to linear baselines. It also explains a larger share of variance (R² = 0.263) and exhibits the strongest ranking ability, with a Spearman correlation of 0.693, highlighting its effectiveness in preserving player ordering within gameweeks.

Linear regression models perform consistently but remain inferior to the gradient boosting approach. The seasonal linear model yields an MAE of 1.217 and an R² of 0.220, while the lag-based linear models show slightly weaker performance, particularly for shorter lag windows. These results suggest that linear models capture part of the signal contained in historical player data but struggle to fully model the non-linear dynamics underlying Fantasy Premier League point outcomes. 

Across all models, prediction bias remains low (below 0.06 in absolute value), indicating that errors are largely symmetric and that no systematic over- or under-prediction is present. Median absolute errors are substantially lower than mean errors, reflecting the heavy-tailed nature of the point distribution, where a small number of high-variance performances drive a significant share of total error. This observation is further supported by the P90_AE values, which range between approximately 2.77 and 2.94 points.

Overall, these results confirm that machine learning models, and gradient boosting in particular, provide measurable and consistent improvements over linear and heuristic baselines, both in terms of point prediction accuracy and ranking quality. The superior performance of the gradient boosting model supports the hypothesis that non-linear interactions between historical player features play an important role in predicting future Fantasy Premier League performance.

## 4.3 Qualitative Analysis and Visualisation

To complement the numerical evaluation, several visual diagnostics are used to assess model behaviour and error structure. These include plots of predicted versus actual points, as well as residual plots.

Predicted-versus-actual scatter plots illustrate how closely model predictions align with realised Fantasy Premier League points. Models with higher predictive accuracy exhibit tighter clustering around the diagonal, indicating better calibration. Residual plots provide insight into the distribution of prediction errors and reveal patterns such as heteroskedasticity, where larger errors tend to occur for extreme point outcomes.

Across these visual diagnostics, the gradient boosting model consistently shows improved alignment with observed values and reduced dispersion relative to simpler models. Nevertheless, all models exhibit increased uncertainty for high-variance outcomes, reflecting the inherently stochastic nature of football performance. It does indeed seem very difficult to predict statistically unusual performances such as a hat trick, even the best players in the world rarely achieve such a performance.

![Figure 2: Predictions vs actual points](figures/gbm_seasonal_pred_vs_actual.png)

The predicted-versus-actual plot shows a clear positive relationship between predicted and realised Fantasy Premier League points, indicating that the model captures overall performance trends. However, predictions are compressed within a narrow range and systematically underpredict extreme outcomes, as illustrated by the regression line lying below the 45-degree reference line. This behaviour reflects a regression-to-the-mean effect and highlights that the model is more effective for ranking players than for accurately predicting exceptional performances.


![Figure 3: Residuals with gbm](figures/gbm_seasonal_residuals.png)

![Figure 4: Residuals with linear (lag 5 features)](figures/linear_anytime_lag5_residuals.png)

The highest concentration of residuals is observed in the interval [0.02; 0.47] for the best-performing model (gradient boosting), while for the weakest-performing model (linear regression with lag-5 features), the highest concentration lies in the interval [0.40; 0.88]. This shift towards a narrower and more central residual interval with gbm indicates a clear improvement in predictive accuracy across models.

It is nevertheless worth noting that these high-density regions are consistently located slightly above zero. This suggests the presence of a small positive bias, which appears to diminish as model complexity increases but is unlikely to be completely eliminated, reflecting the intrinsic uncertainty and stochasticity of football performance outcomes.


![Figure 5: 10 random player points predictions with gbm](predictions/10_random_gbm_predictions.png)

Several players receive the same very low predicted score (0.15 points). These cases most likely correspond to players with little or no recent playing time, for whom the prediction mainly reflects the probability of appearing in the match rather than expected on-field contribution.
Conversely, the player with a high realized score in this sample (Bruno Fernandes) is assigned a substantially higher predicted value, which is a positive signal indicating that the model correctly identifies high-impact player profiles.

![Figure 6: Best player of 10 random gameweeks](tables/best_player_random_gw.png)

The table shows that the model assigns predicted scores clustered between 4.3 and 5.6 points, reflecting comparable expected contributions among selected players. In many cases, realized points are of the same order of magnitude (e.g. Ødegaard: 5.37 vs 5, Trippier: 4.32 vs 6, Martínez: 5.37 vs 7), indicating reasonable calibration for typical outcomes.
Extreme performances remain difficult to predict, as illustrated by Haaland (5.64 predicted vs 23 actual), but the model still assigns him the highest predicted score, correctly identifying his exceptional scoring potential despite unavoidable match-level variance.


![Figure 7: 10 random team strength predictions](predictions/h_team_strength_random_gw.png)

The table shows that the model assigns home win probabilities broadly in line with bookmaker estimates, typically ranging between 0.30 and 0.70 depending on the relative strength of the teams. For most matches, the absolute difference between the model and Bet365 probabilities remains limited (often below 10 percentage points), indicating a good overall calibration.
Larger deviations occur mainly for matches involving top teams, where bookmakers tend to assign very high probabilities (above 0.70), while the model remains more conservative. This suggests that the model captures relative team strength effectively, while avoiding extreme probabilities driven by market sentiment.



![Figure 8: Top 10 player of a single gameweek](tables/top10_player_demo_gw.png)

For a fixed Gameweek (demo GW = GW21), the model assigns predicted scores within a narrow range (approximately 4.1 to 5.4 points), reflecting similar expected contributions across top-ranked players. In several cases, the realized outcomes are close to the predictions (e.g. Ødegaard: 5.37 vs 5, Bruno Fernandes: 4.15 vs 5), indicating reasonable short-term calibration.
At the same time, notable deviations remain (e.g. Trippier: 4.32 vs 8, Rashford: 4.18 vs 7), highlighting the inherent match-level volatility of FPL points rather than systematic model bias.


![Figure 9: Top 5 teams according to predictions](tables/top5_teams.png)

The table shows that the model clearly differentiates teams based on the aggregate strength of their top-11 players. Manchester City ranks first with a Top-11 sum of 56.34 points (average 5.12 per player), followed closely by Liverpool (55.07, 5.01) and Brighton (53.46, 4.86).
The relatively narrow spread between teams (approximately 51.9 to 56.3 points) indicates a competitive upper tier, while still reflecting meaningful differences in squad quality. These results confirm that the model produces consistent and interpretable team-level rankings driven by individual player predictions.

———————————————————————————————————————————————————

# 5. Discussion

The empirical results presented in Section 4 provide a detailed assessment of the strengths and limitations of the proposed predictive framework. Overall, the findings confirm that modelling choices have a substantial impact on predictive performance, while also highlighting intrinsic constraints related to the variability of football outcomes.

A first key result is the consistent superiority of the gradient boosting seasonal model across quantitative metrics. As shown in Table 1, this model achieves the lowest MAE (1.122), the highest explained variance (R² = 0.263), and the strongest rank correlation (Spearman = 0.693). These improvements over linear baselines indicate that non-linear interactions between historical player features contribute meaningfully to predictive accuracy. In contrast, linear models, while stable and interpretable, exhibit higher error levels and weaker ranking consistency, particularly when relying on short rolling lag windows.

The visual diagnostics further support these conclusions. Predicted-versus-actual scatter plots show a clear positive relationship between predictions and realised points, confirming that the models successfully capture general performance trends. However, the compression of predictions toward moderate values and the systematic underprediction of extreme outcomes reveal a pronounced regression-to-the-mean effect. This behaviour is especially evident for high point totals and reflects the difficulty of forecasting rare, high-impact performances using historical data alone.

Residual analyses provide additional insight into model behaviour. For the gradient boosting model, the highest density of residuals lies in the interval [0.02; 0.47], compared to [0.40; 0.88] for the weakest-performing linear model. This shift toward a narrower and more central residual distribution illustrates the reduction in typical prediction error achieved by the more flexible model. At the same time, the slight displacement of high-density regions above zero suggests the presence of a small positive bias. While this bias decreases as model complexity increases, it does not disappear entirely, underscoring the structural uncertainty inherent in football performance data.

From an applied perspective, the ranking-oriented outputs analysed in Section 4 are particularly informative. Gameweek-level predictions, top-player rankings, and team-level aggregates demonstrate that the model produces coherent and interpretable signals aligned with realistic decision-making scenarios in Fantasy Premier League. Although extreme outcomes remain difficult to predict accurately, the model consistently assigns higher expected values to high-impact players, indicating effective identification of relative player quality even in volatile settings.

Finally, the comparison with bookmaker-based probabilities suggests that the model captures team strength patterns broadly consistent with market expectations. While bookmakers benefit from additional information sources and expert input, the relatively small deviations observed for most matches indicate that a purely data-driven approach can generate competitive estimates. This result is encouraging, while also pointing to potential gains from integrating richer contextual features in future extensions.

Taken together, these results suggest that while predictive accuracy at the individual gameweek level is fundamentally constrained by randomness and rare events, carefully designed machine learning models can deliver meaningful improvements in both accuracy and ranking quality. The findings support the initial hypothesis that non-linear modelling approaches offer tangible benefits in the Fantasy Premier League context, while also clarifying the practical limits of statistical prediction in a highly stochastic environment.

———————————————————————————————————————————————————

# 6. Conclusion

## 6.1 Summary

This project examined the extent to which historical Fantasy Premier League data can be leveraged to predict player-level point outcomes at the gameweek level. The primary objective was to develop a reproducible predictive framework capable of estimating expected player performance and to evaluate the effectiveness of different modelling approaches within a realistic forecasting setting.

The results show that data-driven methods provide clear improvements over simple baseline strategies. Supervised learning models, and gradient boosting in particular, achieve lower prediction errors and stronger ranking consistency by capturing non-linear relationships in historical player performance data. Linear regression models also perform competitively while offering greater interpretability, making them useful reference models within the framework. Overall, the project successfully meets its objectives by demonstrating that historical information on player form and playing time contains meaningful predictive signals.

Beyond predictive accuracy, a key contribution of this work lies in the construction of a modular and reproducible analysis pipeline. The project integrates data collection, preprocessing, modelling, evaluation, and reporting within a consistent programming structure, enabling transparent comparison across models and facilitating further experimentation. As such, the framework provides a practical foundation for decision support in Fantasy Premier League contexts rather than attempting to deliver exact point forecasts.

## 6.2 Future Work

Several extensions could further enhance the proposed approach. From a methodological perspective, incorporating richer contextual features—such as advanced performance statistics, opponent-specific effects, or team tactical indicators—could improve predictive power. Exploring probabilistic models or sequential approaches may also allow uncertainty to be quantified more explicitly, providing users with confidence intervals rather than point estimates.

Additional experiments could investigate alternative evaluation schemes, such as rolling retraining or season-by-season backtesting, to better capture evolving player dynamics. Finally, from an applied standpoint, the framework could be extended to support real-time predictions, integration with live data sources, or deployment within interactive dashboards, raising scalability and usability considerations. These directions highlight the potential for further development while acknowledging the fundamental uncertainty inherent in football performance prediction.

———————————————————————————————————————————————————

# References

1. Fantasy Premier League. (2024). *Official Fantasy Premier League Rules and Scoring System*. Retrieved from https://fantasy.premierleague.com  

2. Kaggle. (2023). *Fantasy Premier League Dataset*. Retrieved from https://www.kaggle.com  

3. Hubáček, Šourek and Železný (2019)

4. Berrar et al. (2019)

———————————————————————————————————————————————————

# Appendices

## Appendix A: Code Repository

The complete source code for this project is available on GitHub and is structured to ensure clarity, modularity, and reproducibility.

**GitHub Repository:**  
https://github.com/liardetj/fpl-points-predictor

———————————————————————————————————————————————————


*****TO DO*****  

-> SPEAK ABOUT THE BOOKMAKERS COMPARISON (THE RESULTS OF RUNNING THE 4) OF THE MAIN.PY / MAYBE MAKE EXTRA CODE TO EXPORT A READABLE .MD FILE OF IT... (#4)

-> REVIEW THE COMMENTAIRES OF THE FIGURES (#4)

-> TOTAL RELECTURE 

-> ADJUSTMENTS

-> CONVERT TO PDF 

-> VIDEO










