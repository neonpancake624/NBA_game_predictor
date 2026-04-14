Project: NBA Game Predictor

Varun Sreedhara 018014945

- Problem Statement
  The goal of this project is to build a ML app that predicts the outcome of NBA games using historical and current team performance data.

In addition to predicting which team is more likely to win, the app aims to provide insights for why a certain team is favored to win a matchup. These explanations are based on performance metrics such as shooting efficiency, rebounding advantage, assists, turnovers, point differential, and home court advantage.

- Datasets
  I am using NBA datasets from Kaggle (https://www.kaggle.com/datasets/nathanlauga/nba-games)
  - Game results
  - Team performance statistics (field goal %, rebounds, assists, turnovers, etc.)

- Structure
  - Data processing
    - Loading and cleaning
    - Missing values

  - Feature engineering
    - Compute statistical differences between home and away teams

  - EDA
    - Visualize win distributions (home vs away)
    - Analyze score differences
    - Correlations between features and outcomes

  - ML model
    - Train logistic regression
    - Evaluate using:
      - Accuracy
      - Confusion matrix

  - Model interpretability
    - Analyze feature coefficients
    - Generate explanations for why a team is favored:

  - Interactive app
    - Streamlit app:
      - Select teams
      - View predicted win probability
      - Explanations for the prediction

- Current progress
  - Completed data loading and preprocessing
  - Target variable `home_win` created
  - Feature engineering implemented
  - EDA completed
  - Logistic regression trained + evaluated
  - Feature importance and explanations implemented
