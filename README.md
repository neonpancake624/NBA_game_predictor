NBA Game Predictor

Varun Sreedhara 018014945

- Problem Statement
  The goal of this project is to build a ML app that predicts the outcome of NBA games using historical and current team performance data.

In addition to predicting which team is more likely to win, the app aims to provide insights for why a certain team is favored to win a matchup. These explanations are based on performance metrics such as shooting efficiency, rebounding advantage, assists, turnovers, point differential, and home court advantage.

- Dataset
  Dataset: https://www.kaggle.com/datasets/nathanlauga/nba-games

  Datasets:
  - `games.csv`
  - `teams.csv`

  Data includes:
  - Game results
  - Team performance statistics
    - Field goal %
    - Free throw %
    - Rebounds
    - Assists
    - Turnovers
    - Points scored
  - Home and away team information
  - Game dates and outcomes

- Structure
  - Load datasets using Pandas
  - Clean missing or invalid values
  - Convert date columns into datetime format
  - Create target variable home_win

  - Feature engineering
    - Compute statistical differences between home and away teams
      - FG % diff
      - FT % diff
      - Rebound diff
      - Assist diff
      - Turnover diff

  - EDA
    - Visuals
      - Home vs away win distributions
      - Score difference distributions
      - Correlations between statistical features and game outcomes

  - ML model
    - Train logistic regression model
    - Evaluation
      - Accuracy and confusion matrix
    - Implemented improved model using team average statistics to reduce data leakage and obtain realistic predictions

  - Model interpretability
    - Analyze feature coefficients
    - Generate explanations for why a team is favored:

  - Interactive streamlit app
    - User can:
      - Select teams
      - View predicted win probability
      - Explanations for the prediction

- Current progress
  - Data loading and preprocessing
  - Target variable `home_win`
  - Feature engineering
  - EDA
  - ML model training and evaluation
  - Improved model using team average statistics
  - Feature importance analysis
  - Prediction explanation system
  - Interactive team matchup prediction logic
  - Streamlit app

- Technologies used
  - Python
  - Pandas
  - Matplotlib
  - Scikit-learn
  - Jupyter Notebook
  - Streamlit
