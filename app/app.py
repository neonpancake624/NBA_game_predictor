import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title(" NBA Game Outcome Predictor")

# Load data
games = pd.read_csv("data/raw/games.csv")
teams = pd.read_csv("data/raw/teams.csv")

# Clean data
games = games.dropna(subset=["PTS_home", "PTS_away"])
games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])
games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int)

# Create team averages
home_stats = games.groupby("HOME_TEAM_ID")[[
    "FG_PCT_home", "FT_PCT_home", "REB_home", "AST_home"
]].mean()
home_stats.columns = ["FG_PCT", "FT_PCT", "REB", "AST"]

away_stats = games.groupby("VISITOR_TEAM_ID")[[
    "FG_PCT_away", "FT_PCT_away", "REB_away", "AST_away"
]].mean()
away_stats.columns = ["FG_PCT", "FT_PCT", "REB", "AST"]

team_stats = pd.concat([home_stats, away_stats]).groupby(level=0).mean()

team_names = teams.set_index("TEAM_ID")["NICKNAME"].to_dict()
team_stats["team_name"] = team_stats.index.map(team_names)
team_stats = team_stats.dropna(subset=["team_name"])

# Build improved training dataset
rows = []

for _, row in games.iterrows():
    home_id = row["HOME_TEAM_ID"]
    away_id = row["VISITOR_TEAM_ID"]

    if home_id not in team_stats.index or away_id not in team_stats.index:
        continue

    home = team_stats.loc[home_id]
    away = team_stats.loc[away_id]

    rows.append({
        "fg_pct_diff": home["FG_PCT"] - away["FG_PCT"],
        "ft_pct_diff": home["FT_PCT"] - away["FT_PCT"],
        "reb_diff": home["REB"] - away["REB"],
        "ast_diff": home["AST"] - away["AST"],
        "home_win": row["home_win"]
    })

model_df = pd.DataFrame(rows)

features = ["fg_pct_diff", "ft_pct_diff", "reb_diff", "ast_diff"]
X = model_df[features]
y = model_df["home_win"]

model = LogisticRegression()
model.fit(X, y)

def predict_matchup(home_team_name, away_team_name):
    home_team_id = team_stats[team_stats["team_name"] == home_team_name].index[0]
    away_team_id = team_stats[team_stats["team_name"] == away_team_name].index[0]

    home = team_stats.loc[home_team_id]
    away = team_stats.loc[away_team_id]

    matchup = pd.DataFrame([{
        "fg_pct_diff": home["FG_PCT"] - away["FG_PCT"],
        "ft_pct_diff": home["FT_PCT"] - away["FT_PCT"],
        "reb_diff": home["REB"] - away["REB"],
        "ast_diff": home["AST"] - away["AST"]
    }])

    home_win_prob = model.predict_proba(matchup)[0][1]

    if home_win_prob >= 0.5:
        predicted_winner = home_team_name
        winner_prob = home_win_prob
    else:
        predicted_winner = away_team_name
        winner_prob = 1 - home_win_prob

    return predicted_winner, winner_prob, matchup.iloc[0]

def explain_matchup(matchup_row, home_team_name, away_team_name):
    reasons = []

    if matchup_row["fg_pct_diff"] > 0:
        reasons.append(f"{home_team_name} has better shooting efficiency.")
    else:
        reasons.append(f"{away_team_name} has better shooting efficiency.")

    if matchup_row["ft_pct_diff"] > 0:
        reasons.append(f"{home_team_name} has better free-throw shooting.")
    else:
        reasons.append(f"{away_team_name} has better free-throw shooting.")

    if matchup_row["reb_diff"] > 0:
        reasons.append(f"{home_team_name} has a rebounding advantage.")
    else:
        reasons.append(f"{away_team_name} has a rebounding advantage.")

    if matchup_row["ast_diff"] > 0:
        reasons.append(f"{home_team_name} has better ball movement based on assists.")
    else:
        reasons.append(f"{away_team_name} has better ball movement based on assists.")

    reasons.append(f"{home_team_name} may benefit from home-court advantage.")

    return reasons

team_list = sorted(team_stats["team_name"].unique())

home_team = st.selectbox("Select Home Team", team_list)
away_team = st.selectbox("Select Away Team", team_list)

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Winner"):
        winner, prob, matchup_row = predict_matchup(home_team, away_team)

        st.subheader("Prediction")
        st.write(f"**Predicted Winner:** {winner}")
        st.write(f"**Win Probability:** {prob:.2%}")

        st.subheader("Why this team is favored")
        for reason in explain_matchup(matchup_row, home_team, away_team):
            st.write(f"- {reason}")

        st.subheader("Stat Difference: Home Team - Away Team")
        st.dataframe(matchup_row.to_frame("Difference"))