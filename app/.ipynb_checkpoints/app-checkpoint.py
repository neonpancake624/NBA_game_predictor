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

def explain_matchup(matchup_row, home_team_name, away_team_name, predicted_winner):
    winner_reasons = []
    opponent_reasons = []

    # Shooting efficiency
    if matchup_row["fg_pct_diff"] > 0:
        better_team = home_team_name
        reason = "better shooting efficiency"
    else:
        better_team = away_team_name
        reason = "better shooting efficiency"

    if better_team == predicted_winner:
        winner_reasons.append(f"{better_team} have {reason}.")
    else:
        opponent_reasons.append(f"{better_team} have {reason}.")

    # Free throws
    if matchup_row["ft_pct_diff"] > 0:
        better_team = home_team_name
        reason = "better free-throw shooting"
    else:
        better_team = away_team_name
        reason = "better free-throw shooting"

    if better_team == predicted_winner:
        winner_reasons.append(f"{better_team} have {reason}.")
    else:
        opponent_reasons.append(f"{better_team} have {reason}.")

    # Rebounding
    if matchup_row["reb_diff"] > 0:
        better_team = home_team_name
        reason = "a rebounding advantage"
    else:
        better_team = away_team_name
        reason = "a rebounding advantage"

    if better_team == predicted_winner:
        winner_reasons.append(f"{better_team} have {reason}.")
    else:
        opponent_reasons.append(f"{better_team} have {reason}.")

    # Assists
    if matchup_row["ast_diff"] > 0:
        better_team = home_team_name
        reason = "better ball movement based on assists"
    else:
        better_team = away_team_name
        reason = "better ball movement based on assists"

    if better_team == predicted_winner:
        winner_reasons.append(f"{better_team} have {reason}.")
    else:
        opponent_reasons.append(f"{better_team} have {reason}.")

    # Home-court advantage
    if predicted_winner == home_team_name:
        winner_reasons.append(f"{home_team_name} may benefit from home-court advantage.")

    return winner_reasons, opponent_reasons

team_logos = {
    "Hawks": "https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg",
    "Celtics": "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg",
    "Nets": "https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg",
    "Hornets": "https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg",
    "Bulls": "https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg",
    "Cavaliers": "https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg",
    "Mavericks": "https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg",
    "Nuggets": "https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg",
    "Pistons": "https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg",
    "Warriors": "https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg",
    "Rockets": "https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg",
    "Pacers": "https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg",
    "Clippers": "https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg",
    "Lakers": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg",
    "Grizzlies": "https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg",
    "Heat": "https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg",
    "Bucks": "https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg",
    "Timberwolves": "https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg",
    "Pelicans": "https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg",
    "Knicks": "https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg",
    "Thunder": "https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg",
    "Magic": "https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg",
    "76ers": "https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg",
    "Suns": "https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg",
    "Trail Blazers": "https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg",
    "Kings": "https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg",
    "Spurs": "https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg",
    "Raptors": "https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg",
    "Jazz": "https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg",
    "Wizards": "https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg"
}

team_list = sorted(team_stats["team_name"].unique())

home_team = st.selectbox("Select Home Team", team_list)
away_team = st.selectbox("Select Away Team", team_list)

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Winner"):
        with st.spinner("Analyzing matchup..."):
            winner, prob, matchup_row = predict_matchup(home_team, away_team)

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 0.4, 1])

        with col1:
            st.image(team_logos.get(home_team), width=140)
            st.markdown(f"### {home_team}")

        with col2:
            st.markdown("## vs")

        with col3:
            st.image(team_logos.get(away_team), width=140)
            st.markdown(f"### {away_team}")

        st.markdown("---")

        st.subheader("Prediction")
        st.success(f"Predicted Winner: {winner}")
        st.write(f"**Win Probability:** {prob:.2%}")

        winner_reasons, opponent_reasons = explain_matchup(
            matchup_row,
            home_team,
            away_team,
            winner
        )

        st.subheader(f"Why {winner} is favored")
        for reason in winner_reasons:
            st.write(f"- {reason}")

        if opponent_reasons:
            st.subheader("Strengths for the opposing team")
            for reason in opponent_reasons:
                st.write(f"- {reason}")

        st.subheader("Stat Difference: Home Team - Away Team")
        st.dataframe(matchup_row.to_frame("Difference"))


        