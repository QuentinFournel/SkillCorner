import streamlit as st  
import pandas as pd
from skillcorner.client import SkillcornerClient
from skillcornerviz.standard_plots import radar_plot as radar
from skillcornerviz.utils import skillcorner_physical_utils as p_utils
from skillcornerviz.standard_plots import bar_plot as bar
from skillcornerviz.standard_plots import scatter_plot as scatter
from datetime import datetime

# Dictionnaire des compétitions (stocké en dur pour éviter de le recalculer)
COMPETITION_DICTIONARY = {
    241: 'Championship (ENG) - 2021/2022', 249: 'Pro League (BEL) - 2021/2022', 
    250: 'Primeira Liga (POR) - 2021/2022', 251: 'Eredivisie (NED) - 2021/2022', 
    257: 'Superliga (DEN) - 2021/2022', 260: 'Bundesliga (AUT) - 2021/2022'
}

# Dictionnaire des métriques avec labels et unités
PHYSICAL_OPTIONS = {
    "Total dist.": ("running_distance_full_all", "Total dist.", "m"),
    "Sprint dist.": ("sprint_distance_full_all", "Sprint dist.", "m"),
    "HI dist.": ("hi_distance_full_all", "HI dist.", "m"),
    "Peak Sprint Vel. (99p)": ("psv99", "Peak Sprint Vel. (99p)", "km/h"),
    "Sprint count": ("sprint_count_full_all", "Sprint count", "count"),
    "HSR actions": ("hsr_count_full_all", "HSR actions", "count"),
    "HI efforts": ("hi_count_full_all", "HI efforts", "count"),
    "HI accels": ("highaccel_count_full_all", "HI accels", "count"),
    "HI decels": ("highdecel_count_full_all", "HI decels", "count"),
    "Dist./min": ("total_metersperminute_full_tip", "Dist./min", "m/min"),
}

# Dictionary of run types with labels and units
RUNS_OPTIONS = {
    "Cross Receiver": ("count_cross_receiver_runs_per_30_min_tip", "Cross Receiver Runs", "runs/30min"),
    "In Behind": ("count_runs_in_behind_per_30_min_tip", "In Behind Runs", "runs/30min"),
    "Ahead Of The Ball": ("count_runs_ahead_of_the_ball_per_30_min_tip", "Ahead Of The Ball Runs", "runs/30min"),
    "Overlap": ("count_overlap_runs_per_30_min_tip", "Overlap Runs", "runs/30min"),
    "Underlap": ("count_underlap_runs_per_30_min_tip", "Underlap Runs", "runs/30min"),
    "Support": ("count_support_runs_per_30_min_tip", "Support Runs", "runs/30min"),
    "Coming Short": ("count_coming_short_runs_per_30_min_tip", "Coming Short Runs", "runs/30min"),
    "Dropping Off": ("count_dropping_off_runs_per_30_min_tip", "Dropping Off Runs", "runs/30min"),
    "Pulling Half-Space": ("count_pulling_half_space_runs_per_30_min_tip", "Pulling Half-Space Runs", "runs/30min"),
    "Pulling Wide": ("count_pulling_wide_runs_per_30_min_tip", "Pulling Wide Runs", "runs/30min"),
}

def prétraitement_physical(df):
    # Rename la colonne 'position_group' en 'group'
    df.rename(columns={'position_group': 'group'}, inplace=True)
    df.rename(columns={'player_short_name': 'short_name'}, inplace=True)
    df.rename(columns={'minutes_full_all': 'minutes_played_per_match'}, inplace=True)
    df.rename(columns={'minutes_full_tip': 'adjusted_min_tip_per_match'}, inplace=True)

    df['competition_name'] = df['competition_name'].str.split('-').str[-1].str.strip()

    return df

# Mettre en cache uniquement les données, sans le client SkillCorner
@st.cache_data
def load_off_ball_runs_data():
    """Charge et met en cache les données Off-Ball Runs depuis l'API SkillCorner."""
    client = SkillcornerClient(username=st.secrets["API"]["API_USERNAME"], password=st.secrets["API"]["API_PASSWORD"])
    all_data = []
    
    for id in COMPETITION_DICTIONARY.keys():
        data = client.get_in_possession_off_ball_runs(params={
            'competition_edition': id,
            'playing_time__gte': 60,
            'count_match__gte': 8,
            'average_per': '30_min_tip',
            'group_by': 'player,team,competition,season,group',
            'run_type': 'all,run_in_behind,run_ahead_of_the_ball,'
                        'support_run,pulling_wide_run,coming_short_run,'
                        'underlap_run,overlap_run,dropping_off_run,'
                        'pulling_half_space_run,cross_receiver_run'
        })
        all_data.append(pd.DataFrame(data))
    
    return pd.concat(all_data, ignore_index=True)

@st.cache_data
def load_physical_data():
    """Charge et met en cache les données physiques depuis l'API SkillCorner."""
    client = SkillcornerClient(username=st.secrets["API"]["API_USERNAME"], password=st.secrets["API"]["API_PASSWORD"])
    all_data = []
    
    for id in COMPETITION_DICTIONARY.keys():
        data = client.get_physical(params={
            'competition_edition': id,
            'group_by': 'player,team,competition,season,group',
            'possession': 'all,tip,otip',
            'playing_time__gte': 60, 
            'count_match__gte': 8,
            'data_version': '3'
        })
        all_data.append(pd.DataFrame(data))
    
    return pd.concat(all_data, ignore_index=True)

@st.cache_data
def get_skillcorner_data(dataset_name, df_off_ball_runs, df_physical):
    """Retourne le dataset choisi en utilisant le cache pour éviter les recalculs inutiles."""
    if dataset_name == "Off-Ball Runs":
        return df_off_ball_runs
    elif dataset_name == "Physical Data":
        return df_physical
    return None

def plot_radar_chart(df, player_name, selected_metrics):
    player_id = df[df['player_name'] == player_name]['player_id'].values[0]
    player_group = df[df['player_id'] == player_id]['group'].values[0]

    # Extraction des métriques et labels à partir du dictionnaire sélectionné (RUNS_OPTIONS ou metrics_options)
    metrics_dict = {value[0]: value[1] for value in selected_metrics.values()}  # {metric_to_use: metric_label}

    # Plot du radar chart
    fig, _ = radar.plot_radar(df,
                               data_point_id='player_id',
                               label=player_id,
                               plot_title=f"Performance Profile | {player_name}",
                               metrics=list(metrics_dict.keys()),  # Liste des métriques utilisées
                               metric_labels=metrics_dict,  # Dictionnaire {metric: label}
                               percentiles_precalculated=False,
                               suffix=' Runs P30 TIP' if selected_metrics == RUNS_OPTIONS else '',  
                               positions=player_group,
                               matches=8,
                               minutes=60, 
                               competitions=df[df['player_id'] == player_id]['competition_name'].values[0], 
                               seasons=df[df['player_id'] == player_id]['season_name'].values[0], 
                               add_sample_info=True)

    return fig

def plot_bar(df, metric_to_use, metric_label, metric_unit):
    df['plot_label'] = df['short_name'] + ' | ' + df['team_name']

    # Sélectionner les 3 joueurs avec la plus haute valeur pour la métrique choisie
    top_players = df.nlargest(3, metric_to_use)['player_id'].tolist()

    # Plot du bar chart
    fig, _ = bar.plot_bar_chart(df, 
                                 metric=metric_to_use,  # Utilisation de la métrique sélectionnée
                                 label=metric_label,  # Nom dynamique pour le label
                                 unit=metric_unit,  # Unité dynamique
                                 primary_highlight_group=top_players,  # Met en avant les 3 meilleurs joueurs
                                 add_bar_values=True,
                                 data_point_id='player_id',
                                 data_point_label='plot_label')

    return fig

def plot_scatter_chart(df, teams, metrics_to_use, metrics_label, metrics_unit):
    # Plot du bar chart
    fig, _ = scatter.plot_scatter(df, 
                                x_metric=metrics_to_use[0],
                                y_metric=metrics_to_use[1], 
                                data_point_id='team_name',
                                data_point_label='short_name',
                                x_label=metrics_label[0],
                                y_label=metrics_label[1],
                                x_unit=metrics_unit[0],
                                y_unit=metrics_unit[1],
                                primary_highlight_group=teams[:1], 
                                secondary_highlight_group=teams[-1:],)

    return fig

def load_csv():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ CSV successfully loaded!")
    else:
        st.warning("📂 Please upload a CSV file.")

def load_skillcorner_data(df_off_ball_runs, df_physical):
    data_choice = st.selectbox("Choose a dataset", ["Off-Ball Runs", "Physical Data"])
    
    if data_choice == "Off-Ball Runs":
        st.session_state.df = df_off_ball_runs
        st.session_state.dataset_type = "Off-Ball Runs"
    else:
        st.session_state.df = df_physical
        st.session_state.dataset_type = "Physical Data"
    
    st.success(f"✅ {data_choice} data successfully loaded!")

def st_action(df_off_ball_runs, df_physical):
    st.title("Scouting Player with SkillCorner")

    # Initialiser session_state
    if "df" not in st.session_state:
        st.session_state.df = None

    # Choix du mode d'importation
    with st.sidebar:
        st.header("Data Import")
        importation_system = st.radio("Select data source", ["CSV", "SkillCorner API"])

        if importation_system == "CSV":
            load_csv()
        else:
            load_skillcorner_data(df_off_ball_runs, df_physical)

    # Sélection de la page (désactivée tant qu’aucune donnée n’est chargée)
    if st.session_state.df is not None:
        page = st.sidebar.radio("Navigation", ["Data Overview", "Bar Plot", "Scatter Plot", "Radar Plot", "Recommendation System"])
    else:
        page = "Data Overview"
        st.sidebar.warning("⚠️ Please load data first!")

    # Affichage des pages
    if page == "Data Overview":
        st.header("📊 Data Overview")
        if st.session_state.df is not None:
            st.write(st.session_state.df)
            st.write(f"🔍 **Rows:** {st.session_state.df.shape[0]} | **Columns:** {st.session_state.df.shape[1]}")
        else:
            st.info("No data loaded. Please import data from the sidebar.")

    elif page == "Bar Plot":
        # Vérifie si le dataframe contient bien des données
        if not st.session_state.df.empty:
            st.header("📈 Bar Plot")
        
            col1, col2 = st.columns(2)

            with col1:
                position = st.selectbox("Select a position", ["All"] + list(st.session_state.df['group'].unique()), key="position")

            with col2:
                # Convertir la colonne 'player_birthdate' en datetime
                st.session_state.df['player_birthdate'] = pd.to_datetime(st.session_state.df['player_birthdate'], errors='coerce')

                # Calculer l'âge des joueurs
                today = datetime.today()
                st.session_state.df['age'] = st.session_state.df['player_birthdate'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

                # Filtrer les joueurs en fonction de leur âge
                min_age, max_age = st.slider(
                    "Select age range",
                    min_value=int(st.session_state.df['age'].min()), 
                    max_value=int(st.session_state.df['age'].max()), 
                    value=(int(st.session_state.df['age'].min()), int(st.session_state.df['age'].max())), 
                    key="age"
                )

            championnats = st.multiselect("Select competitions", st.session_state.df['competition_name'].unique(), key="championnats")

            # Copie du dataframe de base pour éviter de modifier `st.session_state.df` directement
            filtered_df = st.session_state.df.copy()

            # 🔹 Filtrer par position
            if position != "All":
                filtered_df = filtered_df[filtered_df['group'] == position]

            # 🔹 Filtrer par âge
            filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

            # 🔹 Filtrer par compétitions
            filtered_df = filtered_df[filtered_df['competition_name'].isin(championnats)]

            # Filtrer par clubs
            clubs = st.multiselect("Select clubs", filtered_df['team_name'].unique(), key="clubs")
            if clubs:
                filtered_df = filtered_df[filtered_df['team_name'].isin(clubs)]

            st.write(filtered_df)

            # Détection du dataset actif
            if "dataset_type" in st.session_state:
                if st.session_state.dataset_type == "Off-Ball Runs":
                    available_metrics = RUNS_OPTIONS
                else:
                    available_metrics = PHYSICAL_OPTIONS
            else:
                available_metrics = PHYSICAL_OPTIONS  # Valeur par défaut au cas où

            # Sélection dynamique de la métrique disponible
            selected_metric_label = st.selectbox("Choose a metric:", list(available_metrics.keys()))
            metric_to_use, metric_label, metric_unit = available_metrics[selected_metric_label]

            if st.button("Generate Bar Chart"):
                fig = plot_bar(filtered_df, metric_to_use, metric_label, metric_unit)
                st.pyplot(fig)
        else:
            st.error("⚠️ No data available. Please upload a dataset first.")

    elif page == "Scatter Plot":
        # Check if the dataframe contains data
        if not st.session_state.df.empty:
            st.header("🔍 Scatter Plot")

            col1, col2 = st.columns(2)

            with col1:
                position = st.selectbox("Select a position", ["All"] + list(st.session_state.df['group'].unique()), key="position")

            with col2:
                # Convertir la colonne 'player_birthdate' en datetime
                st.session_state.df['player_birthdate'] = pd.to_datetime(st.session_state.df['player_birthdate'], errors='coerce')

                # Calculer l'âge des joueurs
                today = datetime.today()
                st.session_state.df['age'] = st.session_state.df['player_birthdate'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

                # Filtrer les joueurs en fonction de leur âge
                min_age, max_age = st.slider(
                    "Select age range",
                    min_value=int(st.session_state.df['age'].min()), 
                    max_value=int(st.session_state.df['age'].max()), 
                    value=(int(st.session_state.df['age'].min()), int(st.session_state.df['age'].max())), 
                    key="age"
                )

            # Competition selection
            competitions = st.multiselect("Select competitions", st.session_state.df['competition_name'].unique(), key="scatter_competition")

            # Copy filtered dataframe to avoid modifying `st.session_state.df`
            filtered_df = st.session_state.df.copy()

            # 🔹 Filtrer par position
            if position != "All":
                filtered_df = filtered_df[filtered_df['group'] == position]

            # 🔹 Filtrer par âge
            filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

            # 🔹 Filter by competition
            if competitions:
                filtered_df = filtered_df[filtered_df['competition_name'].isin(competitions)]

            # Detect active dataset (RUNS or PHYSICAL)
            if "dataset_type" in st.session_state:
                if st.session_state.dataset_type == "Off-Ball Runs":
                    available_metrics = RUNS_OPTIONS
                else:
                    available_metrics = PHYSICAL_OPTIONS
            else:
                available_metrics = PHYSICAL_OPTIONS  # Default if not set

            # Layout for selecting two teams and two metrics
            col3, col4 = st.columns(2)

            # Select two specific teams for comparison
            with col3:
                team_1 = st.selectbox("Select first team", list(filtered_df['team_name'].unique()), key="team_1")

            with col4:
                team_2 = st.selectbox("Select second team", list(filtered_df['team_name'].unique()), key="team_2")

            # Select X and Y metrics
            col5, col6 = st.columns(2)

            with col5:
                selected_metric_x = st.selectbox("Choose X-axis metric", list(available_metrics.keys()), key="scatter_x")
            with col6:
                selected_metric_y = st.selectbox("Choose Y-axis metric", list(available_metrics.keys()), key="scatter_y")

            # Retrieve selected metric details
            metrics_to_use = [available_metrics[selected_metric_x][0], available_metrics[selected_metric_y][0]]
            metrics_label = [available_metrics[selected_metric_x][1], available_metrics[selected_metric_y][1]]
            metrics_unit = [available_metrics[selected_metric_x][2], available_metrics[selected_metric_y][2]]

            st.write(filtered_df)

            # Check before generating scatter plot
            if st.button("Generate Scatter Plot"):
                if filtered_df.empty:
                    st.error("⚠️ No data available for the selected teams.")
                else:
                    fig = plot_scatter_chart(filtered_df, [team_1, team_2], metrics_to_use, metrics_label, metrics_unit)
                    st.pyplot(fig)

        else:
            st.error("⚠️ No data available. Please upload a dataset first.")

    elif page == "Radar Plot":
        # Vérifie si le dataframe contient bien des données
        if not st.session_state.df.empty:
            st.header("🛡️ Radar Plot")

            st.subheader("Player Selection")

            # Sélection du joueur
            player = st.selectbox("Select a player", st.session_state.df['player_name'].unique())

            # Récupérer les postes joués par ce joueur
            player_positions = st.session_state.df[st.session_state.df['player_name'] == player]['group'].unique()

            # Si le joueur a plusieurs postes, afficher un selectbox pour choisir
            if len(player_positions) > 1:
                position = st.selectbox("Select a position for this player", player_positions)
            else:
                position = player_positions[0]  # S'il n'a qu'un poste, on le sélectionne automatiquement

            # Filtrer le dataframe avec le joueur et son poste sélectionné
            player_row = st.session_state.df[(st.session_state.df['player_name'] == player) & (st.session_state.df['group'] == position)]

            st.subheader("Comparison Filters")
        
            col1, col2 = st.columns(2)

            with col1:
                position = st.selectbox("Select a position", ["All"] + list(st.session_state.df['group'].unique()), key="position")

            with col2:
                # Convertir la colonne 'player_birthdate' en datetime
                st.session_state.df['player_birthdate'] = pd.to_datetime(st.session_state.df['player_birthdate'], errors='coerce')

                # Calculer l'âge des joueurs
                today = datetime.today()
                st.session_state.df['age'] = st.session_state.df['player_birthdate'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

                # Filtrer les joueurs en fonction de leur âge
                min_age, max_age = st.slider(
                    "Select age range",
                    min_value=int(st.session_state.df['age'].min()), 
                    max_value=int(st.session_state.df['age'].max()), 
                    value=(int(st.session_state.df['age'].min()), int(st.session_state.df['age'].max())), 
                    key="age"
                )

            championnats = st.multiselect("Select competitions", st.session_state.df['competition_name'].unique(), key="championnats")

            # Copie du dataframe de base pour éviter de modifier `st.session_state.df` directement
            filtered_df = st.session_state.df.copy()

            # 🔹 Filtrer par position
            if position != "All":
                filtered_df = filtered_df[filtered_df['group'] == position]

            # 🔹 Filtrer par âge
            filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

            # 🔹 Filtrer par compétitions
            filtered_df = filtered_df[filtered_df['competition_name'].isin(championnats)]

            # Filtrer par clubs
            clubs = st.multiselect("Select clubs", filtered_df['team_name'].unique(), key="clubs")
            if clubs:
                filtered_df = filtered_df[filtered_df['team_name'].isin(clubs)]

            # Si le player sélectionné n'est pas dans le dataframe filtré, on l'ajoute en première place
            if player not in filtered_df['player_name'].values:
                # Concaténation avec le joueur en premier
                filtered_df = pd.concat([player_row, filtered_df], ignore_index=True)
            else:
                # On le retire pour le remettre en premier
                filtered_df = filtered_df[filtered_df['player_name'] != player]
                filtered_df = pd.concat([player_row, filtered_df], ignore_index=True)

            st.write(filtered_df)

            # Détection du dataset actif
            if "dataset_type" in st.session_state:
                if st.session_state.dataset_type == "Off-Ball Runs":
                    available_metrics = RUNS_OPTIONS
                else:
                    available_metrics = PHYSICAL_OPTIONS
            else:
                available_metrics = PHYSICAL_OPTIONS  # Valeur par défaut au cas où

            if st.button("Generate Radar"):
                fig = plot_radar_chart(filtered_df, player, available_metrics)
                st.pyplot(fig)
        else:
            st.error("⚠️ No data available. Please upload a dataset first.")
    
    elif page == "Recommendation System":
        st.header("🔮 Recommendation System")

        col1, col2 = st.columns(2)

        with col1:
            position = st.selectbox("Select a position", ["All"] + list(st.session_state.df['group'].unique()), key="position")

        with col2:
            # Convertir la colonne 'player_birthdate' en datetime
            st.session_state.df['player_birthdate'] = pd.to_datetime(st.session_state.df['player_birthdate'], errors='coerce')

            # Calculer l'âge des joueurs
            today = datetime.today()
            st.session_state.df['age'] = st.session_state.df['player_birthdate'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

            # Filtrer les joueurs en fonction de leur âge
            min_age, max_age = st.slider(
                "Select age range",
                min_value=int(st.session_state.df['age'].min()), 
                max_value=int(st.session_state.df['age'].max()), 
                value=(int(st.session_state.df['age'].min()), int(st.session_state.df['age'].max())), 
                key="age"
            )

        championnats = st.multiselect("Select competitions", st.session_state.df['competition_name'].unique(), key="championnats")

        # Copie du dataframe de base pour éviter de modifier `st.session_state.df` directement
        filtered_df = st.session_state.df.copy()

        # 🔹 Filtrer par position
        if position != "All":
            filtered_df = filtered_df[filtered_df['group'] == position]

        # 🔹 Filtrer par âge
        filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

        # 🔹 Filtrer par compétitions
        filtered_df = filtered_df[filtered_df['competition_name'].isin(championnats)]

        # Filtrer par clubs
        clubs = st.multiselect("Select clubs", filtered_df['team_name'].unique(), key="clubs")
        if clubs:
            filtered_df = filtered_df[filtered_df['team_name'].isin(clubs)]

        # Détection du dataset actif
        if "dataset_type" in st.session_state:
            if st.session_state.dataset_type == "Off-Ball Runs":
                available_metrics = RUNS_OPTIONS
            else:
                available_metrics = PHYSICAL_OPTIONS
        else:
            available_metrics = PHYSICAL_OPTIONS  # Valeur par défaut au cas où

        st.subheader("Advanced Filtering (Top % Players)")

        # Select metrics for filtering
        selected_metrics = st.multiselect("Select metrics", list(available_metrics.keys()), key="selected_metrics")

        # Create threshold sliders
        thresholds = {}
        for metric in selected_metrics:
            thresholds[metric] = st.slider(f"Select top % for {metric}", min_value=0, max_value=100, value=50, step=5, key=metric)

        if st.button("Generate Recommendations"):
            df_final = filtered_df.copy()

            if selected_metrics:
                # Compute the thresholds for all selected metrics at once
                quantile_values = {metric: df_final[available_metrics[metric][0]].quantile(thresholds[metric] / 100)
                                for metric in selected_metrics}

                # Apply filtering across all selected metrics
                for metric in selected_metrics:
                    metric_to_use, _, _ = available_metrics[metric]
                    df_final = df_final[df_final[metric_to_use] >= quantile_values[metric]]

            st.write(df_final)

if __name__ == "__main__":
    # Authentification utilisateur
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username", type="default")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if username == st.secrets["API"]["API_USERNAME"] and password == st.secrets["API"]["API_PASSWORD"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    
    if st.session_state.authenticated:
        with st.spinner("Loading data from SkillCorner..."):
            try:
                # Charger les datasets depuis le cache
                df_off_ball_runs = load_off_ball_runs_data()
                df_physical = load_physical_data()

                # Prétraitement des données physiques
                df_physical = prétraitement_physical(df_physical)
                
            except Exception as e:
                st.error(f"⚠️ Error fetching data from API: {str(e)}")
                st.stop()

        # Lancer l'application une fois les données chargées
        st_action(df_off_ball_runs, df_physical)