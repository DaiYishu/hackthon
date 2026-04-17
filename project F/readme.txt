Description du projet F 
1. Fichier à la racine
note.txt
Document décrivant le déroulement du projet, l’analyse des données et les conclusions.

stoch_model1_arima.py
Baseline ARIMA par classe d’actifs pour prédire stoch_k(t+15).

stoch_model2_mean_reversion.py
Modèle de mean reversion par régimes (LOW/MID/HIGH) via régression linéaire pour prédire stoch_k(t+15).

stoch_model3_xgb.py
Modèle XGBoost par classe (lags, momentum, mean reversion, interactions) pour prédire stoch_k(t+15).

stoch_model4_lstm.py
Modèle LSTM séquentiel (fenêtre 20) pour prédire stoch_k(t+15).

2. Organisation des répertoires
2.1 data/

Contient les données et les résultats intermédiaires, du niveau 5 minutes au niveau 15 minutes.

*.xlsx : données OHLCV brutes (5 minutes, par actif)
concatenated_raw.csv : données brutes concaténées

cleaned_data/
Première phase de nettoyage (sans interpolation ni création de données).

cleaned_dataset : données nettoyées
quality_report_by_asset : rapport de qualité
suspicious_rows : observations suspectes
gap_summary : statistiques des interruptions (time gaps)

cleaned_data2.0/
Deuxième phase (validation légère, processed2.0), conservant les gaps réels.

processed2.0 : données finales
processed2.0_report : rapport par actif
processed2.0_suspicious : anomalies détectées

data_15mins/
Données utilisées pour la modélisation.

resampled_15min : données rééchantillonnées (sans interpolation)
macd_features_15min : variables EMA/MACD
stoch_features_15min : indicateur Stochastic (14,3,5)
2.2 data cleaner/

Scripts de traitement des données.

preprocess_timeseries.py : nettoyage initial
processed2_validation.py : validation et construction de processed2.0
resample_15min.py : passage à 15 minutes
build_macd_features_15min.py : calcul des indicateurs techniques
build_stoch_features_15min.py : calcul de l’indicateur Stochastic (14,3,5)
2.3 model_MACD/

Modèles pour prédire MACD(t+15).

model1_arima.py : baseline ARIMA par classe d’actifs
model2_structural_arima.py : ARIMA sur EMA puis reconstruction du MACD
model3_arimax_jump.py : ARIMAX avec variables exogènes (gaps, régimes)
model4_ML.py : modèles supervisés (Ridge, XGBoost)
2.4 results_MACD/

Résultats des modèles (CSV), organisés par méthode :
ARIMA, EMA structurel, ARIMAX, modèles supervisés.

2.5 sujet/
ha_time_series.docx : énoncé du projet
2.6 pycache/

Fichiers cache Python, non pertinents.

2.7 results_stoch/

Résultats des modèles STOCH (CSV), organisés par méthode.

model1_arima/ : baseline ARIMA (stoch_arima_results, stoch_arima_summary)
model2_mean_reversion/ : mean reversion par régimes (stoch_mean_reversion_results, stoch_mean_reversion_summary)
model3_xgb/ : modèle XGBoost (stoch_xgb_results, stoch_xgb_summary)
model4_lstm/ : modèle LSTM (stoch_lstm_results)

results_STOCH/
Répertoire de résultats équivalent (ancienne convention de nommage, mêmes types de sorties).

3. Flux de traitement

Données brutes (5 min)
→ nettoyage
→ validation (processed2.0)
→ rééchantillonnage (15 min)
→ construction des variables (MACD, STOCH)
→ modélisation
→ résultats
