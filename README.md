# aide globale
python main.py --help

# construire les données (première exécution)
python main.py build-data --kaggle --player-gameweeks --fixtures --odds

# prédire une gameweek
python main.py predict --model gw_seasonal_gbm --season "2022/23" --gw 1 --head 10

# évaluer un modèle
python main.py evaluate --model gw_seasonal_gbm --season "2022/23"

# benchmark vs bookmakers
python main.py compare-bookmakers --model gw_seasonal_gbm --season "2022/23" --examples 10
