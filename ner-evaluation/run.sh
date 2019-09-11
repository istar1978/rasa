DATA_FOLDERS=(
  "data/AddToPlaylist"
  "data/BookRestaurant"
  "data/GetWeather"
  "data/RateBook"
  "data/SearchCreativeWork"
  "data/SearchScreeningEvent"
  "data/BTC"
  "data/re3d"
  "data/WNUT17"
  "data/Ritter"
)
PIPELINES=(
  "default"
  "default-spacy"
  "spacy"
  "flair"
)
TYPO=(
  "yes"
  "no"
)
TRAIN_FRAC=(
  "0.5"
  "0.8"
)
RUNS=5
OUTPUT="results"
REPORTING="no"

for data_folder in "${DATA_FOLDERS[@]}";
do
    if [[ "$REPORTING" == "yes" ]]; then
        NOTIFICATION="<@UGW2TPJF8> Starting evaluation on $data_folder."
        curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$NOTIFICATION\"}" https://hooks.slack.com/services/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
    fi

    LOCAL_DATA_FOLDER="tmp/$data_folder"

    NUMBER_OF_EXPERIMENTS=$((${#PIPELINES[@]} * ${#TYPO[@]} * ${#TRAIN_FRAC[@]}))
    CURRENT_EXPERIMENT=0

    # download data
    if [ -d "$LOCAL_DATA_FOLDER" ]; then
        echo "$LOCAL_DATA_FOLDER already exists."
    else
        echo "Download $LOCAL_DATA_FOLDER."
        mkdir -p $LOCAL_DATA_FOLDER
        gsutil cp -R gs://artifacts.rasa-research.appspot.com/ner-datasets/$data_folder $LOCAL_DATA_FOLDER
    fi

    # run evaluation
    for pipeline in "${PIPELINES[@]}";
        do
            for train_frac in "${TRAIN_FRAC[@]}";
            do
                for typo in "${TYPO[@]}";
                do

                    if ! ((CURRENT_EXPERIMENT % 4)); then
                        if [[ "$REPORTING" == "yes" ]]; then
                            NOTIFICATION="<@UGW2TPJF8> Experiment $CURRENT_EXPERIMENT/$NUMBER_OF_EXPERIMENTS on $data_folder running."
                            curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$NOTIFICATION\"}" https://hooks.slack.com/services/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
                        fi
                    fi

                    if [[ "$typo" == "yes" ]]; then
                        python scripts/evaluate.py --typo --output $OUTPUT --runs $RUNS --pipeline $pipeline --train-frac $train_frac $LOCAL_DATA_FOLDER
                    else
                        python scripts/evaluate.py --output $OUTPUT --runs $RUNS --pipeline $pipeline --train-frac $train_frac $LOCAL_DATA_FOLDER
                    fi

                    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
                done
            done
        done

    # remove local data
    if [ -d "$LOCAL_DATA_FOLDER" ]; then
        rm -r $LOCAL_DATA_FOLDER
    fi

    if [[ "$REPORTING" == "yes" ]]; then
        NOTIFICATION="<@UGW2TPJF8> Finished evaluation on $data_folder."
        curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$NOTIFICATION\"}" https://hooks.slack.com/services/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
    fi

done