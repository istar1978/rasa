DATA_FOLDERS=(
  "data/CCPE"
  "data/BTC"
  "data/WNUT17"
  "data/Ritter"
)
PIPELINES=(
  "combined"
  "tf-lstm"
  "tf-transformer"
)
TYPO=(
  "no"
#  "yes"
)
TRAIN_FRAC=(
  "0.8"
)
RUNS=1
OUTPUT="results"
REPORTING="no"
EPOCHS=50

LOG_FOLDER="logs"
if [ -d "$LOG_FOLDER" ]; then
    echo "$LOG_FOLDER already exists."
else
    mkdir $LOG_FOLDER
fi

for data_folder in "${DATA_FOLDERS[@]}";
do
    LOCAL_DATA_FOLDER="tmp/$data_folder"

    LOG_FILE="$LOG_FOLDER/$(basename $data_folder).txt"
    if [ -f "$LOG_FILE" ]; then
        rm $LOG_FILE
    fi

    NUMBER_OF_EXPERIMENTS=$((${#PIPELINES[@]} * ${#TYPO[@]} * ${#TRAIN_FRAC[@]}))
    CURRENT_EXPERIMENT=0

    # report start of evaluation
    if [[ "$REPORTING" == "yes" ]]; then
        NOTIFICATION="<@UGW2TPJF8> Starting evaluation on $data_folder."
        curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$NOTIFICATION\"}" https://hooks.slack.com/services/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
    fi

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
                    echo "Experiment $CURRENT_EXPERIMENT/$NUMBER_OF_EXPERIMENTS on $data_folder is running."

                    if [[ "$typo" == "yes" ]]; then
                        python scripts/evaluate.py --typo --output $OUTPUT --runs $RUNS --epochs $EPOCHS --pipeline $pipeline --train-frac $train_frac $LOCAL_DATA_FOLDER > $LOG_FILE
                    else
                        python scripts/evaluate.py --output $OUTPUT --runs $RUNS --epochs $EPOCHS --pipeline $pipeline --train-frac $train_frac $LOCAL_DATA_FOLDER > $LOG_FILE
                    fi

                    if [ $? -ne 0 ]; then
                        NOTIFICATION="<@UGW2TPJF8> Experiment $CURRENT_EXPERIMENT/$NUMBER_OF_EXPERIMENTS on $data_folder failed!"
                        echo $NOTIFICATION
                        if [[ "$REPORTING" == "yes" ]]; then
                            curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$NOTIFICATION\"}" https://hooks.slack.com/services/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
                        fi
                    fi

                    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
                done
            done
        done

    # remove local data
    if [ -d "$LOCAL_DATA_FOLDER" ]; then
        rm -r $LOCAL_DATA_FOLDER
    fi

    # report end of evaluation
    if [[ "$REPORTING" == "yes" ]]; then
        NOTIFICATION="<@UGW2TPJF8> Finished evaluation on $data_folder."
        curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$NOTIFICATION\"}" https://hooks.slack.com/services/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
    fi

done
