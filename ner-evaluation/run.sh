DATA_FOLDER="data/AddToPlaylist"
PIPELINES=("default" "default-spacy" "spacy" "flair")
TYPO=("yes" "no")
TRAIN_FRAC=("0.5" "0.8")
RUNS=5
OUTPUT="results"
LOCAL_DATA_FOLDER="tmp/$DATA_FOLDER"

# download data
mkdir -p $LOCAL_DATA_FOLDER
gsutil cp -R gs://artifacts.rasa-research.appspot.com/ner-datasets/$DATA_FOLDER $LOCAL_DATA_FOLDER

# run evaluation
for pipeline in "${PIPELINES[@]}";
    do
        for train_frac in "${TRAIN_FRAC[@]}";
        do
            for typo in "${TYPO[@]}";
            do
                if [[ "$typo" == "yes" ]]; then
                    python scripts/evaluate.py --typo --output $OUTPUT --runs $RUNS --pipeline $pipeline --train-frac $train_frac $LOCAL_DATA_FOLDER
                else
                    python scripts/evaluate.py --output $OUTPUT --runs $RUNS --pipeline $pipeline --train-frac $train_frac $LOCAL_DATA_FOLDER
                fi
            done
        done
    done

# remove local data
rm -r $LOCAL_DATA_FOLDER