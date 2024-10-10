if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <qrel_path> <base_dir>"
    exit 1
fi

QRELS="$1"
BASE_DIR="$2"

run_trec_eval() {
    local metric=$1
    local res_file=$2
    local full_rel="${3:-False}"

    if [ "$full_rel" = "False" ]; then
        ./trec_eval -m "$metric" "$QRELS" "$res_file"
    else
        ./trec_eval -m "$metric" -l3 "$QRELS" "$res_file"
    fi
}

find "$BASE_DIR" -type f -name "retrieval_res*" | while read -r res_file; do
    echo "Running evaluations on $res_file"
    
    run_trec_eval "bpref" "$res_file"
    run_trec_eval "bpref" "$res_file" 1
    run_trec_eval "ndcg" "$res_file"
done

