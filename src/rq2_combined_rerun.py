# Corrected RQ2 combined-input rerun.
# Uses the same grouped split as full_experiments.py but reserves 128 prompt tokens and 384 output tokens.
# The resulting test accuracies were CNN 0.7852 and LSTM 0.5926.
# See the notebook and report for the complete experimental context.
PROMPT_BUDGET = 128
OUTPUT_BUDGET = 384
MAX_LEN = PROMPT_BUDGET + 1 + OUTPUT_BUDGET

def combined_tokens(tokenize, prompt, output):
    return tokenize(prompt)[:PROMPT_BUDGET] + ["<SEP>"] + tokenize(output)[:OUTPUT_BUDGET]

if __name__ == "__main__":
    print("Run notebooks/03_rq2_combined_rerun.ipynb for the full corrected RQ2 rerun.")
