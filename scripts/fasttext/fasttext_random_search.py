import itertools
import os
import random
import pandas as pd
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from scipy.stats import spearmanr


def load_eval_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['word1'] = df['word1'].astype(str)
    df['word2'] = df['word2'].astype(str)
    df['gold_score'] = pd.to_numeric(df['gold_score'], errors='coerce')
    return df.dropna(subset=['gold_score'])


def evaluate_model(model: FastText, eval_df: pd.DataFrame) -> float:
    preds, gold_scores = [], []

    for _, row in eval_df.iterrows():
        w1, w2, gold = row['word1'], row['word2'], row['gold_score']
        try:
            sim = model.wv.similarity(w1, w2)
            preds.append(sim)
            gold_scores.append(gold)
        except Exception:
            continue

    if len(preds) < 2:
        return -1.0

    correlation, _ = spearmanr(preds, gold_scores)
    return correlation if not pd.isna(correlation) else -1.0


def fasttext_random_search(
    corpus_file_path: str,
    eval_csv_path: str,
    param_distributions: dict,
    n_iter: int = 50,
    workers: int = 4,
    output_csv_path: str = 'random_search_fasttext_results.csv',
    save_every: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    eval_df = load_eval_data(eval_csv_path)

    keys, values = zip(*param_distributions.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_combinations = [p for p in all_combinations if p.get('min_n', 0) <= p.get('max_n', 0)]

    random.seed(random_state)
    random.shuffle(all_combinations)

    results = []
    evaluated_tuples = set()

    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        results = existing_df.to_dict('records')

        for _, row in existing_df.iterrows():
            config_tuple = tuple(row[k] for k in keys)
            evaluated_tuples.add(config_tuple)

        print(f"Loaded {len(evaluated_tuples)} from file: {output_csv_path}")

    unevaluated_combinations = [
        p for p in all_combinations if tuple(p[k] for k in keys) not in evaluated_tuples
    ]

    combinations_to_run = unevaluated_combinations[:n_iter]

    if not combinations_to_run:
        return pd.DataFrame(results).sort_values(by='spearman_correlation', ascending=False)

    sentences = LineSentence(corpus_file_path)

    for idx, params in enumerate(combinations_to_run, 1):
        print(f"[{idx}/{len(combinations_to_run)}] Training with: {params}")

        model = FastText(
            sentences=sentences,
            vector_size=params.get('vector_size', 100),
            window=params.get('window', 5),
            min_count=params.get('min_count', 5),
            sg=params.get('sg', 0),
            min_n=params.get('min_n', 3),
            max_n=params.get('max_n', 6),
            epochs=params.get('epochs', 5),
            workers=workers,
            seed=random_state,
        )

        score = evaluate_model(model, eval_df)

        result_row = {**params, 'spearman_correlation': score}
        results.append(result_row)
        print(f"Spearman correlation: {score:.4f}\n")

        if idx % save_every == 0 or idx == len(combinations_to_run):
            temp_df = (
                pd.DataFrame(results)
                .sort_values(by='spearman_correlation', ascending=False)
                .reset_index(drop=True)
            )
            temp_df.to_csv(output_csv_path, index=False)

    results_df = (
        pd.DataFrame(results)
        .sort_values(by='spearman_correlation', ascending=False)
        .reset_index(drop=True)
    )

    return results_df


if __name__ == '__main__':
    param_distributions = {
        'vector_size': [100, 150, 300],
        'window': [2, 5, 10, 20],
        'min_count': [1, 3, 10],
        'sg': [0, 1],
        'epochs': [20, 50],
        'min_n': [3, 4, 5],
        'max_n': [5, 6, 7, 8],
    }

    CORPUS_PATH = '../pg_corpus_lemmatized_clean.txt'
    EVAL_CSV_PATH = 'test_data.csv'
    OUTPUT_CSV_PATH = 'random_search_fasttext_results.csv'

    results_df = fasttext_random_search(
        corpus_file_path=CORPUS_PATH,
        eval_csv_path=EVAL_CSV_PATH,
        param_distributions=param_distributions,
        n_iter=200,
        workers=32,
        output_csv_path=OUTPUT_CSV_PATH,
        save_every=5,
        random_state=42,
    )

