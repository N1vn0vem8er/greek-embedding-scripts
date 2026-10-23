import os
import optuna
import pandas as pd
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from scipy.stats import spearmanr

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def run_optuna_study(
    corpus_file_path: str,
    eval_csv_path: str,
    n_trials: int = 50,
    workers: int = 4,
    db_path: str = 'fasttext_optuna.db',
    output_csv_path: str = 'optuna_fasttext_results.csv',
) -> pd.DataFrame:
    eval_df = load_eval_data(eval_csv_path)
    sentences = LineSentence(corpus_file_path)

    def objective(trial: optuna.Trial) -> float:
        vector_size = trial.suggest_categorical('vector_size', [100, 150, 300])
        window = trial.suggest_int('window', 2, 20)
        min_count = trial.suggest_int('min_count', 1, 10)
        sg = trial.suggest_categorical('sg', [0, 1])
        epochs = trial.suggest_int('epochs', 20, 50)
        min_n = trial.suggest_int('min_n', 3, 5)
        max_n = trial.suggest_int('max_n', max(min_n, 5), 8)

        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            min_n=min_n,
            max_n=max_n,
            epochs=epochs,
            workers=workers,
            seed=42,
        )

        score = evaluate_model(model, eval_df)
        return score

    storage_name = f'sqlite:///{db_path}'
    study = optuna.create_study(
        study_name='fasttext_optimization',
        storage=storage_name,
        load_if_exists=True,
        direction='maximize',
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    results_df = study.trials_dataframe()
    results_df = results_df.sort_values(by='value', ascending=False).reset_index(drop=True)
    results_df.to_csv(output_csv_path, index=False)

    return results_df


if __name__ == '__main__':
    CORPUS_PATH = '../pg_corpus_lemmatized_clean.txt'
    EVAL_CSV_PATH = 'test_data.csv'
    OUTPUT_CSV_PATH = 'optuna_fasttext_results.csv'

    results_df = run_optuna_study(
        corpus_file_path=CORPUS_PATH,
        eval_csv_path=EVAL_CSV_PATH,
        n_trials=100,
        workers=32,
        db_path='fasttext_optuna.db',
        output_csv_path=OUTPUT_CSV_PATH,
    )

    best_trial = results_df.iloc[0]
    print(f"Best correlation: {best_trial['value']:.4f}")
    print("Params:")
    for col in results_df.columns:
        if col.startswith('params_'):
            print(f"  {col.replace('params_', '')}: {best_trial[col]}")
