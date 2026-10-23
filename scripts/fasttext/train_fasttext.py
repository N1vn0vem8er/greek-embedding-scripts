import os
from gensim.models import FastText
from gensim.models.word2vec import LineSentence

CORPUS_PATH = "hexaemeron_merged_lemmatized.txt"
MODEL_SAVE_PATH = "fasttext.model"

PARAMS = {
    "vector_size": 300,
    "window": 20,
    "min_count": 3,
    "workers": 32,
    "min_n": 3,
    "max_n": 8,
    "epochs": 100,
    "sg": 0,
    "seed": 42
}

def train_fasttext():
    if not os.path.exists(CORPUS_PATH):
        print(f"File '{CORPUS_PATH}' does not exists")
        return None


    sentences = LineSentence(CORPUS_PATH)

    model = FastText(
        sentences=sentences,
        vector_size=PARAMS["vector_size"],
        window=PARAMS["window"],
        min_count=PARAMS["min_count"],
        workers=PARAMS["workers"],
        min_n=PARAMS["min_n"],
        max_n=PARAMS["max_n"],
        sg=PARAMS["sg"],
        epochs=PARAMS["epochs"],
        seed=PARAMS["seed"]
    )


    model.save(MODEL_SAVE_PATH)
    print(f"Saved: {MODEL_SAVE_PATH}")
    return model

def test_model(model):

    test_word = "λόγος"

    if test_word in model.wv:
        print(f"Words similar to '{test_word}':")
        similar_words = model.wv.most_similar(test_word, topn=5)
        for word, score in similar_words:
            print(f"  - {word}: {score:.4f}")
    else:
        print(f"Word '{test_word}' not in vocab.")

    oov_word = "βασileus_test"
    print(f"\nOOV word: '{oov_word}'")
    try:
        similar_oov = model.wv.most_similar(oov_word, topn=3)
        print(f"Words similar to OOV '{oov_word}':")
        for word, score in similar_oov:
            print(f"  - {word}: {score:.4f}")
    except KeyError:
        print("Model was unable to generate n-grams for this word.")

if __name__ == "__main__":
    trained_model = train_fasttext()

    if trained_model:
        test_model(trained_model)
