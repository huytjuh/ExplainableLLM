from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "_01_NLP" / "_01_preprocessing"

sys.path.insert(0, str(PROJECT_ROOT))

from src._01_NLP._01_preprocessing._00_normalize import Normalizer
from src._01_NLP._01_preprocessing._01_tokenize import Tokenizer, TokenizerConfig
from src._01_NLP._01_preprocessing._02_stemming import Stemmer
from src._01_NLP._01_preprocessing._03_lemmatize import Lemmatizer

from src._01_NLP._02_feature_engineering._01_lexical import Lexical
from src._01_NLP._02_feature_engineering._02_phonetics import Phonetics
from src._01_NLP._02_feature_engineering._03_oov import OutOfVocabulary
# from src._01_NLP._02_feature_engineering._02_spelling import Spelling 

SYNTHETIC_COMPLAINTS = [
    "The delivery was late en de klantenservice reageerde niet!!!",
    "Ik kan niet inloggen; checkout keeps crashing at payment.",
    "Refund was not processed en de factuur klopt niet.",
    "Mijn order is missing, and support zegt dat ik moet wachten...",
]


def main() -> None:
    print(SYNTHETIC_COMPLAINTS)

    normalizer = Normalizer()
    normalized = [normalizer(complaint).text for complaint in SYNTHETIC_COMPLAINTS]
    print(normalized)

    tokenizer = Tokenizer(TokenizerConfig(english=False, dutch=False))
    tokenized = [tokenizer(complaint) for complaint in normalized]
    print(tokenized)

    # stemmer = Stemmer(tokenizer)
    # stemmed = [stemmer(complaint).stemmed_text for complaint in normalized]
    # print(stemmed)

    # lemmatizer = Lemmatizer(tokenizer)
    # lemmatized = [lemmatizer(complaint).lemmatized_text for complaint in normalized]
    # print(lemmatized)

    lexical = Lexical()
    feature_lexical = [lexical(complaint) for complaint in tokenized]
    print(feature_lexical)

    phonetics = Phonetics()
    feature_phonetics = [phonetics(complaint) for complaint in tokenized]
    print(feature_phonetics)

    oov = OutOfVocabulary()
    feature_oov = [oov(complaint) for complaint in tokenized]
    print(feature_oov)


if __name__ == "__main__":
    main()
