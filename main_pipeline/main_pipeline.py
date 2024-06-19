"""Main pipeline for training and updating the classifier model."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from expand_training import get_training
from phrase_matching import AssistantPhraseMatcher

ENTITIES_PATH = Path(__file__).parent.parent / 'custom_entities'
TRAINING_PATH = Path(__file__).parent.parent / 'model_training'

MODEL_PATH = Path(__file__).parent / 'model'
SPACY_MODEL_PATH = Path(__file__).parent / 'entity_model'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('main_pipeline/main_pipeline.log')
f_format = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(funcName)s | '
    '%(lineno)s | %(message)s'
)
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.setLevel(level=logging.INFO)

# Produce features for dataset given a vectorizer
def get_features(vectorizer: CountVectorizer, training_data):
    train_features = vectorizer.fit_transform(training_data)
    # print(vectorizer.get_feature_names())
    # print(train_features.toarray())

    # Apply the vectorizer to create features for the dev and test data
    dev_features = vectorizer.transform(training_data)

    # Check out the data shapes
    logger.info(
        'Model vectorizer vocab size: %s',
        str(len(vectorizer.vocabulary_))
    )
    logger.info('Train features shape: %s', str(train_features.shape))
    logger.info('Dev features shape: %s', str(dev_features.shape))

    return vectorizer, train_features, dev_features

def update_logistic_regression_model(c_value, features, labels, vectorizer):
    model = LogisticRegression(C=c_value, penalty='l2', random_state=0)
    model.fit(features, labels)
    with MODEL_PATH.open('wb') as f:
        pickle.dump([vectorizer, model], f)
    now = datetime.now()
    logger.info('Model updated at %s', str(now.strftime('%H:%M:%S')))

def prepare_entity_terms(entities_path: Path) -> list[tuple[tuple[str], str]]:
    """Prepare entity terms for training."""
    entity_terms = []
    for entity_file in entities_path.iterdir():
        if entity_file.is_file():
            with entity_file.open(mode='r', encoding='utf-8') as source:
                entity_dict = json.load(source)
            for label in entity_dict['labels']:
                synonyms = []
                eid = label['label']
                for custom_entities in label['custom_entities'].values():
                    synonyms.extend(custom_entities)
                entity_terms.append((tuple(synonyms), eid))
    return entity_terms

def update_entity_model(
        assistant_phrase_matcher: AssistantPhraseMatcher
    ) -> None:
    entity_terms = prepare_entity_terms(ENTITIES_PATH)
    for term in entity_terms:
        assistant_phrase_matcher.add_terms(term[0], term[1])
    with SPACY_MODEL_PATH.open('wb') as f:
        pickle.dump(assistant_phrase_matcher, f)
    now = datetime.now()
    logger.info('Entity model updated at %s', str(now.strftime('%H:%M:%S')))

def main():
    """Main for main_pipeline which trains the classifier."""
    train_data, train_labels = get_training(TRAINING_PATH, ENTITIES_PATH)

    # Using full vocab get features
    vectorizer, train_features, dev_features = get_features(
        vectorizer=CountVectorizer(
            binary=True,
            analyzer='word',
            ngram_range=(1,3)
        ),
        training_data=train_data
    )

    update_logistic_regression_model(
        0.1,
        train_features,
        train_labels,
        vectorizer
    )

    update_entity_model(AssistantPhraseMatcher())

if __name__ == '__main__':
    main()
