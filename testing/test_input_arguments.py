from consts import DATASET_PATHS, ALLOWED_MODEL_NAMES, ALLOWED_TEXT_COLUMNS, GENSIM_MODEL_PATHS, \
    ALLOWED_TOKENIZED_TEXT_COLUMNS


def test_model_names(model_names):
    assert type(model_names) == list,  "Model names type should be list"
    assert None not in model_names, f"No None values are allowed in {model_names}"
    assert len(model_names) > 0, "Length of model names needs to be greater than 0"
    for model_name in model_names:
        assert model_name in ALLOWED_MODEL_NAMES, f"Model name {model_name} should be in {ALLOWED_MODEL_NAMES}"


def test_column_names(text_columns):
    assert type(text_columns) == list, "Text columns type should be list"
    assert None not in text_columns, f"No None values are allowed in {text_columns}"
    assert len(text_columns) > 0, "Length of text columns needs to be greater than 0"
    for text_column in text_columns:
        assert text_column in ALLOWED_TEXT_COLUMNS, f"Text column {text_column} should be in {ALLOWED_TEXT_COLUMNS}"


def test_tokenized_column_names(tokenized_text_columns):
    assert type(tokenized_text_columns) == list, "Tokenized text columns type should be list"
    assert None not in tokenized_text_columns, f"No None values are allowed in {tokenized_text_columns}"
    assert len(tokenized_text_columns) > 0, "Length of tokenized text columns needs to be greater than 0"
    for tokenized_text_column in tokenized_text_columns:
        assert tokenized_text_column in ALLOWED_TOKENIZED_TEXT_COLUMNS, f"Text tokenized column {tokenized_text_column} should be in {ALLOWED_TOKENIZED_TEXT_COLUMNS}"


def test_dataset_names(dataset_names):
    assert type(dataset_names) == list, "Dataset names type should be list"
    assert None not in dataset_names, f"No None values are allowed in {dataset_names}"
    assert len(dataset_names) > 0, "Length of dataset names needs to be greater than 0"
    for dataset_name in dataset_names:
        assert dataset_name in list(DATASET_PATHS.keys()), f"Dataset name: {dataset_name} should be in {list(DATASET_PATHS.keys())}"


def test_embedding_names(embedding_names):
    assert type(embedding_names) == list, "Embedding names type should be list"
    assert None not in embedding_names, f"No None values are allowed in {embedding_names}"
    assert len(embedding_names) > 0, "Length of embedding names needs to be greater than 0"
    for embedding_name in embedding_names:
        assert embedding_name in list(GENSIM_MODEL_PATHS.keys()), f"Embedding name: {embedding_name} should be in {list(GENSIM_MODEL_PATHS.keys())}"
