import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import sklearn
import umap
import tensorflow as tf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from encoder import build_model, attention
from sklearn.manifold import TSNE
def generate_vectors(algorithm, target, vectors, dimension,limit=-1, test_size=0.8, batch_size=64, weighting=None):
    # Load Original BERT Vectors
    # Load Dataset

    target = target#[0:limit]
    vectors = vectors#[0:limit]

    dims = vectors[0].shape

    #Non-Supervised Algorithms
    if algorithm in ["PCA", "UMAP", "PARAMETRIC_UMAP", "TSNE"]:
        if algorithm == "PCA":
            reduced_vectors = sklearn.decomposition.PCA(n_components=dimension).fit_transform(vectors)

        elif algorithm == "UMAP":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine'}
            reduced_vectors = umap.UMAP(**umap_args).fit(vectors).embedding_

        elif algorithm == "PARAMETRIC_UMAP":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(vectors)
            reduced_vectors = reducer.transform(vectors)
        elif algorithm == "TSNE":
            model = TSNE(n_components=dimension, n_jobs=-1)
            reduced_vectors = model.fit_transform(vectors)

    #Supervised Algorithms
    elif algorithm in ["LDA", "UMAP_SUPERVISED", "PARAMETRIC_UMAP_SUPERVISED", "UMAP_SUPERVISED_RNN", "UMAP_SUPERVISED_TRANSFORMER"]:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            vectors, target, test_size=test_size, random_state=42)
        #Create
        y_test = np.full_like(y_test, -1)
        del y_test
        del X_test
        #del vectors
        del target

        if algorithm == "LDA":
            try:

                lda = LinearDiscriminantAnalysis(n_components=dimension)
                lda.fit(X_train, y=y_train)
                reduced_vectors = lda.transform(vectors)
            except Exception as e:
                print(e)
                reduced_vectors = np.array(np.zeros(dimension))

        elif algorithm == "UMAP_SUPERVISED":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine'}
            reducer = umap.UMAP(**umap_args).fit(X_train, y_train)
            reduced_vectors = reducer.transform(vectors)

        elif algorithm == "PARAMETRIC_UMAP_SUPERVISED":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "batch_size": batch_size,
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y_train)
            reduced_vectors = reducer.transform(vectors)

        elif algorithm == "UMAP_SUPERVISED_RNN":
            # Construct the RNN with Attention (Hyperparams identified through parameter optimization with OpTuna)
            encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=dims),
                tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1192, activation="relu", return_sequences=True)),
                attention(return_sequences=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(933, activation="tanh", return_sequences=True)),
                attention(return_sequences=True),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1340),
                tf.keras.layers.Dense(units=dimension),
            ])
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "encoder": encoder,
                         "dims": dims,
                         "batch_size": batch_size,
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y=y_train)
            del X_train
            del y_train
            reduced_vectors = reducer.transform(vectors)

        elif algorithm == "UMAP_SUPERVISED_TRANSFORMER":
            # Construct the Transformer Model (Hyperparams identified through parameter optimization with OpTuna)
            #encoder = build_model(
            #    (dims),
            #    head_size=20,
            #    num_heads=16,
            #    ff_dim=768,
            #    num_transformer_blocks=3,
            #    mlp_units=[512,512],
            #    mlp_dropout=0.0003058954347195825,
            #    dropout=0.015244566146468075,
            #    n_classes=dimension,
            #)
            #TREC6: encoder = build_model(
            #TREC6:         (dims),
            #TREC6:                 head_size=10,
            #TREC6:                 num_heads=11,
            #TREC6:                 ff_dim=235,
            #TREC6:                 num_transformer_blocks=1,
            #TREC6:                 mlp_units=[210],
            #TREC6:                 mlp_dropout=0.0003058954347195825,
            #TREC6:                 dropout=0.015244566146468075,
            #TREC6:                 n_classes=dimension
            #TREC6:      )
            #TREC50: 0.41263440860215056
            encoder = build_model(
                (dims),
                head_size=2,
                num_heads=31,
                ff_dim=610,
                num_transformer_blocks=2,
                mlp_units=[928,569,564],
                mlp_dropout=0.0010533336845051193,
                dropout=0.055311304614280965,
                n_classes=dimension
            )
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "encoder": encoder,
                         "dims": dims,
                         "batch_size": batch_size,
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y=y_train)
            del X_train
            del y_train
            reduced_vectors = reducer.transform(vectors)

    return reduced_vectors


"""def generate_vectors_weighting(algorithm, target, vectors, dimension,limit=-1, test_size=0.8, batch_size=64, weighting=None):
    # Load Original BERT Vectors
    # Load Dataset

    target = target#[0:limit]
    vectors = vectors#[0:limit]

    dims = vectors[0].shape

    #Non-Supervised Algorithms
    if algorithm in ["PCA", "UMAP", "PARAMETRIC_UMAP", "TSNE"]:
        if algorithm == "PCA":
            reduced_vectors = sklearn.decomposition.PCA(n_components=dimension).fit_transform(vectors)

        elif algorithm == "UMAP":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine'}
            reduced_vectors = umap.UMAP(**umap_args).fit(vectors).embedding_

        elif algorithm == "PARAMETRIC_UMAP":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(vectors)
            reduced_vectors = reducer.transform(vectors)
        elif algorithm == "TSNE":
            model = TSNE(n_components=dimension, n_jobs=-1)
            reduced_vectors = model.fit_transform(vectors)

    # Supervised Algorithms
    elif algorithm in ["LDA", "UMAP_SUPERVISED", "PARAMETRIC_UMAP_SUPERVISED", "UMAP_SUPERVISED_RNN",
                       "UMAP_SUPERVISED_TRANSFORMER"]:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            vectors, target, test_size=test_size, random_state=42)
        # Create
        y_test = np.full_like(y_test, -1)
        vectors = np.vstack([X_train, X_test])
        target = np.concatenate([y_train,y_test])
        if algorithm == "UMAP_SUPERVISED":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine'}
            reducer = umap.UMAP(**umap_args).fit(X_train, y_train)
            reduced_vectors = reducer.transform(vectors)
        elif algorithm == "PARAMETRIC_UMAP_SUPERVISED":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "batch_size": batch_size,
                         "keras_fit_kwargs": {"verbose": 1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y_train)
            reduced_vectors = reducer.transform(vectors)
        elif algorithm == "UMAP_SUPERVISED_RNN":
            # Construct the RNN with Attention (Hyperparams identified through parameter optimization with OpTuna)
            encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=dims),
                tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1192, activation="relu", return_sequences=True)),
                attention(return_sequences=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(933, activation="tanh", return_sequences=True)),
                attention(return_sequences=True),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1340),
                tf.keras.layers.Dense(units=dimension),
            ])
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "encoder": encoder,
                         "dims": dims,
                         "batch_size": batch_size,
                         "keras_fit_kwargs": {"verbose": 1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y=y_train)
            del X_train
            del y_train
            reduced_vectors = reducer.transform(vectors)
        elif algorithm == "UMAP_SUPERVISED_TRANSFORMER":
            # Construct the Transformer Model (Hyperparams identified through parameter optimization with OpTuna)
            encoder = build_model(
                (dims),
                head_size= 10,
                num_heads= 11,
                ff_dim= 235,
                num_transformer_blocks= 1,
                mlp_units= 210,
                mlp_dropout= 0.0003058954347195825,
                dropout= 0.015244566146468075,
                n_classes=dimension,
            )
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "encoder": encoder,
                         "dims": dims,
                         "batch_size": batch_size,
                         "target_weight":weighting,
                         "keras_fit_kwargs": {"verbose": 1}}
            reduced_vectors = umap.parametric_umap.ParametricUMAP(**umap_args).fit_transform(vectors, y=target)
            #del X_train
            #del y_train
            #reduced_vectors = reducer.transform(vectors)
    return reduced_vectors"""

def load_dataset(dataset_name):
    if dataset_name == "20newsgroups":
        bert_vectors = np.array(joblib.load("resources/20Newsgroups_BERT.pkl"))
        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        target = data.target
    elif dataset_name == "trec6":
        bert_vectors = np.array(joblib.load("resources/trec_BERT.pkl"))
        data = pd.read_csv("data/trec.csv")
        target = data["label-coarse"]
    elif dataset_name == "trec50":
        bert_vectors = np.array(joblib.load("resources/trec_BERT.pkl"))
        data = pd.read_csv("data/trec.csv")
        target = data["label-fine"]
    elif dataset_name == "agnews":
        bert_vectors = np.array(joblib.load("resources/agnews_BERT.pkl"))
        train = pd.read_csv("data/ag_news/train.csv", header=None)
        test = pd.read_csv("data/ag_news/test.csv", header=None)
        data = pd.concat([train, test])
        target = data[0]
    return target, bert_vectors
