"""External classifiers on frozen arrays; no encoder or Omiclass training imports."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import copy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def embeddings(values, width=256):
    array = np.array(values, dtype=np.float32, copy=True)
    if array.ndim != 2 or array.shape[1] != width or not len(array) or not np.isfinite(array).all():
        raise ValueError('Expected nonempty finite (n_samples, 256) frozen embeddings')
    return array


def training_labels(y, n_samples, n_classes):
    raw = np.asarray(y)
    labels = np.asarray(y, dtype=np.int64)
    if labels.shape != (n_samples,) or not np.array_equal(raw, labels):
        raise ValueError('Training labels must be aligned integer class indices')
    if n_classes < 2 or not np.array_equal(np.unique(labels), np.arange(n_classes)):
        raise ValueError('Training must contain every canonical class, indexed from zero')
    return labels


def fit_svm(X_train, y_train, seed):
    """Fixed project configuration; libsvm calibration sees only training samples."""
    X = embeddings(X_train)
    y = training_labels(y_train, len(X), len(np.unique(y_train)))
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1.0, kernel='rbf', gamma='scale', class_weight=None,
                    probability=True, decision_function_shape='ovr', random_state=seed,
                    shrinking=True, tol=1e-3, cache_size=200, max_iter=-1,
                    break_ties=False)),
    ])
    model.fit(X, y)
    return model


@dataclass(frozen=True)
class MLPConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    validation_fraction: float = 0.2
    min_delta: float = 0.0
    threads: int = 1
    resampling: str = 'none'
    smote_k_neighbors: int = 5

    def validate(self):
        if self.resampling not in {'none', 'smote'} or self.smote_k_neighbors < 1:
            raise ValueError('Invalid SMOTE configuration')
        if (not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not 0 < self.validation_fraction < 1
                or not np.isfinite(self.min_delta) or self.min_delta < 0
                or min(self.batch_size, self.max_epochs, self.patience, self.threads) < 1):
            raise ValueError('Invalid MLP training configuration')

    def metadata(self):
        return dict(asdict(self), architecture=[256, 256, 128, 'n_classes'],
                    activation='SELU', output='logits', optimizer='Adam',
                    adam_betas=[0.9, 0.999], adam_eps=1e-8, weight_decay=0.0,
                    loss='unweighted_cross_entropy', dropout=0.0, layer_norm=False,
                    scaler=None, scheduler=None, class_weight=None,
                    initialization='PyTorch Linear defaults', device='cpu', dtype='float32',
                    early_stopping='minimum inner validation cross-entropy; restore best epoch',
                    refit_on_all_training=False)


def make_mlp(n_classes, width=256):
    # Lazy import: SVM and input validation do not need PyTorch or JAX.
    from torch import nn
    if n_classes < 2:
        raise ValueError('At least two classes are required')
    return nn.Sequential(nn.Linear(width, 256), nn.SELU(), nn.Linear(256, 128),
                         nn.SELU(), nn.Linear(128, n_classes))


def inner_partition(train_ids, y_train, config, seed):
    config.validate()
    sample_ids = list(train_ids)
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Training sample IDs must be nonempty and unique')
    labels = training_labels(y_train, len(sample_ids), len(np.unique(y_train)))
    if np.bincount(labels).min() < 2:
        raise ValueError('Stratified MLP validation requires at least two training samples per class')
    try:
        train, validation = train_test_split(np.arange(len(labels)),
            test_size=config.validation_fraction, stratify=labels, random_state=seed)
    except ValueError as exc:
        raise ValueError(f'Cannot form training-only stratified MLP validation: {exc}') from exc
    expected = set(labels)
    if set(labels[train]) != expected or set(labels[validation]) != expected:
        raise ValueError('Every class must occur in both inner partitions; adjust validation fraction')
    return train, validation


def smote_settings(y, k_neighbors):
    try:
        import imblearn
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError('SMOTE requires imbalanced-learn in the training Python environment') from exc
    counts = np.bincount(y)
    minority = counts[counts < counts.max()]
    if len(minority) and minority.min() < 2:
        raise ValueError('SMOTE needs at least two inner-training samples in every class needing oversampling')
    effective = min(k_neighbors, int(minority.min()) - 1) if len(minority) else None
    return dict(method='smote', requested_k_neighbors=k_neighbors,
                effective_k_neighbors=effective, class_counts_before=counts.tolist(),
                imbalanced_learn_version=imblearn.__version__)


@contextmanager
def torch_reproducibility(seed, threads):
    from omiclass.utils.reproducibility import random_state_scope
    with random_state_scope(seed, threads=threads, deterministic=True):
        yield


def fit_mlp(X_train, y_train, train_ids, n_classes, config, inner_seed, input_width=256):
    """The API deliberately accepts only outer-training data, never a holdout."""
    import torch
    X = embeddings(X_train, input_width)
    y = training_labels(y_train, len(X), n_classes)
    if len(train_ids) != len(X):
        raise ValueError('Training sample IDs and embeddings must align')
    train, validation = inner_partition(train_ids, y, config, inner_seed)
    original_train = train.copy()
    resampling = dict(method='none')
    if config.resampling == 'smote':
        resampling = smote_settings(y[train], config.smote_k_neighbors)
        effective = resampling['effective_k_neighbors']
        if effective is not None:
            from imblearn.over_sampling import SMOTE
            augmented_x, augmented_y = SMOTE(random_state=inner_seed, k_neighbors=effective,
                                             sampling_strategy='auto').fit_resample(X[train], y[train])
            synthetic_x, synthetic_y = augmented_x[len(train):], augmented_y[len(train):]
            train = np.concatenate([train, np.arange(len(X), len(X) + len(synthetic_y))])
            X = np.concatenate([X, synthetic_x])
            y = np.concatenate([y, synthetic_y])
        resampling.update(seed=inner_seed, class_counts_after=np.bincount(y[train]).tolist(),
                          synthetic_sample_count=len(train) - len(original_train),
                          scope='inner training only; raw frozen embedding coordinates')
    with torch_reproducibility(inner_seed, config.threads):
        model = make_mlp(n_classes, input_width)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        criterion = torch.nn.CrossEntropyLoss()
        features, labels = torch.from_numpy(X), torch.from_numpy(y)
        generator = torch.Generator().manual_seed(inner_seed)
        best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
        history = []
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            order = train[torch.randperm(len(train), generator=generator).numpy()]
            loss_sum = 0.0
            for start in range(0, len(order), config.batch_size):
                batch = order[start:start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features[batch]), labels[batch])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite MLP training loss')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch)
            model.eval()
            with torch.no_grad():
                validation_loss = float(criterion(model(features[validation]), labels[validation]))
            if not np.isfinite(validation_loss):
                raise ValueError('Nonfinite MLP inner validation loss')
            history.append(dict(epoch=epoch, train_loss=loss_sum / len(train),
                                validation_loss=validation_loss))
            if validation_loss < best_loss - config.min_delta:
                best_loss, best_epoch = validation_loss, epoch
                best_state, stale = copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= config.patience:
                break
        model.load_state_dict(best_state)
        model.eval()
    metadata = dict(best_epoch=best_epoch, best_validation_loss=best_loss,
                    epochs_trained=len(history), inner_seed=inner_seed,
                    inner_train_ids=[train_ids[i] for i in original_train],
                    resampling=resampling,
                    inner_validation_ids=[train_ids[i] for i in validation],
                    trainable_parameters=sum(p.numel() for p in model.parameters()),
                    history=history)
    return model, metadata


def predict_mlp(model, X, threads=1, input_width=256):
    import torch
    features = torch.from_numpy(embeddings(X, input_width))
    with torch_reproducibility(0, threads), torch.no_grad():
        model.eval()
        logits = torch.cat([model(batch) for batch in features.split(1024)])
        probabilities = torch.softmax(logits, dim=1).numpy()
    return probabilities.argmax(axis=1), probabilities, logits.numpy()
