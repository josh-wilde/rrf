import numpy as np
import torch

from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit


class NeuralNetThreshClassifier(NeuralNetClassifier):

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.NLLLoss,
            train_split=CVSplit(5, stratified=True),
            classes=None,
            fail_threshold: float = 0.5,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )
        self.classes = classes
        self.fail_threshold = fail_threshold

    def predict(self, X):
        if len(self.classes_) == 2:
            predicate = self.predict_proba(X)[:, 1] > self.fail_threshold
            return np.where(predicate, self.classes_[1], self.classes_[0])
        else:
            return super().predict(X)
