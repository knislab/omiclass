# generic-classifier/sampling/svmsmote.py
from imblearn.over_sampling import SVMSMOTE

class SafeSVMSMOTE(SVMSMOTE):
    def fit_resample(self, X, y):
        try:
            return super().fit_resample(X, y)
        except ValueError as e:
            msg = str(e)
            benign_substrings = (
                "Expected n_neighbors",
                "not enough neighbors",
                "No samples will be generated with the provided ratio settings",
                "All support vectors are considered as noise",
            )
            if any(s in msg for s in benign_substrings):
                return X, y
            # anything else is a real error
            raise

