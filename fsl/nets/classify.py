#!/usr/bin/env python
#
# classify.py - Train a classifier on netmats to diffentiate groups.

# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import warnings

import numpy                          as     np
from   sklearn.pipeline               import Pipeline
from   sklearn.preprocessing          import StandardScaler
from   sklearn.discriminant_analysis  import QuadraticDiscriminantAnalysis
from   sklearn.model_selection        import LeaveOneOut


def classify(netmats, groups, classifier=None):
    """Train a machine-learning classifier to differentiate groups based on
    netmat edge strengths.

    netmats:    (runs, edges) array containing per-subject netmats.
    groups:     Number of subjects in each group
    classifier: scikit-learn classifier object. The default is to use a
                QuadraticDiscriminantAnalysis classifier.
    """

    if classifier is None:
        classifier = QuadraticDiscriminantAnalysis(store_covariance=True)

    labels = np.zeros(netmats.shape[0], dtype=int)

    for i, group in enumerate(groups):
        start             = int(np.sum(groups[:i]))
        end               = start + group
        labels[start:end] = i

    pipe = Pipeline([('preproc', StandardScaler()),
                     ('fit',     classifier)])

    loo = LeaveOneOut()

    predictions = np.zeros(labels.shape, dtype=int)
    for fold, (train, test) in enumerate(loo.split(netmats)):

        test_label   =  labels[test[0]]
        train_labels = [labels[i] for i in train]

        # Suppress this warning:
        #   sklearn/discriminant_analysis.py:926: UserWarning: Variables are collinear
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipe.fit(netmats[train], train_labels)
            result = pipe.predict(netmats[test])[0]

        predictions[fold] = result

        print(f'Training fold {fold+1:2d} label: {test_label}, prediction: {result}')

    correct  = (labels == predictions).sum()
    accuracy = correct / len(labels)

    print(f'Accuracy during training: {100 * accuracy:0.2f}%')

    predictions = pipe.predict(netmats)
    correct     = (labels == predictions).sum()
    accuracy    = correct / len(labels)
    print(f'Accuracy on input data:   {100 * accuracy:0.2f}%')
