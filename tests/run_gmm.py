"""
=================================
Gaussian Mixture Model Ellipsoids
=================================

Plot the confidence ellipsoids of a mixture of two Gaussians
obtained with Expectation Maximisation (``GaussianMixture`` class) and
Variational Inference (``BayesianGaussianMixture`` class models with
a Dirichlet process prior).

Both models have access to five components with which to fit the data. Note
that the Expectation Maximisation model will necessarily use all five
components while the Variational Inference model will effectively only use as
many as are needed for a good fit. Here we can see that the Expectation
Maximisation model splits some components arbitrarily, because it is trying to
fit too many components, while the Dirichlet Process model adapts it number of
state automatically.

This example doesn't show it, as we're in a low-dimensional space, but
another advantage of the Dirichlet process model is that it can fit
full covariance matrices effectively even when there are less examples
per cluster than there are dimensions in the data, due to
regularization properties of the inference algorithm.
"""

import sys
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

# from sklearn import mixture
sys.path.append("../")
from gaussian_mixture import GaussianMixture
from bayesian_mixture import BayesianGaussianMixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, index, title, covar_type='diag'):
    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        if covar_type == 'diag':
            covar = np.diag(covar[:2])
        elif covar_type == 'tied':
            covar = covariances
        elif covar_type == 'spherical':
            covar = np.eye(means.shape[1]) * covar
        else:
            covar = covar[:2, :2]
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(np.concatenate(Y_, axis=0) == i):
            continue

        for idx in range(len(X)):
            plt.scatter(X[idx][Y_[idx] == i, 0], X[idx][Y_[idx] == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of docs
n_docs = 2

n_components = 3
# weights = [[.3, .4, .3], [.5, .2, .3]]
weights = np.random.dirichlet((1, n_docs, n_components), n_docs)


# Number of samples per component
n_samples = 1000

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
word_cls = [np.dot(np.random.randn(n_samples, 2), C), .7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
                .4 * np.random.randn(n_samples, 2) + .6 * np.random.randn(n_samples, 2)]

# Generate a corpus
corpus = []
for each in range(n_docs):
    n0 = int(weights[each][0] * n_samples)
    n1 = int(weights[each][1] * n_samples)
    n2 = int(weights[each][2] * n_samples)
    idx0 = np.random.choice(range(n_samples), n0, replace=False)
    idx1 = np.random.choice(range(n_samples), n1, replace=False)
    idx2 = np.random.choice(range(n_samples), n2, replace=False)
    X = np.r_[word_cls[0][idx0], word_cls[1][idx1], word_cls[2][idx2]]
    corpus.append(X)
# import pdb;pdb.set_trace()
# Fit a Gaussian mixture with EM using five components
covar_type = 'diag'
gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, tol=1e-6, max_iter=1000, n_init=1, verbose=1).fit(corpus)
gmm2 = GaussianMixture(n_components=n_components, covariance_type=covar_type, tol=1e-6, max_iter=1000, n_init=10, verbose=1).fit(corpus)

print "gold weights:"
print weights


# print "means: %s" % gmm.means_
# print "covariances: %s" % gmm.covariances_
print "system output weights:"
print gmm.weights_
print "system output weights2:"
print gmm2.weights_
# plot_results(corpus, gmm.predict(corpus, range(len(corpus))), gmm.means_, gmm.covariances_, 0,
#              'Gaussian Mixture', covar_type)

# plt.show()
