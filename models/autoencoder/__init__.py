"""
Autoencoder package.

New (unknown-anomaly branch):  PatchConvAE, AnomalyScorer  -- trained on NORMAL seabed only.
Legacy (backward compatible):  SonarConvAutoencoder, SonarAnomalyDetector -- the original module,
kept so app.py / scripts / tests importing `models.autoencoder` keep working unchanged.
"""

from models.autoencoder.legacy import SonarAnomalyDetector, SonarConvAutoencoder  # noqa: F401
from models.autoencoder.conv_ae import PatchConvAE  # noqa: F401
from models.autoencoder.anomaly import AnomalyScorer, error_to_score  # noqa: F401
