Extrinsic UQ Algorithms
=======================

Auxiliary Interval Predictor
----------------------------

.. autoclass:: uq360.algorithms.auxiliary_interval_predictor.AuxiliaryIntervalPredictor
   :members:


Blackbox Metamodel Classification
---------------------------------

.. autoclass:: uq360.algorithms.blackbox_metamodel.BlackboxMetamodelClassification
   :members:

Blackbox Metamodel Regression
-----------------------------

.. autoclass:: uq360.algorithms.blackbox_metamodel.BlackboxMetamodelRegression
   :members:

Infinitesimal Jackknife
-----------------------

.. autoclass:: uq360.algorithms.infinitesimal_jackknife.InfinitesimalJackknife
   :members:

Classification Calibration
--------------------------

.. autoclass:: uq360.algorithms.classification_calibration.ClassificationCalibration
   :members:

UCC Recalibration
-----------------

.. autoclass:: uq360.algorithms.ucc_recalibration.UCCRecalibration
   :members:


Structured Data Predictor
-----------------

.. autoclass:: uq360.algorithms.blackbox_metamodel.structured_data_classification.StructuredDataClassificationWrapper
   :members:

Short Text Predictor
-----------------

.. autoclass:: uq360.algorithms.blackbox_metamodel.short_text_classification.ShortTextClassificationWrapper
   :members:

Confidence Predictor
-----------------

.. autoclass:: uq360.algorithms.blackbox_metamodel.confidence_classification.confidenceclassificationwrapper
   :members:

Latent Space Anomaly Detection Scores
---------------------------------------

.. autoclass:: uq360.algorithms.layer_scoring.mahalanobis.MahalanobisScorer
   :members:

.. autoclass:: uq360.algorithms.layer_scoring.knn.KNNScorer
   :members:

.. autoclass:: uq360.algorithms.layer_scoring.aklpe.AKLPEScorer
   :members:

Nearest Neighbors Algorithms for KNN-based anomaly detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: uq360.utils.transformers.nearest_neighbors.exact.ExactNearestNeighbors

.. autoclass:: uq360.utils.transformers.nearest_neighbors.pynndescent.PyNNDNearestNeighbors

.. autoclass:: uq360.utils.transformers.nearest_neighbors.faiss.FAISSNearestNeighbors