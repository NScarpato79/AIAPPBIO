"""VancoTwin: prototipo di Digital Twin farmacocinetico della vancomicina.

Pipeline: dati sintetici -> QC/pseudonimizzazione -> modello ODE -> calibrazione
bayesiana (PyMC) -> assimilazione real-time (EnKF) -> scenari terapeutici con
bande di incertezza -> validazione e analisi di sensitivita'.
"""

__version__ = "1.0.0"
