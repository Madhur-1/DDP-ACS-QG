import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import benepar

benepar.download("benepar_en3")
