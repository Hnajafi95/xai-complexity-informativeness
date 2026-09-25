#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CALIME setup on the HPC -- venv-based, and deliberately fim/sdv/conda-FREE.
#
# Why no conda / sdv / fim:
#   * conda may not be on PATH on a cluster node -> use a plain venv.
#   * sdv 0.13.0 was YANKED from PyPI and is only used by CALIME's competitors/
#     baselines, NOT its core path -> drop it.
#   * fim (Borgelt PyFIM) is only used by the causal-DISCOVERY step
#     (get_causal_model -> NCDApriori). Our DGP has a KNOWN causal structure,
#     so e3_calime_hpc.py supplies the graph directly and calls
#     CausalDataGeneration (sklearn/scipy only) -> fim is never imported.
#
# This installs into a throwaway venv so it does NOT touch your existing
# (SLALOM/torch) environment, which uses a newer numpy and would break under
# CALIME's pinned numpy 1.23.5.
#
# Usage (on the machine that will run CALIME):
#     bash setup_calime.sh
# then:
#     cp e3_calime_hpc.py ~/CALIME/ && cd ~/CALIME
#     source ~/calime_venv/bin/activate && python e3_calime_hpc.py
# ---------------------------------------------------------------------------
set -uo pipefail

VENV="$HOME/calime_venv"

echo "==> [1/4] creating venv at $VENV"
python3 --version
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo "==> [2/4] cloning CALIME into ~/CALIME"
cd ~ && rm -rf CALIME && git clone https://github.com/marti5ini/CALIME.git

echo "==> [3/4] installing ONLY the core deps (fim/sdv/sklearn-stub/pyod not needed)"
pip install \
    numpy==1.23.5 scipy==1.10.1 networkx==2.8.8 pandas==1.5.3 \
    scikit-learn==1.3.0 "matplotlib>=3.3,<3.8" lime==0.2.0.1

echo "==> [4/4] import smoke test (confirms the fim-free path imports cleanly)"
cd ~/CALIME
python - <<'PY'
# Import ONLY the discovery-free pieces (these do NOT import fim or sdv):
from src.gencda.generative_model import CausalDataGeneration
from src.calime_explainer import CALimeExplainer
from src.lime_explainer import LimeExplainer
import numpy, sklearn, networkx, scipy, pandas
from importlib.metadata import version
print("IMPORTS OK | numpy", numpy.__version__, "| sklearn", sklearn.__version__,
      "| networkx", networkx.__version__, "| lime", version("lime"),
      "| scipy", scipy.__version__, "| pandas", pandas.__version__)
PY
echo "==> if you see 'IMPORTS OK' above, you're set."
echo "==> Next:  cp e3_calime_hpc.py ~/CALIME/ && cd ~/CALIME && source $VENV/bin/activate && python e3_calime_hpc.py"
