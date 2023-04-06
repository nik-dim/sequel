from .agem import AGEM
from .der import DER
from .er import ER
from .ewc import EWC
from .jax_base_algo import JaxBaseAlgorithm, Naive
from .joint import JointTraining
from .lfl import LFL
from .mas import MAS
from .mcsgd import MCSGD
from .si import SI

# stablesgd is added as another algorithm so that it is easily identified and differentiated by the logging services.
# For the moment, it is up to the user to set the correct hyperparams.
ALGOS = dict(
    agem=AGEM,
    base=JaxBaseAlgorithm,
    der=DER,
    er=ER,
    ewc=EWC,
    joint=JointTraining,
    lfl=LFL,
    mas=MAS,
    mcsgd=MCSGD,
    naive=Naive,
    stablesgd=Naive,
    si=SI,
)
