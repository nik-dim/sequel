from .agem import AGEM
from .der import DER
from .er import ER
from .ewc import EWC
from .icarl import Icarl
from .joint import JointTraining
from .kcl import KCL
from .lamaml import LaMAML
from .lfl import LFL
from .mas import MAS
from .mcsgd import MCSGD
from .pytorch_base_algo import Naive, PytorchBaseAlgorithm
from .si import SI

# stablesgd is added as another algorithm so that it is easily identified and differentiated by the logging services.
# For the moment, it is up to the user to set the correct hyperparams.

ALGOS = dict(
    agem=AGEM,
    base=PytorchBaseAlgorithm,
    der=DER,
    er=ER,
    ewc=EWC,
    icarl=Icarl,
    joint=JointTraining,
    kcl=KCL,
    lamaml=LaMAML,
    lfl=LFL,
    mas=MAS,
    mcsgd=MCSGD,
    naive=Naive,
    stablesgd=Naive,
    si=SI,
)
