from pathlib import Path
import sys


sys.path.append(Path().parent.parent.absolute().as_posix())
print(sys.path)
