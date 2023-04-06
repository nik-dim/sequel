import logging

import coloredlogs


def install_logging(level=logging.INFO):
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES
    level_styles["info"] = {"color": "yellow"}

    coloredlogs.install(
        level=level,
        fmt="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
        level_styles=level_styles,
    )

    # hide the gazillion cryptic debug messages coming from PIL
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.category").disabled = True
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("jax.jit").setLevel(logging.WARNING)
    logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
    logging.getLogger("jax.dispatch").setLevel(logging.WARNING)
    logging.getLogger("dispatch").setLevel(logging.WARNING)
    # plt.set_loglevel("WARNING")
