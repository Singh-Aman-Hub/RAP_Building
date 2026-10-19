"""
GOAL:  Replace scattered print() debugging with the logging module (levels, format, file output, exceptions).

RUN:   cd 01_debugging_and_environment/08_logging_vs_print && python main.py
       quieter/louder:  LOG_LEVEL=DEBUG python main.py     LOG_LEVEL=WARNING python main.py

EXPECTED OUTPUT: timestamped log lines at different levels, and a log file in _scratch/.

NEEDS: nothing (standard library only).
"""

import logging
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "_scratch").mkdir(exist_ok=True)

# basicConfig sets up the root logger ONCE for the whole program.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),           # messages below this level are dropped
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s",   # what each line looks like
    handlers=[logging.StreamHandler(),                    # to the terminal
              logging.FileHandler(HERE / "_scratch" / "app.log", mode="w")],  # and to a file
)
log = logging.getLogger("inference")   # a named logger: shows which part of the app spoke


def predict(text):
    log.debug("predict() called with %r", text)           # %-style args: formatting only if the level is enabled
    if not text:
        log.warning("empty input, returning default")
        return "unknown"
    return "positive" if "good" in text else "negative"


# Levels from least to most severe: DEBUG < INFO < WARNING < ERROR < CRITICAL
log.info("service starting")
print("print() output has no time, level or source ->", predict("good movie"))
predict("")

# logger.exception() logs an ERROR *with the full traceback* -- use it inside `except`.
try:
    1 / 0
except ZeroDivisionError:
    log.exception("prediction failed")

log.info("done; see %s", HERE / "_scratch" / "app.log")

# WHEN print() IS FINE: quick throwaway checks while exploring (print(type(x), x.shape)).
# WHEN LOGGING WINS: anything that runs in a server/container, needs levels, timestamps, files, or must stay in code.
# TRY IT: run with LOG_LEVEL=DEBUG and see the extra line from predict().
# BREAK IT: call logging.basicConfig(level="INFO") a SECOND time later in the file: it is silently ignored (it only configures once).
