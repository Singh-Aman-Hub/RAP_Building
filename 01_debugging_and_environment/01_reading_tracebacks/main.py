"""
GOAL:  Learn the anatomy of a traceback by producing one on purpose and taking it apart.

RUN:   cd 01_debugging_and_environment/01_reading_tracebacks && python main.py
       Then run the crashing version:  python broken_traceback.py   (EXPECTED to crash)

EXPECTED OUTPUT: a printed traceback plus a breakdown of its parts.

NEEDS: nothing (standard library only), no internet.
"""

import traceback  # standard library: lets us capture/print tracebacks as text


def load_config():
    return {"model": "gpt2"}            # note: there is NO "max_tokens" key


def get_limit(config):
    return config["max_tokens"]         # <-- the real bug: KeyError happens HERE


def run():
    config = load_config()
    return get_limit(config)            # <-- call site (middle frame)


try:
    run()                               # <-- outermost frame (module level)
except KeyError as err:
    # traceback.format_exc() returns the same text Python prints when a program crashes.
    text = traceback.format_exc()
    print(text)

    # ---- anatomy: read BOTTOM -> TOP ----------------------------------------
    last_line = text.strip().splitlines()[-1]
    print("1) LAST LINE  = error type + message  :", last_line)
    print("2) The frame just above it            : where the exception was RAISED")
    print("3) Frames above that                  : who called whom (oldest call at the top)")
    print("4) err.args                           :", err.args)

    # tb_frame / traceback.extract_tb give structured access to each frame
    for frame in traceback.extract_tb(err.__traceback__):
        print(f"   frame: file={frame.filename.split('/')[-1]} line={frame.lineno} "
              f"function={frame.name} code={frame.line!r}")

# TRY IT: change config["max_tokens"] to config.get("max_tokens", 100) and re-run: no error, no except branch.
# BREAK IT: rename get_limit's parameter but not its use inside -> NameError. Read the line number it reports.
