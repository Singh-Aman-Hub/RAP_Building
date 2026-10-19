"""
GOAL:   Hands-on with the four everyday tools for scalability + reliability:
        (1) caching, (2) timeouts + retries with backoff, (3) bounded queue / load shedding,
        (4) capacity maths (Little's Law).  Then see real numbers in load_test.py.

RUN:    cd 10_ml_system_architecture/08_scalability_and_reliability && python main.py

EXPECTED OUTPUT (shape): cache hit rate, a flaky call succeeding on attempt 3, a timeout caught,
        accepted vs rejected requests from an overloaded queue, a capacity table.

NEEDS:  standard library only.
"""

import queue                                            # built-in: thread-safe FIFO queues
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import lru_cache                         # built-in: memoize a function (remember results)

# ----------------------------- 1. caching ---------------------------------------
CALLS = {"model": 0}


@lru_cache(maxsize=1024)                                # decorator: same arguments -> return the saved result
def cached_predict(text: str) -> str:
    CALLS["model"] += 1                                 # counts REAL model executions
    time.sleep(0.01)                                    # pretend inference takes 10 ms
    return "positive" if "good" in text else "negative"


def demo_cache():
    traffic = ["good film", "bad film", "good film", "good film", "bad film", "good film"] * 50   # repeats are common
    start = time.perf_counter()
    for t in traffic:
        cached_predict(t)
    ms = (time.perf_counter() - start) * 1000
    info = cached_predict.cache_info()                  # namedtuple: hits, misses, maxsize, currsize
    print(f"1. cache: {len(traffic)} requests -> {CALLS['model']} model runs, hits={info.hits}, misses={info.misses}, "
          f"total {ms:.0f} ms (without cache ~{len(traffic) * 10} ms)")
    print("   Limits: only for repeated INPUTS, stale if the model changes (include model_version in the key!), uses memory.")


# ----------------------------- 2. timeout + retry -------------------------------
attempts = {"n": 0}


def flaky_service() -> str:
    """Pretend downstream call (DB, another API) failing the first 2 times -- typical transient error."""
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError(f"attempt {attempts['n']}: connection reset")
    return "ok"


def with_retries(fn, retries: int = 4, base_delay: float = 0.05):
    """Retry with EXPONENTIAL BACKOFF: wait 0.05s, 0.1s, 0.2s ... so a struggling service is not hammered."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except ConnectionError as err:                  # retry ONLY errors that can be transient
            print(f"   {err} -> retrying" if attempt < retries else f"   {err} -> giving up")
            if attempt == retries:
                raise                                   # re-raise the original error after the last attempt
            time.sleep(base_delay * 2 ** (attempt - 1))  # ** is the power operator


def call_with_timeout(fn, seconds: float):
    """Never wait forever: a hung dependency would pile up requests until the whole service falls over."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except FutureTimeout:
            return "TIMEOUT -> return fallback answer / 504 to the client"


def demo_retry_timeout():
    print("2. retry with backoff:")
    print("   result:", with_retries(flaky_service))
    print("   timeout:", call_with_timeout(lambda: time.sleep(0.5) or "slow result", seconds=0.1))
    print("   Rule: retry only idempotent operations (safe to repeat); always add timeouts AND a max retry count.")


# ----------------------------- 3. bounded queue / load shedding ----------------
def demo_queue():
    work: queue.Queue = queue.Queue(maxsize=5)          # BOUNDED: at most 5 waiting jobs
    done = []

    def worker():
        while True:
            item = work.get()                           # blocks until a job is available
            if item is None:                            # sentinel value = "stop"
                return
            time.sleep(0.05)                            # processing time per job
            done.append(item)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    accepted = rejected = 0
    for i in range(40):                                 # a burst of 40 requests arrives at once
        try:
            work.put_nowait(i)                          # raises queue.Full instead of waiting
            accepted += 1
        except queue.Full:
            rejected += 1                               # LOAD SHEDDING: answer 429/503 fast instead of timing out slowly
        time.sleep(0.01)                                # requests arrive every 10 ms (100/s) but 2 workers only finish 40/s
    for _ in threads:
        work.put(None)
    for t in threads:
        t.join()
    print(f"3. burst of 40 into queue(max 5) with 2 workers: accepted={accepted}, rejected (HTTP 429/503)={rejected}, processed={len(done)}")
    print("   An UNBOUNDED queue never rejects but latency grows without limit and memory runs out -- bounded + backpressure is safer.")


# ----------------------------- 4. capacity maths --------------------------------
def demo_capacity():
    print("4. capacity planning (Little's Law: concurrent requests = arrival rate x latency)")
    print("   latency per request | workers | max throughput | workers needed for 200 req/s")
    for latency_s, workers in [(0.05, 1), (0.05, 4), (0.2, 4), (1.0, 4)]:
        per_worker = 1 / latency_s                      # a sync worker does one request at a time
        needed = 200 * latency_s                        # = arrival rate x latency
        print(f"   {latency_s*1000:>7.0f} ms          | {workers:^7} | {per_worker*workers:>8.0f} req/s  | {needed:>5.0f}")
    print("   Slow model => you need MORE workers/replicas, OR faster inference (smaller model, quantization, batching, GPU).")


if __name__ == "__main__":
    demo_cache()
    demo_retry_timeout()
    demo_queue()
    demo_capacity()
    # TRY IT: change maxsize=5 to 50 and 1 -- how do accepted/rejected change?
    # TRY IT: make flaky_service fail 5 times and see "giving up" with the re-raised ConnectionError traceback.
