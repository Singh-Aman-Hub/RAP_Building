# Terms — Cost per token

Folder: `21_llm_inference_engineering/16_cost_per_token` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Cost per million tokens](#cost-per-million-tokens) · [cost_per_mtok](#cost_per_mtok) · [request_cost](#request_cost) · [Utilisation](#utilisation) · [Input vs output token cost](#input-vs-output-token-cost) · [Prefix-cache saving](#prefix-cache-saving) · [Roofline throughput estimate](#roofline-throughput-estimate) · [Break-even volume](#break-even-volume) · [Blended API price](#blended-api-price) · [GPU_PRICE_PER_HOUR / API_PRICE_PER_MTOK](#gpu_price_per_hour--api_price_per_mtok)

---

## Cost per million tokens  `(concept)`
**1. Definition:** $/1M tokens = ($ per GPU-hour) / (tokens/s x 3600 x utilisation) x 1e6.
**2. In plain words:** A unit conversion from rental price and speed to price per token.
**3. Example to quote:** > "At 1000 tok/s and $3.6 per hour, a million tokens cost exactly $1."
**Seen in:** `main.py` section S2

## cost_per_mtok  `(function)`
**1. Definition:** Function implementing the formula above, with a utilisation divisor.
**2. In plain words:** One line of arithmetic that the whole step depends on.
**3. Example to quote:** > "The assert checks it against a hand-calculable case."
**Seen in:** `main.py`

## request_cost  `(function)`
**1. Definition:** Cost of one request: uncached input x input price + cached input x cached price + output x output price, all per 1M tokens.
**2. In plain words:** Prices each token type separately.
**3. Example to quote:** > "A request with a cache hit pays the cached rate for those tokens."
**Seen in:** `main.py`

## Utilisation  `(concept)`
**1. Definition:** Fraction of paid time the GPU does useful work; effective $/token divides by it.
**2. In plain words:** Idle hours still cost money.
**3. Example to quote:** > "At 20% utilisation each token costs five times as much."
**Seen in:** `main.py` section S5

## Input vs output token cost  `(concept)`
**1. Definition:** Input (prefill) is compute-bound and fast per token; output (decode) is memory-bound and slow per token, so output costs more per token.
**2. In plain words:** Why API output prices are several times input prices.
**3. Example to quote:** > "Our measured decode was about 35 times slower per token than prefill on this CPU."
**Seen in:** `main.py` section S3

## Prefix-cache saving  `(concept)`
**1. Definition:** A prefix-cache hit skips prefill for the cached tokens, reducing GPU time and (for APIs) billing at a cached-input discount.
**2. In plain words:** Repeated prompt parts become cheap.
**3. Example to quote:** > "An 83% hit rate cut the per-request cost by about 40% in the example."
**Seen in:** `main.py` section S5

## Roofline throughput estimate  `(concept)`
**1. Definition:** Decode step time = max(memory time, compute time); tokens/s = batch / step time.
**2. In plain words:** A paper model of GPU speed used because no GPU is available here.
**3. Example to quote:** > "The 8B-on-H100 numbers are an estimate, not a benchmark."
**Seen in:** `main.py` section S4

## Break-even volume  `(concept)`
**1. Definition:** Monthly tokens above which fixed GPU cost beats the API: (GPUs x $/h x 730) / blended price per 1M.
**2. In plain words:** Buy versus rent threshold.
**3. Example to quote:** > "Below the break-even the API is cheaper because you do not pay for idle hours."
**Seen in:** `main.py` section S6

## Blended API price  `(concept)`
**1. Definition:** share_in x input price + share_out x output price for an assumed token mix.
**2. In plain words:** One price for a typical workload.
**3. Example to quote:** > "A RAG workload is mostly input, so the blended price is close to the input price."
**Seen in:** `main.py` section S6

## GPU_PRICE_PER_HOUR / API_PRICE_PER_MTOK  `(syntax)`
**1. Definition:** Two editable dictionaries of example prices.
**2. In plain words:** The assumptions, kept apart from the logic.
**3. Example to quote:** > "These are example prices; check current pricing."
**Seen in:** `main.py` section S1
