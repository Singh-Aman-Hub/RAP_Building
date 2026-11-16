# Terms — Checkpoint formats and loading

Folder: `21_llm_inference_engineering/01_checkpoint_formats_and_loading` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Pickle (and __reduce__)](#pickle-and-__reduce__) · [weights_only (torch.load)](#weights_only-torchload) · [safetensors](#safetensors) · [data_offsets](#data_offsets) · [np.frombuffer](#npfrombuffer) · [mmap (memory-mapped file)](#mmap-memory-mapped-file) · [safe_open (lazy loading)](#safe_open-lazy-loading) · [Peak RSS (ru_maxrss)](#peak-rss-ru_maxrss) · [Page cache (warm vs cold load)](#page-cache-warm-vs-cold-load) · [Sharded checkpoint (index.json, weight_map)](#sharded-checkpoint-indexjson-weight_map) · [Meta device](#meta-device) · [device_map='auto' / low_cpu_mem_usage](#device_mapauto--low_cpu_mem_usage) · [dtype on load](#dtype-on-load) · [Hugging Face cache (blobs / snapshots / refs)](#hugging-face-cache-blobs--snapshots--refs) · [HF_HOME / HF_HUB_CACHE / HF_HUB_OFFLINE](#hf_home--hf_hub_cache--hf_hub_offline)

---

## Pickle (and __reduce__)  `(concept)`
**1. Definition:** Python's object serialisation format. It is a tiny stack-machine program; the REDUCE opcode calls any callable named in the file, and an object's __reduce__ method chooses which callable and arguments are stored.
**2. In plain words:** A pickle is a recipe that Python follows when loading, and the recipe can say 'run this command'.
**3. Example to quote:** > "Loading a pickle is running code, so only unpickle files you made yourself."
```python
class Harmless:
    def __reduce__(self):
        return (print, ("code ran",))
pickle.loads(pickle.dumps(Harmless()))   # prints 'code ran'
```
**Seen in:** `main.py` section 1

## weights_only (torch.load)  `(function)`
**1. Definition:** Argument of torch.load. With weights_only=True (the default since torch 2.6) a restricted unpickler only rebuilds tensors, dicts, lists and a small allow-list of types, and refuses anything else.
**2. In plain words:** A safety filter on the pickle loader that lets numbers through and blocks code.
**3. Example to quote:** > "torch.load is safer by default now, but safetensors removes the question entirely."
```python
torch.load(path, weights_only=True)
```
**Seen in:** `main.py` section 1

## safetensors  `(concept)`
**1. Definition:** A weight file format by Hugging Face: 8-byte little-endian header length N, N bytes of JSON header (name -> dtype, shape, data_offsets, plus optional __metadata__), then one raw byte buffer. No code is executed on load.
**2. In plain words:** A labelled box of numbers: a table of contents followed by the numbers.
**3. Example to quote:** > "safetensors is a JSON table of contents plus raw bytes, so it is safe and mmap-friendly."
```python
n = struct.unpack('<Q', raw[:8])[0]
header = json.loads(raw[8:8+n])
```
**Seen in:** `main.py` section 2

## data_offsets  `(concept)`
**1. Definition:** Per tensor, a [begin, end) byte range relative to the start of the data buffer (which starts at 8 + N). Ranges are contiguous and non-overlapping.
**2. In plain words:** Where in the file each tensor starts and stops.
**3. Example to quote:** > "To find a tensor you only need the header and a slice, no scanning."
```python
arr = np.frombuffer(raw, np.float32, offset=8+n+begin)
```
**Seen in:** `main.py` section 2

## np.frombuffer  `(function)`
**1. Definition:** numpy function that interprets an existing bytes-like buffer as an array without copying (dtype, count, offset arguments).
**2. In plain words:** Looks at raw bytes through a typed window.
**3. Example to quote:** > "np.frombuffer is the whole 'decoder' for safetensors data."
```python
np.frombuffer(raw, dtype=np.float16, offset=off)
```
**Seen in:** `main.py` section 2

## mmap (memory-mapped file)  `(concept)`
**1. Definition:** An OS facility that maps a file into the process address space; pages are read from disk lazily on first access and can be shared and evicted by the OS page cache.
**2. In plain words:** The file pretends to be memory; only the parts you touch are read.
**3. Example to quote:** > "Lazy loading with mmap means opening a 70 GB file costs almost no RAM until you read it."
```python
mm = np.memmap(path, dtype=np.uint8, mode='r')
```
**Seen in:** `main.py` section 3

## safe_open (lazy loading)  `(function)`
**1. Definition:** safetensors API that opens a file (mmap), exposes keys() and get_tensor(name) so you can read single tensors or slices.
**2. In plain words:** Open the box, read the contents list, take out only the item you need.
**3. Example to quote:** > "safe_open lets you read one tensor out of many without loading the rest."
```python
with safe_open(p, framework='pt') as f:
    w = f.get_tensor('layer0.w')
```
**Seen in:** `main.py` section 3

## Peak RSS (ru_maxrss)  `(concept)`
**1. Definition:** Resident set size: RAM pages of a process currently in memory. resource.getrusage(...).ru_maxrss is its high-water mark (bytes on macOS, kilobytes on Linux) and never decreases within a process.
**2. In plain words:** The most RAM the process ever used.
**3. Example to quote:** > "I measured each loader in a fresh process because peak RSS only goes up."
```python
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
```
**Seen in:** `main.py` section 3

## Page cache (warm vs cold load)  `(concept)`
**1. Definition:** The OS keeps recently read file pages in RAM. A first read of a file not in the cache (cold) hits the disk; a repeat (warm) is served from RAM and looks much faster.
**2. In plain words:** The computer remembers files it just read.
**3. Example to quote:** > "Load-time benchmarks must say whether the cache was cold or warm."
**Seen in:** `main.py` section 3, WALKTHROUGH

## Sharded checkpoint (index.json, weight_map)  `(concept)`
**1. Definition:** A large model split into model-0000X-of-0000N.safetensors files plus model.safetensors.index.json whose weight_map maps each tensor name to its shard file and whose metadata.total_size gives total bytes.
**2. In plain words:** A multi-volume book with an index saying which volume holds which page.
**3. Example to quote:** > "Shards let a loader hold one file in RAM at a time and let you fetch one tensor from one shard."
```python
idx['weight_map']['model.layers.4.mlp.w']   # -> 'model-00002-of-00003.safetensors'
```
**Seen in:** `main.py` section 4

## Meta device  `(concept)`
**1. Definition:** A PyTorch device whose tensors carry shape and dtype but have no storage. Building a model under torch.device('meta') costs almost no RAM and is used to create the skeleton before weights are loaded.
**2. In plain words:** A blueprint of the model with no bricks yet.
**3. Example to quote:** > "Meta-device init is how huge models are built without first allocating random weights."
```python
with torch.device('meta'):
    m = AutoModelForCausalLM.from_config(cfg)
```
**Seen in:** `main.py` section 5

## device_map='auto' / low_cpu_mem_usage  `(syntax)`
**1. Definition:** from_pretrained options (accelerate-backed) that build the model on meta, then place each loaded tensor straight onto its target device (GPU/CPU/disk) instead of materialising a random-initialised copy first. device_map='auto' also splits layers over available devices.
**2. In plain words:** Load weights directly to where they will live.
**3. Example to quote:** > "low_cpu_mem_usage avoids needing two copies of the model in RAM."
```python
AutoModelForCausalLM.from_pretrained(name, device_map='auto')
```
**Seen in:** `main.py` section 5

## dtype on load  `(syntax)`
**1. Definition:** from_pretrained(..., dtype=torch.bfloat16) converts weights to that dtype while loading (older releases call the argument torch_dtype). The default is fp32 unless the checkpoint config says otherwise or you pass 'auto'.
**2. In plain words:** Pick the number format at load time.
**3. Example to quote:** > "Loading in bf16 halves RAM compared with fp32 at the cost of precision."
```python
from_pretrained(name, dtype=torch.bfloat16)
```
**Seen in:** `main.py` section 5

## Hugging Face cache (blobs / snapshots / refs)  `(concept)`
**1. Definition:** ~/.cache/huggingface/hub/models--org--name/ holds blobs/ (content-addressed files), snapshots/<commit>/ (folders of symlinks into blobs, one per revision) and refs/<branch> (text file with the commit hash).
**2. In plain words:** Files are stored once by content; each revision is a folder of shortcuts.
**3. Example to quote:** > "Because snapshots are symlinks, two revisions share unchanged files on disk."
```python
os.readlink(snapshot/'model.safetensors')   # ../../blobs/<sha256>
```
**Seen in:** `main.py` section 6

## HF_HOME / HF_HUB_CACHE / HF_HUB_OFFLINE  `(syntax)`
**1. Definition:** Environment variables: HF_HOME relocates all Hugging Face data, HF_HUB_CACHE the hub cache folder, HF_HUB_OFFLINE=1 forbids network calls so only cached files are used.
**2. In plain words:** Switches that decide where files live and whether the internet may be used.
**3. Example to quote:** > "I set HF_HUB_OFFLINE=1 in CI so a missing file fails fast instead of downloading."
```python
HF_HUB_OFFLINE=1 python main.py
```
**Seen in:** `main.py` section 6
