# Terms — File Paths and the Working Directory

Folder: `01_debugging_and_environment/04_paths_and_cwd` · Files covered: `main.py`, `broken_paths.py`, `data.txt`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Working directory (cwd)](#working-directory-cwd) · [os.getcwd()](#osgetcwd) · [__file__](#__file__) · [Path.resolve()](#pathresolve) · [Path.parent](#pathparent) · [Relative path](#relative-path) · [Absolute path](#absolute-path) · [Anchored path](#anchored-path) · [Path.exists()](#pathexists) · [Path.read_text()](#pathread_text) · [open() with a relative path](#open-with-a-relative-path) · [FileNotFoundError](#filenotfounderror) · [Case sensitivity](#case-sensitivity) · [os.listdir](#oslistdir) · [Works on my machine](#works-on-my-machine) · [Docker WORKDIR](#docker-workdir) · [Data directory via env var](#data-directory-via-env-var)

---

## Working directory (cwd)  `(concept)`
**1. Definition:** The current working directory is the folder a process considers "here"; relative paths are resolved from it, and it is set by where you launched the command, not by where the script file lives.
**2. In plain words:** If you run `python script.py` from your home folder, the working directory is your home folder, even though `script.py` sits somewhere else. That is why `open("data.txt")` works in one terminal location and fails in another.
**3. Example to quote:**
> "My script worked from its own folder but crashed from the repo root, because the working directory changed and `data.txt` wasn't there."
```python
import os
print(os.getcwd())
```
**Seen in:** `main.py`; `WALKTHROUGH.md`, goal

## os.getcwd()  `(standard library)`
**1. Definition:** `os.getcwd()` returns the current working directory as a string.
**2. In plain words:** It's the first thing to print when a file isn't found: it shows where Python believes it is. Compare it with where the file really is.
**3. Example to quote:**
> "I printed `os.getcwd()` and saw the repo root, not the script folder, which explained the `FileNotFoundError`."
```python
print(os.getcwd())
```
**Seen in:** `main.py`, first `print`

## __file__  `(Python built-in)`
**1. Definition:** `__file__` is a variable set in each running module holding the path of its source file.
**2. In plain words:** It lets a script find its own location, no matter where it was launched from. Combined with `Path(...).resolve().parent` it gives the script's folder.
**3. Example to quote:**
> "`Path(__file__).resolve()` printed the full path of my script, regardless of where I ran it from."
```python
from pathlib import Path
print(Path(__file__).resolve())
```
**Seen in:** `main.py`

## Path.resolve()  `(standard library)`
**1. Definition:** `Path.resolve()` returns the absolute, normalised form of a path, following symbolic links.
**2. In plain words:** It turns `data.txt` into `/full/path/to/data.txt` based on the current working directory. Printing it shows where Python is really looking.
**3. Example to quote:**
> "`Path('data.txt').resolve()` showed me the exact absolute path Python tried to open, which wasn't the one I assumed."
```python
print(Path("data.txt").resolve())
```
**Seen in:** `main.py`, "absolute form of relative path"

## Path.parent  `(standard library)`
**1. Definition:** `Path.parent` is the path of the directory that contains the given path.
**2. In plain words:** `Path(__file__).resolve().parent` gives the folder the script lives in. It's the anchor for robust file paths.
**3. Example to quote:**
> "I took `.parent` of the script path to get its folder, then joined `data.txt` to it."
```python
print(Path("/a/b/c.txt").parent)   # /a/b
```
**Seen in:** `main.py`, script folder line and anchored path

## Relative path  `(concept)`
**1. Definition:** A relative path such as `data.txt` is interpreted starting from the current working directory.
**2. In plain words:** It's convenient but fragile: the same string points to different places depending on where you launch the program.
**3. Example to quote:**
> "`Path('data.txt').exists()` was `True` from the script folder and `False` from the repo root."
```python
print(Path("data.txt").exists())
```
**Seen in:** `main.py`, "Fragile" block

## Absolute path  `(concept)`
**1. Definition:** An absolute path specifies a location starting from the filesystem root (e.g. `/Users/me/project/data.txt`), independent of the working directory.
**2. In plain words:** It always points to the same file no matter where you run from, which makes it safer for debugging. Printing the absolute form reveals mistakes.
**3. Example to quote:**
> "I printed the absolute path before opening the file and noticed it pointed to the wrong folder."
```python
print(Path("data.txt").resolve())
```
**Seen in:** `main.py`; `WALKTHROUGH.md`, debugging practice

## Anchored path  `(concept)`
**1. Definition:** An anchored path is built from a fixed reference point, here the script's own location: `Path(__file__).resolve().parent / "data.txt"`.
**2. In plain words:** Instead of trusting the working directory, you tie the path to the script. It then works from any folder. That's the fix for `broken_paths.py`; verify it from two different directories.
**3. Example to quote:**
> "I changed `open('data.txt')` to open `Path(__file__).resolve().parent / 'data.txt'` and tested from both the script folder and the repo root."
```python
anchored = Path(__file__).resolve().parent / "data.txt"
```
**Seen in:** `main.py`, "Robust" block

## Path.exists()  `(standard library)`
**1. Definition:** `Path.exists()` returns `True` if the path points to an existing file or directory.
**2. In plain words:** A quick check before reading. In the demo it shows `True`/`False` for the relative and anchored versions of the same file.
**3. Example to quote:**
> "`relative.exists()` printed `False` from the repo root while `anchored.exists()` stayed `True`."
```python
print(Path("data.txt").exists())
```
**Seen in:** `main.py`

## Path.read_text()  `(standard library)`
**1. Definition:** `Path.read_text()` reads a whole file and returns its content as a string.
**2. In plain words:** One call to read a small text file. Add `.strip()` to remove the trailing newline.
**3. Example to quote:**
> "`anchored.read_text().strip()` returned `hello from data.txt`."
```python
print(Path("data.txt").read_text().strip())
```
**Seen in:** `main.py`, content line

## open() with a relative path  `(Python built-in)`
**1. Definition:** `open("data.txt")` opens a file located relative to the current working directory.
**2. In plain words:** It is exactly what makes `broken_paths.py` fragile: it works only when run from its own folder.
**3. Example to quote:**
> "`broken_paths.py` used `open('data.txt')`, so it crashed with `FileNotFoundError` when I ran it from the repo root."
```python
with open("data.txt") as f:
    print(f.read())
```
**Seen in:** `broken_paths.py`

## FileNotFoundError  `(Python built-in)`
**1. Definition:** `FileNotFoundError` is raised when a file or directory cannot be found at the given path.
**2. In plain words:** The message shows the path Python tried: `No such file or directory: 'data.txt'`. Print the absolute path, `os.getcwd()` and the directory listing, then compare them with where the file really is.
**3. Example to quote:**
> "`[Errno 2] No such file or directory: 'data.txt'` made me print `os.getcwd()` and the absolute path."
```python
print(os.listdir("."))
```
**Seen in:** `broken_paths.py`; `WALKTHROUGH.md`

## Case sensitivity  `(concept)`
**1. Definition:** Some filesystems (macOS default, Windows) ignore letter case in file names, while others (Linux, most Docker images) treat `data.txt` and `Data.txt` as different files.
**2. In plain words:** A path with the wrong capitalisation may work on your Mac but crash in a Linux container. Always match the exact case.
**3. Example to quote:**
> "My code read `Data.txt` fine on macOS, but the Linux Docker build failed because the real file is `data.txt`."
```python
print(Path("Data.txt").exists())   # macOS may say True
```
**Seen in:** `main.py` BREAK IT; `WALKTHROUGH.md`, experiment 1

## os.listdir  `(standard library)`
**1. Definition:** `os.listdir(path)` returns a list of the names of entries in a directory.
**2. In plain words:** It lets you see what files are actually in the folder Python is looking at, which often reveals that the file isn't there or is spelled differently.
**3. Example to quote:**
> "I called `os.listdir('.')` and saw `data.txt` was missing from the current folder."
```python
import os
print(os.listdir("."))
```
**Seen in:** `WALKTHROUGH.md`, debugging practice

## Works on my machine  `(concept)`
**1. Definition:** "Works on my machine" describes code that runs in one environment but fails in another because of hidden differences such as paths, case sensitivity, installed packages or the working directory.
**2. In plain words:** Paths are a frequent cause. Robust anchored paths and containerised environments reduce the problem.
**3. Example to quote:**
> "It worked locally but failed in CI because the working directory differed, so I anchored the path to `__file__`."
```python
anchored = Path(__file__).resolve().parent / "data.txt"
```
**Seen in:** `WALKTHROUGH.md`, experiment 1

## Docker WORKDIR  `(concept)`
**1. Definition:** In a Dockerfile, `WORKDIR` sets the working directory for subsequent instructions and for the running container; only files copied into the image exist there.
**2. In plain words:** Inside a container, "here" is the `WORKDIR`, and your laptop's files aren't there unless you `COPY` them. Relative paths that worked locally can fail in a container.
**3. Example to quote:**
> "In Docker, the cwd is `WORKDIR` and only copied files exist, so I print the cwd and listing when a path fails."
```dockerfile
WORKDIR /app
COPY data.txt .
```
**Seen in:** `WALKTHROUGH.md`, debugging practice

## Data directory via env var  `(concept)`
**1. Definition:** Reading a data folder location from an environment variable (e.g. `os.environ.get("DATA_DIR")`) makes the path configurable without editing code.
**2. In plain words:** Anchoring to `__file__` works for scripts, but packaged apps and containers need configuration. A data-directory setting solves that.
**3. Example to quote:**
> "For the deployed app I read the data folder from `DATA_DIR`, with the script folder as the default."
```python
data_dir = Path(os.environ.get("DATA_DIR", "."))
```
**Seen in:** `WALKTHROUGH.md`, trade-off
