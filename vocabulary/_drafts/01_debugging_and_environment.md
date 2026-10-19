### Traceback
<!-- cat: python -->
- **Technical definition:** The printed call stack at the point an exception was raised, oldest call first and the exception type/message last.
- **Plain English:** The trail of function calls that led to the crash.
- **Example / analogy:** Footprints leading to the spot where something broke.
- **Snippet:**
```python
def f(): return {}['x']
f()
```
- **Snippet explained:** Last line `KeyError: 'x'`; the frame above points to the `return` line.
- **Related terms:** [[Exception]], [[Stack frame]]

### Stack frame
<!-- cat: python -->
- **Technical definition:** One entry in the call stack: a function call with its file, line number and local variables.
- **Plain English:** One step in the chain of calls.
- **Example / analogy:** One floor in a tall building you are descending.
- **Snippet:**
```python
import traceback
for fr in traceback.extract_tb(err.__traceback__):
    print(fr.name, fr.lineno)
```
- **Snippet explained:** Lists each function and line in the failing chain.
- **Related terms:** [[Traceback]]

### SyntaxError
<!-- cat: python -->
- **Technical definition:** An error found while compiling source code, before any of it runs, because the text is not valid Python.
- **Plain English:** Grammar mistake in the code.
- **Example / analogy:** A sentence with a missing full stop that nobody can read.
- **Snippet:**
```python
compile('if x print(1)', 'f', 'exec')
```
- **Snippet explained:** Raises SyntaxError, so no line of the file would have executed.
- **Related terms:** [[Traceback]], [[Runtime error]]

### Runtime error
<!-- cat: python -->
- **Technical definition:** An error that occurs while a valid program is executing (KeyError, TypeError, ZeroDivisionError ...).
- **Plain English:** A crash during running.
- **Example / analogy:** A recipe that is fine on paper but fails when the egg is missing.
- **Snippet:**
```python
1 / 0
```
- **Snippet explained:** Raises ZeroDivisionError at run time.
- **Related terms:** [[SyntaxError]], [[Exception]]

### KeyError
<!-- cat: python -->
- **Technical definition:** Raised when a dict key is not found.
- **Plain English:** Asked for something not in the lookup table.
- **Example / analogy:** Looking up a name missing from the phone book.
- **Snippet:**
```python
d = {'a': 1}
d.get('b', 0)
```
- **Snippet explained:** `.get` avoids KeyError by returning a default.
- **Related terms:** [[Dictionary]]

### TypeError
<!-- cat: python -->
- **Technical definition:** Raised when an operation gets a value of the wrong type or wrong number of arguments.
- **Plain English:** Mixing incompatible kinds of values.
- **Example / analogy:** Adding 'apples' to 3.
- **Snippet:**
```python
'age: ' + 30
```
- **Snippet explained:** Fails: cannot concatenate str and int; use f-strings.
- **Related terms:** [[Type conversion]], [[Type hint]]

### AttributeError
<!-- cat: python -->
- **Technical definition:** Raised when an object has no such attribute or method; often the object is None.
- **Plain English:** Asked an object for something it doesn't have.
- **Example / analogy:** Asking a bicycle for its steering-wheel size.
- **Snippet:**
```python
None.upper()
```
- **Snippet explained:** `AttributeError: 'NoneType' object has no attribute 'upper'` usually means a missing `return`.
- **Related terms:** [[TypeError]]

### ImportError / ModuleNotFoundError
<!-- cat: python -->
- **Technical definition:** Raised when a module cannot be found (ModuleNotFoundError) or a name cannot be imported from it (ImportError).
- **Plain English:** Python can't find or load what you asked to import.
- **Example / analogy:** A book not in this library branch.
- **Snippet:**
```python
import numpyy
```
- **Snippet explained:** `No module named 'numpyy'`: typo or not installed in this interpreter.
- **Related terms:** [[Virtual environment]], [[Module shadowing]]

### Virtual environment
<!-- cat: python -->
- **Technical definition:** An isolated directory with its own Python interpreter and installed packages, created with `python -m venv .venv`.
- **Plain English:** A private toolbox per project.
- **Example / analogy:** Separate kitchens so recipes don't share ingredients.
- **Snippet:**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install requests
```
- **Snippet explained:** Creates, activates and installs packages only into that project.
- **Related terms:** [[pip]], [[Dependency conflict]]

### pip
<!-- cat: python -->
- **Technical definition:** Python's package installer; `python -m pip` ensures it matches the interpreter.
- **Plain English:** The app store for Python libraries.
- **Example / analogy:** Ordering parts from a catalogue.
- **Snippet:**
```bash
python -m pip install numpy
python -m pip show numpy
python -m pip check
```
- **Snippet explained:** Install, inspect and check dependency consistency.
- **Related terms:** [[Virtual environment]], [[requirements.txt]]

### requirements.txt
<!-- cat: python -->
- **Technical definition:** A text file listing package names (and versions) so an environment can be recreated with `pip install -r`.
- **Plain English:** A shopping list for the project's libraries.
- **Example / analogy:** A recipe's ingredient list.
- **Snippet:**
```bash
numpy==2.5.4
requests>=2.30
```
- **Snippet explained:** One requirement per line; pins keep versions reproducible.
- **Related terms:** [[pip]], [[Dependency conflict]]

### Dependency conflict
<!-- cat: python -->
- **Technical definition:** Two packages need incompatible versions of a third, so they can't both be satisfied.
- **Plain English:** Libraries disagreeing about which version to use.
- **Example / analogy:** Two recipes demanding different oven temperatures.
- **Snippet:**
```bash
python -m pip check
```
- **Snippet explained:** Reports installed packages whose requirements are not met.
- **Related terms:** [[pip]], [[Virtual environment]]

### Module shadowing
<!-- cat: python -->
- **Technical definition:** A local file or folder with the same name as a library hides the real one because the script folder is first on sys.path.
- **Plain English:** Your file named like a library breaks the library.
- **Example / analogy:** A fake sign hiding the real shop.
- **Snippet:**
```python
# a local file named json.py makes this fail:
import json
json.loads('1')
```
- **Snippet explained:** Python imports your json.py, which lacks `loads`.
- **Related terms:** [[Module and import]], [[ImportError / ModuleNotFoundError]]

### sys.path
<!-- cat: python -->
- **Technical definition:** The list of directories Python searches when importing, starting with the script's folder.
- **Plain English:** Where Python looks for modules.
- **Example / analogy:** The list of shelves checked in order.
- **Snippet:**
```python
import sys
print(sys.path[:2])
```
- **Snippet explained:** Shows the first two search locations.
- **Related terms:** [[Module shadowing]]

### Working directory
<!-- cat: python -->
- **Technical definition:** The folder a process starts in; relative paths are resolved against it (`os.getcwd()`).
- **Plain English:** The folder you launched the command from.
- **Example / analogy:** Where you are standing in the building when you ask for 'the red folder'.
- **Snippet:**
```python
import os
print(os.getcwd())
```
- **Snippet explained:** Shows the cwd; it differs from the script's folder if you run from elsewhere.
- **Related terms:** [[pathlib.Path]], [[FileNotFoundError]]

### FileNotFoundError
<!-- cat: python -->
- **Technical definition:** Raised when opening a path that doesn't exist; often a wrong cwd, spelling or case.
- **Plain English:** The file isn't where Python is looking.
- **Example / analogy:** Searching the wrong drawer.
- **Snippet:**
```python
from pathlib import Path
print(Path('x.txt').resolve())
```
- **Snippet explained:** Prints the exact absolute path Python would try.
- **Related terms:** [[Working directory]]

### Input validation
<!-- cat: python -->
- **Technical definition:** Checking that data has the expected type, range and shape before using it.
- **Plain English:** Checking ingredients before cooking.
- **Example / analogy:** A bouncer checking IDs.
- **Snippet:**
```python
if not isinstance(n, int) or n <= 0:
    raise ValueError('n must be a positive int')
```
- **Snippet explained:** Rejects bad input early with a clear message.
- **Related terms:** [[Pydantic model]], [[TypeError]]

### Pydantic model
<!-- cat: python -->
- **Technical definition:** A class deriving from BaseModel whose type-hinted fields are validated and coerced when instances are created.
- **Plain English:** A schema that checks data automatically.
- **Example / analogy:** A form that rejects invalid answers.
- **Snippet:**
```python
from pydantic import BaseModel, Field
class R(BaseModel):
    n: int = Field(gt=0)
R(n='5')
```
- **Snippet explained:** The string '5' is coerced to 5; `-1` would raise ValidationError.
- **Related terms:** [[Input validation]], [[Type hint]]

### HTTP status code
<!-- cat: python -->
- **Technical definition:** A 3-digit number in an HTTP response: 2xx success, 4xx client error, 5xx server error.
- **Plain English:** The server's short answer: ok, you erred, or I erred.
- **Example / analogy:** A shop assistant saying 'here you go', 'we don't have that' or 'our till is broken'.
- **Snippet:**
```python
r = requests.get(url, timeout=3)
r.status_code
```
- **Snippet explained:** 200 ok, 404 not found, 422 invalid body, 500 server bug.
- **Related terms:** [[Timeout]], [[ConnectionError]]

### Timeout
<!-- cat: python -->
- **Technical definition:** The maximum time to wait for a response before raising an error; requests has none by default.
- **Plain English:** How long you're willing to wait.
- **Example / analogy:** Hanging up if nobody answers after 10 rings.
- **Snippet:**
```python
requests.get(url, timeout=3)
```
- **Snippet explained:** Raises `requests.Timeout` if no response within 3 s.
- **Related terms:** [[HTTP status code]], [[ConnectionError]]

### ConnectionError
<!-- cat: python -->
- **Technical definition:** Raised when a network connection cannot be established (server down, wrong host/port).
- **Plain English:** Nobody is listening at that address.
- **Example / analogy:** Dialling a number that is out of service.
- **Snippet:**
```python
try:
    requests.get('http://127.0.0.1:9', timeout=1)
except requests.ConnectionError:
    print('unreachable')
```
- **Snippet explained:** Catches the unreachable-server case separately from a slow server.
- **Related terms:** [[Timeout]], [[HTTP status code]]

### Retry with backoff
<!-- cat: python -->
- **Technical definition:** Re-attempting a failed transient operation after increasing delays.
- **Plain English:** Try again, waiting longer each time.
- **Example / analogy:** Knocking again after 1 s, then 2 s, then 4 s.
- **Snippet:**
```python
import time
for wait in [1, 2, 4]:
    try:
        ...
        break
    except requests.Timeout:
        time.sleep(wait)
```
- **Snippet explained:** Waits longer after each failed attempt.
- **Related terms:** [[Timeout]], [[HTTP status code]]

### Environment variable
<!-- cat: python -->
- **Technical definition:** A named string value supplied by the operating system/process (read via os.environ).
- **Plain English:** A setting given from outside the program.
- **Example / analogy:** A sticky note on the desk the program reads on startup.
- **Snippet:**
```python
import os
port = int(os.environ.get('PORT', '8000'))
```
- **Snippet explained:** Reads PORT as text, falling back to '8000', then converts to int.
- **Related terms:** [[dotenv file]], [[Type conversion]]

### dotenv file
<!-- cat: python -->
- **Technical definition:** .env file of KEY=VALUE lines loaded into the environment by python-dotenv; never committed to git.
- **Plain English:** A local file of settings and secrets.
- **Example / analogy:** A private settings notebook.
- **Snippet:**
```python
from dotenv import load_dotenv
load_dotenv()
```
- **Snippet explained:** Reads `.env` and copies its pairs into os.environ.
- **Related terms:** [[Environment variable]]

### Logging
<!-- cat: python -->
- **Technical definition:** The logging module records messages with levels (DEBUG..CRITICAL), timestamps and destinations.
- **Plain English:** A structured diary of what the program did.
- **Example / analogy:** A ship's log vs scribbled notes.
- **Snippet:**
```python
import logging
logging.basicConfig(level='INFO')
logging.getLogger('app').info('started')
```
- **Snippet explained:** Prints a timestamped INFO line; DEBUG lines would be hidden.
- **Related terms:** [[Print debugging]], [[Log level]]

### Log level
<!-- cat: python -->
- **Technical definition:** A severity label (DEBUG < INFO < WARNING < ERROR < CRITICAL); messages below the configured level are dropped.
- **Plain English:** How important a message is.
- **Example / analogy:** Volume knob for diagnostics.
- **Snippet:**
```python
log.warning('empty input')
```
- **Snippet explained:** Shown at INFO or DEBUG levels, hidden if level is ERROR.
- **Related terms:** [[Logging]]

### Print debugging
<!-- cat: python -->
- **Technical definition:** Inserting print statements to inspect values while running.
- **Plain English:** Quick-and-dirty peeking at values.
- **Example / analogy:** Torch in a dark room.
- **Snippet:**
```python
print(type(x), repr(x), len(x))
```
- **Snippet explained:** Shows the kind, exact value and size of x.
- **Related terms:** [[Logging]], [[Breakpoint]]

### Breakpoint
<!-- cat: python -->
- **Technical definition:** `breakpoint()` pauses the program in the pdb debugger so you can inspect variables and step through lines.
- **Plain English:** Pause button inside your code.
- **Example / analogy:** Freezing a movie to look at one frame.
- **Snippet:**
```python
breakpoint()  # commands: n, s, p x, c, q
```
- **Snippet explained:** n runs the next line, p prints a variable, c continues.
- **Related terms:** [[Print debugging]], [[Assertion]]

### Assertion
<!-- cat: python -->
- **Technical definition:** `assert cond, msg` raises AssertionError when cond is false; used to state and check assumptions or tests.
- **Plain English:** A claim the program checks for you.
- **Example / analogy:** A smoke detector for wrong assumptions.
- **Snippet:**
```python
assert average([2, 4]) == 3, 'average broken'
```
- **Snippet explained:** Silent if true; raises with the message if false.
- **Related terms:** [[Regression test]]

### Minimal reproducible example
<!-- cat: python -->
- **Technical definition:** The smallest self-contained code and data that still shows a bug, with expected vs actual output.
- **Plain English:** A tiny version of the problem anyone can run.
- **Example / analogy:** Isolating the squeak in a car by removing parts.
- **Snippet:**
```python
def add_item(x, bucket=[]):
    bucket.append(x); return bucket
print(add_item(1), add_item(2))
```
- **Snippet explained:** Six lines reproduce a bug found in a 40-line program.
- **Related terms:** [[Root-cause analysis]], [[Mutable default argument trap]]

### Root-cause analysis
<!-- cat: python -->
- **Technical definition:** Systematically tracing a failure back to the underlying cause, testing hypotheses one at a time.
- **Plain English:** Finding why, not just what.
- **Example / analogy:** Treating the illness, not the fever.
- **Snippet:**
n/a
- **Snippet explained:** Observe, reproduce, shrink, hypothesise, test, fix, verify.
- **Related terms:** [[Minimal reproducible example]], [[Regression test]]

### Regression test
<!-- cat: python -->
- **Technical definition:** A test added after a bug fix that fails on the old behaviour and passes on the fixed one, preventing it returning.
- **Plain English:** A guard so the bug cannot return unnoticed.
- **Example / analogy:** A plug in the hole you just fixed.
- **Snippet:**
```python
assert add_item_fixed('b') == ['b']
```
- **Snippet explained:** Would have failed with the buggy shared list.
- **Related terms:** [[Assertion]], [[Root-cause analysis]]

### Docker container logs
<!-- cat: python -->
- **Technical definition:** Output from a container's stdout/stderr read with `docker logs`; the first place to look when a container misbehaves.
- **Plain English:** The container's printed output, tracebacks included.
- **Example / analogy:** The black-box recorder of a flight.
- **Snippet:**
```bash
docker ps -a
docker logs --tail 50 mycontainer
```
- **Snippet explained:** Shows whether it exited and the last 50 log lines.
- **Related terms:** [[Traceback]], [[Logging]]
