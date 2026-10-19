### Time complexity
<!-- cat: python -->
- **Technical definition:** A function describing how the number of basic operations grows with input size n, usually expressed in Big-O notation.
- **Plain English:** How much slower your code gets when the input gets bigger.
- **Example / analogy:** Looking for a name in an unsorted phone list (read every line) vs jumping straight to the page in an index.
- **Snippet:**
```python
for x in nums:      # one pass -> O(n)
    pass
```
- **Snippet explained:** A single loop does work proportional to len(nums), so doubling the list doubles the work.
- **Related terms:** [[Space complexity]], [[Big-O notation]], [[Brute force]]

### Space complexity
<!-- cat: python -->
- **Technical definition:** How much extra memory an algorithm needs as input size grows, not counting the input itself.
- **Plain English:** How much extra scratch paper your solution needs.
- **Example / analogy:** Two Sum with a dict keeps up to n entries: O(n) extra space.
- **Snippet:**
```python
seen = {}   # grows with the input -> O(n) space
```
- **Snippet explained:** The dict stores at most one entry per element, so memory grows linearly.
- **Related terms:** [[Time complexity]], [[Hash map (dict)]]

### Big-O notation
<!-- cat: python -->
- **Technical definition:** Asymptotic upper bound that describes the growth rate of an algorithm's cost, ignoring constants and lower-order terms.
- **Plain English:** A label for how fast the work grows: O(1), O(log n), O(n), O(n log n), O(n^2).
- **Example / analogy:** Washing one dish is O(1); washing n dishes O(n); comparing every pair of guests at a party O(n^2).
- **Snippet:**
```python
n = 1000
print(n * (n - 1) // 2)   # pair comparisons: 499500 ~ O(n^2)
```
- **Snippet explained:** The number of pairs grows with the square of n, which is why nested loops get slow.
- **Related terms:** [[Time complexity]], [[Brute force]], [[Space complexity]]

### Brute force
<!-- cat: python -->
- **Technical definition:** A solution that tries every possible candidate without exploiting structure in the problem.
- **Plain English:** The simplest approach that obviously works, even if slow.
- **Example / analogy:** Trying every key on a ring until the door opens.
- **Snippet:**
```python
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]
```
- **Snippet explained:** Checks all pairs, O(n^2); a correct baseline to compare an optimised solution against.
- **Related terms:** [[Time complexity]], [[Complement lookup]], [[Dry run]]

### Dry run
<!-- cat: python -->
- **Technical definition:** Manually executing code with a small example, recording variable values after each step.
- **Plain English:** Playing computer on paper before (or after) you code.
- **Example / analogy:** Following a recipe step by step with a tiny portion to check it works.
- **Snippet:**
```python
def total(nums):
    t = 0
    for i, x in enumerate(nums):
        t += x
        print(i, x, t)   # dry run printed
    return t
```
- **Snippet explained:** The print shows index, value and running total each loop so you can check your mental trace.
- **Related terms:** [[Edge case]], [[Assertion (assert)]], [[Brute force]]

### Edge case
<!-- cat: python -->
- **Technical definition:** An input at the boundary of valid input that often breaks naive code: empty, single element, duplicates, negatives, maximum size.
- **Plain English:** The weird inputs you must test: nothing, one item, all the same, negatives.
- **Example / analogy:** A door that works for adults but jams for toddlers or giants.
- **Snippet:**
```python
assert find_max([]) is None
assert find_max([-5, -2]) == -2
```
- **Snippet explained:** Two edge cases: empty input and an all-negative list.
- **Related terms:** [[Dry run]], [[Assertion (assert)]]

### Assertion (assert)
<!-- cat: python -->
- **Technical definition:** A statement that raises AssertionError if its condition is False; used for simple tests.
- **Plain English:** A tripwire: 'I claim this is true; crash loudly if not.'
- **Example / analogy:** A smoke detector: silent when all is fine.
- **Snippet:**
```python
assert sum([1, 2]) == 3
```
- **Snippet explained:** If the claim is false Python stops with AssertionError and the failing line.
- **Related terms:** [[Edge case]], [[Dry run]]

### Hash map (dict)
<!-- cat: python -->
- **Technical definition:** A key-value structure using a hash function to locate entries; average O(1) insert, lookup and delete. In Python: dict.
- **Plain English:** A lookup table: give a key, instantly get its value.
- **Example / analogy:** A coat-check: ticket number straight to your coat, no searching.
- **Snippet:**
```python
ages = {'amy': 30}
ages['bob'] = 25
print(ages['amy'])
```
- **Snippet explained:** Creates a dict, adds a key, reads a value; each step is a hash lookup.
- **Related terms:** [[Hash set]], [[Hashing]], [[Frequency counting]]

### Hash set
<!-- cat: python -->
- **Technical definition:** An unordered collection of unique hashable items with average O(1) membership test. In Python: set.
- **Plain English:** A bag that remembers what it has seen and ignores duplicates.
- **Example / analogy:** A guest list where checking a name is instant.
- **Snippet:**
```python
seen = set()
seen.add(3)
print(3 in seen)   # True
```
- **Snippet explained:** Adds 3 and tests membership in constant average time.
- **Related terms:** [[Hash map (dict)]], [[Duplicate detection]]

### Hashing
<!-- cat: python -->
- **Technical definition:** Mapping a key to an integer (hash) used to pick a storage slot; keys must be immutable/hashable.
- **Plain English:** Turning a key into a number that says where to store it.
- **Example / analogy:** A librarian who computes a shelf number from the book title.
- **Snippet:**
```python
print(hash('abc') == hash('abc'))   # True within one run
# hash([1, 2])  -> TypeError: unhashable type: 'list'
```
- **Snippet explained:** Equal strings hash equally; lists are mutable so Python refuses to hash them.
- **Related terms:** [[Hash map (dict)]], [[Tuple]]

### Frequency counting
<!-- cat: python -->
- **Technical definition:** Counting how many times each distinct item occurs, typically with a dict or collections.Counter.
- **Plain English:** A tally chart.
- **Example / analogy:** Counting votes by making a mark next to each candidate name.
- **Snippet:**
```python
from collections import Counter
print(Counter('banana'))   # Counter({'a': 3, 'n': 2, 'b': 1})
```
- **Snippet explained:** Counter builds the item -> count mapping in one call.
- **Related terms:** [[Counter]], [[Hash map (dict)]], [[Valid anagram]]

### Counter
<!-- cat: python -->
- **Technical definition:** collections.Counter: a dict subclass for counting hashable items; offers most_common(k) and arithmetic.
- **Plain English:** A ready-made tally chart.
- **Example / analogy:** A scoreboard that adds one each time you call out a name.
- **Snippet:**
```python
from collections import Counter
c = Counter([1, 1, 2])
print(c.most_common(1))   # [(1, 2)]
```
- **Snippet explained:** most_common(1) returns the single most frequent (item, count) pair.
- **Related terms:** [[Frequency counting]], [[defaultdict]]

### defaultdict
<!-- cat: python -->
- **Technical definition:** collections.defaultdict: a dict that creates a default value for missing keys via a factory function.
- **Plain English:** A dict that never raises KeyError; it makes an empty entry for you.
- **Example / analogy:** A filing cabinet that adds a new empty folder when you ask for a missing one.
- **Snippet:**
```python
from collections import defaultdict
groups = defaultdict(list)
groups['a'].append(1)
```
- **Snippet explained:** Accessing groups['a'] auto-creates [] then append adds to it.
- **Related terms:** [[Grouping by key]], [[Hash map (dict)]]

### Complement lookup
<!-- cat: python -->
- **Technical definition:** Searching a hash structure for target - x instead of scanning for a partner of x.
- **Plain English:** Instead of searching for a partner, ask 'have I already seen the number that completes me?'
- **Example / analogy:** At a dance, checking the registry for your partner instead of asking everyone.
- **Snippet:**
```python
complement = target - x
if complement in seen:
    return [seen[complement], i]
```
- **Snippet explained:** Constant-time dict check replaces an inner loop in Two Sum.
- **Related terms:** [[Hash map (dict)]], [[Brute force]], [[Two sum]]

### Two sum
<!-- cat: python -->
- **Technical definition:** Classic problem: find indices of two numbers in an array that add to a target; O(n) with a dict.
- **Plain English:** Find the pair that adds up to the goal.
- **Example / analogy:** Finding two coins in a jar that total exactly 9 rupees.
- **Snippet:**
```python
seen = {}
for i, x in enumerate(nums):
    if target - x in seen:
        return [seen[target - x], i]
    seen[x] = i
```
- **Snippet explained:** Check for the complement before storing the current number.
- **Related terms:** [[Complement lookup]], [[Two pointers]]

### Duplicate detection
<!-- cat: python -->
- **Technical definition:** Determining whether any value repeats, usually with a set (O(n)) or by sorting (O(n log n)).
- **Plain English:** Checking if anything shows up twice.
- **Example / analogy:** Ticking names on an attendance list and noticing someone signs twice.
- **Snippet:**
```python
def has_dup(nums):
    return len(set(nums)) != len(nums)
```
- **Snippet explained:** A set drops duplicates; if its size is smaller the list had repeats.
- **Related terms:** [[Hash set]], [[Duplicate detection]]

### Valid anagram
<!-- cat: python -->
- **Technical definition:** Check that two strings contain the same characters with the same counts; sort-compare O(n log n) or count O(n).
- **Plain English:** Same letters, different order.
- **Example / analogy:** 'listen' and 'silent'.
- **Snippet:**
```python
from collections import Counter
Counter('rat') == Counter('tar')   # True
```
- **Snippet explained:** Equal letter counts means anagram.
- **Related terms:** [[Frequency counting]], [[Counter]], [[Grouping by key]]

### Two pointers
<!-- cat: python -->
- **Technical definition:** Technique with two indices moving through a sequence (often from both ends of a sorted array) to avoid nested loops.
- **Plain English:** Two fingers walking along the list.
- **Example / analogy:** Two people meeting from opposite ends of a corridor.
- **Snippet:**
```python
left, right = 0, len(nums) - 1
while left < right:
    left += 1
    right -= 1
```
- **Snippet explained:** Pointers move inward until they meet.
- **Related terms:** [[Sliding window]], [[In-place algorithm]]

### Sliding window
<!-- cat: python -->
- **Technical definition:** Maintain a contiguous range [left, right] and update its aggregate incrementally while moving the edges.
- **Plain English:** A frame that slides along the list, adding one item and dropping one.
- **Example / analogy:** A magnifying glass moved along a line of text.
- **Snippet:**
```python
window += nums[right] - nums[right - k]
```
- **Snippet explained:** One element enters, one leaves; the sum updates in O(1).
- **Related terms:** [[Two pointers]], [[Prefix sum]]

### Prefix sum
<!-- cat: python -->
- **Technical definition:** An array where prefix[i] is the sum of the first i elements, enabling O(1) range-sum queries.
- **Plain English:** A running total you keep so you can subtract two totals to get any range.
- **Example / analogy:** Odometer readings: distance between two stops = difference of readings.
- **Snippet:**
```python
prefix = [0]
for x in nums:
    prefix.append(prefix[-1] + x)
range_sum = prefix[r + 1] - prefix[l]
```
- **Snippet explained:** Build running totals once, subtract two entries to get sum of nums[l..r].
- **Related terms:** [[Sliding window]], [[Subarray]]

### Subarray
<!-- cat: python -->
- **Technical definition:** A contiguous slice of an array (unlike a subsequence, which may skip elements).
- **Plain English:** A continuous chunk of the list.
- **Example / analogy:** A few adjacent pages of a book.
- **Snippet:**
```python
nums[2:5]   # elements at index 2,3,4
```
- **Snippet explained:** Slicing gives a contiguous subarray (copy).
- **Related terms:** [[Sliding window]], [[Prefix sum]]

### In-place algorithm
<!-- cat: python -->
- **Technical definition:** Algorithm that transforms the input using O(1) extra space by modifying the input structure.
- **Plain English:** Edit the list itself instead of making a new one.
- **Example / analogy:** Rearranging books on the same shelf rather than copying them to another shelf.
- **Snippet:**
```python
nums[i], nums[j] = nums[j], nums[i]
```
- **Snippet explained:** Tuple swap modifies the list directly.
- **Related terms:** [[Two pointers]], [[Space complexity]]

### Binary search
<!-- cat: python -->
- **Technical definition:** Search a sorted array by repeatedly halving the range; O(log n).
- **Plain English:** Guess the middle, throw away the half that can't contain the answer.
- **Example / analogy:** Finding a word in a dictionary by opening to the middle.
- **Snippet:**
```python
mid = (lo + hi) // 2
if nums[mid] < target:
    lo = mid + 1
else:
    hi = mid - 1
```
- **Snippet explained:** Moves the search bounds into the half that may contain the target.
- **Related terms:** [[Time complexity]], [[Sorting (stable)]]

### Sorting (stable)
<!-- cat: python -->
- **Technical definition:** Sorting where equal elements keep their original relative order; Python's sorted()/list.sort() use stable Timsort, O(n log n).
- **Plain English:** Ties stay in the order they were in before.
- **Example / analogy:** Sorting a class list by grade: students with the same grade stay alphabetical if they were before.
- **Snippet:**
```python
sorted([('a', 2), ('b', 1), ('c', 2)], key=lambda p: p[1])
```
- **Snippet explained:** Result keeps ('a',2) before ('c',2) because the sort is stable.
- **Related terms:** [[Lambda]], [[Binary search]]

### Lambda
<!-- cat: python -->
- **Technical definition:** An anonymous single-expression function written with the lambda keyword.
- **Plain English:** A tiny throw-away function.
- **Example / analogy:** A sticky note instruction rather than a full manual.
- **Snippet:**
```python
sorted(words, key=lambda w: len(w))
```
- **Snippet explained:** The lambda tells sort to order words by length.
- **Related terms:** [[Sorting (stable)]], [[List comprehension]]

### List comprehension
<!-- cat: python -->
- **Technical definition:** Compact syntax [expr for item in iterable if cond] to build a list.
- **Plain English:** A one-line loop that builds a list.
- **Example / analogy:** A conveyor belt that transforms each item as it passes.
- **Snippet:**
```python
squares = [x * x for x in range(5) if x % 2 == 0]   # [0, 4, 16]
```
- **Snippet explained:** Loops 0..4, keeps evens, squares them.
- **Related terms:** [[Generator expression]], [[Lambda]]

### Generator expression
<!-- cat: python -->
- **Technical definition:** Lazy comprehension written with parentheses that yields items one at a time instead of building a list.
- **Plain English:** A list comprehension that doesn't store everything.
- **Example / analogy:** A tap versus a filled bucket.
- **Snippet:**
```python
total = sum(1 for x in nums if x > 0)
```
- **Snippet explained:** Counts positives without creating an intermediate list.
- **Related terms:** [[List comprehension]]

### Tuple
<!-- cat: python -->
- **Technical definition:** An immutable, hashable ordered sequence; usable as a dict key.
- **Plain English:** A list you can't change.
- **Example / analogy:** A sealed envelope containing values.
- **Snippet:**
```python
key = (1, 0, 2)
d = {key: 'ok'}
```
- **Snippet explained:** A tuple is hashable so it can be a dict key; a list cannot.
- **Related terms:** [[Hashing]], [[Grouping by key]]

### Grouping by key
<!-- cat: python -->
- **Technical definition:** Placing items into buckets keyed by a function of the item (e.g., sorted letters for anagrams).
- **Plain English:** Sorting things into piles by a label.
- **Example / analogy:** Sorting laundry by colour.
- **Snippet:**
```python
groups = defaultdict(list)
for w in words:
    groups[''.join(sorted(w))].append(w)
```
- **Snippet explained:** Anagrams share the sorted-letters key and land in the same bucket.
- **Related terms:** [[defaultdict]], [[Tuple]], [[Valid anagram]]

### Enumerate
<!-- cat: python -->
- **Technical definition:** Built-in that yields (index, item) pairs while iterating.
- **Plain English:** Loop with a counter attached.
- **Example / analogy:** Numbering the items on a to-do list as you read them.
- **Snippet:**
```python
for i, ch in enumerate('ab'):
    print(i, ch)   # 0 a / 1 b
```
- **Snippet explained:** Gives both position and value without range(len()).
- **Related terms:** [[Two sum]], [[List comprehension]]
