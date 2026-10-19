"""
GOAL   : Build prompts from reusable TEMPLATES instead of pasting strings by hand, using the
         three Python tools beginners meet: f-strings, str.format, string.Template.
RUN    : cd 07_prompt_engineering/03_prompt_templates && python main.py
EXPECTED OUTPUT: the same prompt rendered three ways, a safe-rendering demo, and a KeyError
         that is caught on purpose when a placeholder value is missing.
NEEDS  : nothing (standard library only).
"""
from string import Template  # Template is in Python's standard library

labels = ["positive", "negative", "neutral"]
text = "Battery life is great but the screen scratches easily."

# 1) f-string: the {} are filled IMMEDIATELY, at the line where the string is written.
fstring_prompt = f"Classify into {', '.join(labels)}.\nText: {text}"

# 2) str.format: the template is a normal string with {names}; fill it LATER with .format(...).
#    Great for templates you keep as constants at the top of a file.
FORMAT_TEMPLATE = "Classify into {labels}.\nText: {text}"
format_prompt = FORMAT_TEMPLATE.format(labels=", ".join(labels), text=text)

# 3) string.Template: placeholders look like $name. Curly braces are NOT special, so
#    it is safer when the template itself contains JSON (which is full of { }).
JSON_TEMPLATE = Template('Return JSON like {"label": "<one of $labels>"} for this text:\n$text')
template_prompt = JSON_TEMPLATE.substitute(labels="/".join(labels), text=text)

print("--- f-string ---\n" + fstring_prompt)
print("\n--- str.format ---\n" + format_prompt)
print("\n--- string.Template ---\n" + template_prompt)

# --- Pitfall 1: JSON braces inside str.format must be doubled {{ }} ------------------
BAD = 'Return {"label": "x"} for {text}'
try:
    BAD.format(text="hi")
except KeyError as err:   # Python reads '"label"' as a placeholder name -> KeyError
    print("\nKeyError caught on purpose:", err)
GOOD = 'Return {{"label": "x"}} for {text}'    # doubled braces print as single braces
print(GOOD.format(text="hi"))

# --- Pitfall 2: a missing value should fail loudly, not silently produce a bad prompt --
try:
    FORMAT_TEMPLATE.format(labels="a, b")          # forgot text=
except KeyError as err:
    print("Missing placeholder:", err)

# --- A tiny reusable function: a template + validation = a "prompt builder" -------------
def build_prompt(user_text: str, allowed: list[str]) -> str:   # type hints: str in, str out
    if not user_text.strip():
        raise ValueError("user_text is empty - refuse to build a prompt for nothing")
    return FORMAT_TEMPLATE.format(labels=", ".join(allowed), text=user_text.strip())

print("\n" + build_prompt("  fine, I guess  ", labels))

# TRY IT: add a third placeholder {tone} to FORMAT_TEMPLATE and fix every call.
# BREAK IT: call build_prompt("   ", labels) and read the traceback (ValueError, your own message).
# Versioning tip: store templates in constants named PROMPT_V1, PROMPT_V2 ... (used in step 07).
