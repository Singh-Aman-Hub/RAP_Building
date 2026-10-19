"""Solution for broken_syntax.py: 3 errors = missing ':' after def, missing ']' , '=' instead of '=='."""
print("now the whole file compiles, so this runs")


def greet(name):                 # 1) colon added
    print("Hello, " + name)


values = [1, 2, 3]               # 2) closing bracket added
print(values)

if values[0] == 1:               # 3) == compares, = assigns
    print("one")
greet("world")
