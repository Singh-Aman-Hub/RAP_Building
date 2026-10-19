"""
native_schema.py -- helper for step 09.

1) Loads the registry/validator/executor from step 03 BY FILE PATH. (We cannot write
   `from 03_tool_registry_and_validation.main import ...` because a Python name cannot start with a digit,
   and two files called main.py would clash.  importlib lets us load a file under a name we choose.)
2) Converts a registry entry into the JSON-Schema tool definition that native tool-calling APIs expect.
"""
import importlib.util
from pathlib import Path

_step03 = Path(__file__).resolve().parents[1] / "03_tool_registry_and_validation" / "main.py"
_spec = importlib.util.spec_from_file_location("step03_registry", _step03)   # (module name, file path)
step03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step03)                                              # actually run the file

REGISTRY, validate_call, execute = step03.REGISTRY, step03.validate_call, step03.execute


def to_native_schema(name, spec):
    """Build an Anthropic-style tool definition from our registry entry."""
    json_type = {str: "string", int: "number", float: "number", (int, float): "number"}
    properties = {}
    for param, py_type in spec["params"].items():
        properties[param] = {"type": json_type[py_type]}
        if param in spec["allowed"]:
            properties[param]["enum"] = spec["allowed"][param]                # allowed values -> JSON Schema enum
    return {"name": name, "description": spec["description"],
            "input_schema": {"type": "object", "properties": properties, "required": spec["required"]}}
