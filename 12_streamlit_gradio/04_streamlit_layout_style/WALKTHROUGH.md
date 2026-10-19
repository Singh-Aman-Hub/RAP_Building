# Step 04 - Layouts, columns, tabs and basic customisation

## Goal
Arrange a UI sensibly and customise its look with page config and a theme file.

## Run it
```bash
cd 12_streamlit_gradio/04_streamlit_layout_style
../../.venv/bin/streamlit run app.py      # must be run from this folder for .streamlit/config.toml
```
Verified headlessly (see ../test_apps.py): Classify and Compare tabs render with no exceptions.

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `with st.sidebar:` / `st.sidebar.x()` | put widgets in the side panel | |
| `a, b = st.columns(2)` | tuple unpacking of two column objects | `a, b = (1, 2)` |
| `with left:` vs `c1.metric(...)` | two equivalent ways to target a container | |
| `(la, sa), (lb, sb) = f(a), f(b)` | nested unpacking | `(x, y), z = (1, 2), 3` |
| `key="a"` | unique widget identity; required when labels repeat | |
| TOML `[theme]` | config file format: sections and key = value | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `st.set_page_config` | streamlit | title, icon, layout | `layout="wide"` | None |
| `st.tabs` | streamlit | tabbed sections | list of names | list of containers |
| `st.columns` | streamlit | side-by-side | number or ratio list | list of columns |
| `st.expander`, `st.status` | streamlit | collapsible / progress box | label, expanded | context manager |
| `st.slider`, `st.checkbox` | streamlit | numeric / boolean input | min, max, default | number / bool |
| `time.sleep` | built-in (time) | pause | seconds | None |

## Execution trace
Run starts -> page config -> sidebar widgets return current slider/checkbox values -> tabs drawn -> only the clicked button's branch executes; threshold compares to the model confidence; layout containers just decide *where* output appears, not *what* runs.

## Experiments
1. Move the slider and note the Single tab message only changes after a click (button returns True only on click rerun).
2. Switch to `layout="centered"`.
3. Change `primaryColor`.

## Debugging practice
Two widgets with identical label and no key -> `DuplicateWidgetID`. Fix with unique `key=`.

## Rebuild checklist
- [ ] Sidebar + two columns + tabs from memory.
- [ ] Explain why `set_page_config` goes first.

## Say it aloud
"Layout in Streamlit is declarative: sidebar for settings, tabs for views, columns for side-by-side comparison. Appearance comes from set_page_config and a config.toml theme. Trade-off: deep custom styling needs fragile CSS hacks, so Streamlit suits internal tools and demos rather than pixel-perfect products."
