"""
GOAL: Headless checks for every UI in this module (no browser, no server left running).
RUN: cd 12_streamlit_gradio && ../.venv/bin/python -m pytest -q test_apps.py     (or: ../.venv/bin/python test_apps.py)
NEEDS: streamlit, gradio, scikit-learn, fastapi.
"""
from pathlib import Path
from streamlit.testing.v1 import AppTest   # runs a Streamlit script without a browser

HERE = Path(__file__).resolve().parent


def test_step01():
    at = AppTest.from_file(str(HERE / "01_streamlit_basics/app.py")).run()
    assert not at.exception
    at.button[0].click().run()
    assert at.success[0].value == "HELLO WORLD"


def test_step02_predict_and_history():
    at = AppTest.from_file(str(HERE / "02_streamlit_model_state_uploads/app.py"), default_timeout=30).run()
    assert not at.exception
    # Inside a st.form, set the value and click submit in the SAME run (values are only sent on submit).
    at.text_input[0].set_value("I love this")
    at.button[0].click().run()          # first button = form submit
    assert not at.exception
    assert len(at.session_state.history) == 1
    assert at.session_state.history[0][1] in ("positive", "negative")


def test_step02_empty_text_shows_error():
    at = AppTest.from_file(str(HERE / "02_streamlit_model_state_uploads/app.py"), default_timeout=30).run()
    at.text_input[0].set_value("   ")
    at.button[0].click().run()
    assert len(at.error) == 1
    assert at.session_state.history == []


def test_step04_layout():
    at = AppTest.from_file(str(HERE / "04_streamlit_layout_style/app.py"), default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 2
    at.button(key="single_btn").click().run()
    assert not at.exception and (at.success or at.warning)
    at.button(key="cmp_btn").click().run()
    assert not at.exception and len(at.metric) == 2


def _load(path, name):
    """Import a script by file path (folder names start with digits, so a normal import won't work)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step05_gradio_function_and_error():
    import gradio as gr
    app = _load("05_gradio_basics/app.py", "g05")
    result = app.classify("I love this")
    assert set(result) == {"negative", "positive"} and abs(sum(result.values()) - 1) < 1e-6
    import pytest
    with pytest.raises(gr.Error):
        app.classify("   ")


def test_step06_gradio_blocks_callbacks():
    import gradio as gr, pytest
    app = _load("06_gradio_blocks_api_share/app.py", "g06")
    out, state, state2 = app.local_predict("I love this", [])
    assert len(state) == 1 and state is state2
    rows = app.batch_predict(str(HERE / "02_streamlit_model_state_uploads/sample_sentences.txt"))
    assert len(rows) == 4
    with pytest.raises(gr.Error):
        app.batch_predict(None)
    with pytest.raises(gr.Error):
        app.api_predict("hi", "http://127.0.0.1:9", progress=lambda *a, **k: None)   # nothing listens on port 9


def _start_mock_api(port):
    """Run common/mock_api.py with uvicorn inside a background thread; return (server, thread)."""
    import sys, threading, time, uvicorn
    sys.path.insert(0, str(HERE / "common"))
    import mock_api
    server = uvicorn.Server(uvicorn.Config(mock_api.app, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):                      # wait up to ~5 s for startup
        if server.started:
            break
        time.sleep(0.1)
    return server, thread


def test_step03_api_client_success_then_down(monkeypatch):
    port = 8765
    monkeypatch.setenv("API_URL", f"http://127.0.0.1:{port}")
    server, thread = _start_mock_api(port)
    try:
        at = AppTest.from_file(str(HERE / "03_streamlit_api_client/app.py"), default_timeout=30).run()
        at.text_area[0].set_value("I love this")
        at.button[0].click().run()          # AppTest lists main-area widgets first: button[0] = Predict, button[1] = sidebar health
        assert not at.exception and not at.error
        assert at.metric[0].value in ("positive", "negative")
    finally:
        server.should_exit = True           # ask uvicorn to stop
        thread.join(timeout=5)
    at = AppTest.from_file(str(HERE / "03_streamlit_api_client/app.py"), default_timeout=30).run()
    at.button[0].click().run()              # API is now down
    assert "Cannot reach the API" in at.error[0].value


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main(["-q", __file__]))
