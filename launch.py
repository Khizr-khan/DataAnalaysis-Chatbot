# launch.py
import os, sys, time, webbrowser, socket, traceback
from pathlib import Path

def base_dir():
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

def log_and_popup(msg: str):
    (base_dir() / "app_error.log").write_text(msg, encoding="utf-8")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("App error", msg)
    except Exception:
        pass

def free_port(start=8501, span=200):
    for p in range(start, start+span):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return start

def ensure_groq_key():
    if os.getenv("GROQ_API_KEY"):
        return
    key_file = base_dir() / "groq_key.txt"
    if key_file.exists():
        val = key_file.read_text(encoding="utf-8").strip()
        if val:
            os.environ["GROQ_API_KEY"] = val
            return
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        root = tk.Tk(); root.withdraw()
        val = simpledialog.askstring("Groq API Key", "Paste your GROQ_API_KEY:")
        if not val:
            messagebox.showerror("Missing key", "A GROQ_API_KEY is required to run the app.")
            sys.exit(1)
        key_file.write_text(val.strip(), encoding="utf-8")
        os.environ["GROQ_API_KEY"] = val.strip()
    except Exception:
        log_and_popup("Tk not available. Create 'groq_key.txt' next to the app with your GROQ_API_KEY (no quotes).")
        sys.exit(1)

def main():
    ensure_groq_key()
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENTMODE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from streamlit.web.cli import main as st_main

    port = free_port()
    url = f"http://localhost:{port}"
    app_path = base_dir() / "app.py"

    import threading
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(url)), daemon=True).start()

    sys.argv = [
        "streamlit", "run", str(app_path),
        "--global.developmentMode", "false",
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    st_main()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        log_and_popup("The app crashed. See 'app_error.log' next to the app for details.\n\n" + err)
        sys.exit(1)
