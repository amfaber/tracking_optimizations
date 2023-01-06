from .gpu_tracking import *
from .lib import *

def run_app():
    from .app import app
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050")
    app.run_server()