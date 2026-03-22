"""
Standalone static file server for the dashboard HTML.
Use this if you want to open the HTML directly WITHOUT running the FastAPI backend.
The HTML will show skeleton loaders and connection errors when API is offline.

Run: python dashboard/static_server.py
Then open: http://localhost:9090
"""
import http.server
import socketserver
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = 9090

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/index.html'
        return super().do_GET()
    def log_message(self, *a): pass  # suppress noise

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Static dashboard: http://localhost:{PORT}")
    print("NOTE: Real data requires the FastAPI server (run.sh)")
    httpd.serve_forever()
