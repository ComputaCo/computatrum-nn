import threading
import typer
import signal
import os


from computatrum.ui.service import ChatService
from computatrum.ui.web import ChatServer

app = typer.Typer()

@app.command('start')
def start():
    service = ChatService()
    server = ChatServer(service)

    def handle_sigint(sig, frame):
        print("\nShutting down gracefully...")
        server.stop()
        typer.Exit()

    signal.signal(signal.SIGINT, handle_sigint)

    server_thread = threading.Thread(target=server.start, args=("localhost", 5000))
    server_thread.start()

    print("Press Ctrl+C to exit")
    server_thread.join()

if __name__ == "__main__":
    app()
