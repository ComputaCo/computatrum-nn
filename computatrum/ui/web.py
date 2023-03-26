import json
import logging
import asyncio
import threading
import websockets

from fastapi import FastAPI, WebSocket, HTTPException
from uvicorn import Config, Server
from pydantic import BaseModel

from computatrum.ui.message import Message
from computatrum.ui.service import ChatService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
)


class ChatServer(FastAPI):
    class MessageModel(BaseModel):
        guid: int
        content: str
        user: str

    def __init__(self, chat_service: ChatService):
        super().__init__()
        self.chat_service = chat_service
        self.websockets = set()
        self.register_websocket_listener()
        self.define_routes()

    def define_routes(self):
        self.add_api_route("/", self.head_index, methods=["HEAD"])
        self.add_api_route("/messages", self.get_messages, methods=["GET"])
        self.add_api_route("/messages", self.add_message, methods=["POST"])
        self.add_api_route("/messages/{guid}", self.edit_message, methods=["PUT"])
        self.add_api_route(
            "/messages/{guid}", self.delete_message, methods=["DELETE"]
        )

    async def websocket_handler(self, websocket: WebSocket):
        self.websockets.add(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # If needed, you can process received messages from clients here
        finally:
            self.websockets.remove(websocket)

    def head_index(self):
        logging.info("Head index")
        return {}

    def get_messages(self):
        logging.info("Getting messages")
        messages = self.chat_service.get_all_messages()
        return messages

    def add_message(self, message: MessageModel):
        logging.info("Adding message")
        self.chat_service.add(Message(**message.dict()))
        return {"status": "success", "message": "Message added"}

    def edit_message(self, guid: int, message: MessageModel):
        logging.info("Editing message")
        self.chat_service.edit(guid, Message(**message.dict()))
        return {"status": "success", "message": "Message updated"}

    def delete_message(self, guid: int):
        logging.info("Deleting message")
        self.chat_service.delete(guid)
        return {"status": "success", "message": "Message deleted"}

    def register_websocket_listener(self):
        self.chat_service.register_listener(self.notify_clients)

    def notify_clients(self, event):
        asyncio.run_coroutine_threadsafe(
            self.broadcast_event(event), asyncio.get_event_loop()
        )

    async def broadcast_event(self, event):
        if self.websockets:
            message = json.dumps(event)
            await asyncio.gather(
                *[websocket.send_text(message) for websocket in self.websockets]
            )

    def start(self, addr, port, ws_port=None):
        ws_port = ws_port or port + 1

        def run_server():
            logging.info(f"Starting server on {addr}:{port}")
            uvicorn_config = Config(app=self, host=addr, port=port)
            server = Server(config=uvicorn_config)
            server.run()

        async def run_ws_server():
            async with websockets.serve(self.websocket_handler, addr, ws_port):
                logging.info(f"Starting WebSocket server on {addr}:{ws_port}")
                await asyncio.Future()

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.start()

        self.ws_server_thread = threading.Thread(
            target=asyncio.run, args=(run_ws_server(),)
        )
        self.ws_server_thread.start()

    def stop(self):
        if self.server_thread is not None:
            self.server_thread.join()
            self.server_thread = None
        if self.ws_server_thread is not None:
            self.ws_server_thread.join()
            self.ws_server_thread = None