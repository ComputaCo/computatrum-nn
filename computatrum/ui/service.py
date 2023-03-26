import sqlite3
import json
import os
from dataclasses import asdict
from typing import Callable, List

from computatrum.ui.message import Message


class ChatService:
    def __init__(self, db_name: str = "chat.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_table()
        self.listeners: List[Callable[[str], None]] = []

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                guid INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                text TEXT NOT NULL,
                time REAL NOT NULL
            )
        """
        )
        self.conn.commit()

    def save(self, fname: str):
        messages = self.get_all_messages()
        with open(fname, "w") as f:
            json.dump(messages, f)

    def load(self, fname: str):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname} not found")

        with open(fname, "r") as f:
            messages = json.load(f)

        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO messages (guid, name, text, time) VALUES (?, ?, ?, ?)
        """,
            [(m["guid"], m["name"], m["text"], m["time"]) for m in messages],
        )
        self.conn.commit()

        self.notify_listeners("load")

    def add(self, msg: Message):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (guid, name, text, time) VALUES (?, ?, ?, ?)
        """,
            (msg.guid, msg.name, msg.text, msg.time),
        )
        self.conn.commit()

        self.notify_listeners("add")

    def edit(self, guid: int, msg: Message):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE messages SET name=?, text=?, time=? WHERE guid=?
        """,
            (msg.name, msg.text, msg.time, guid),
        )
        self.conn.commit()

        self.notify_listeners("edit")

    def delete(self, guid: int):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages WHERE guid=?", (guid,))
        self.conn.commit()

        self.notify_listeners("delete")

    def get_all_messages(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT guid, name, text, time FROM messages")
        rows = cursor.fetchall()
        messages = [Message(row[0], row[1], row[2], row[3]) for row in rows]
        return [asdict(m) for m in messages]

    def register_listener(self, listener: Callable[[str], None]):
        self.listeners.append(listener)

    def notify_listeners(self, event: str):
        for listener in self.listeners:
            listener(event)
