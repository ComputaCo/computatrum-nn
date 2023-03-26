from dataclasses import dataclass


@dataclass
class Message:
    guid: int
    name: str
    text: str
    time: float  # seconds since epoch
