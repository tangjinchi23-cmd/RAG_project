
import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id        # Session id
        self.storage_path = storage_path    # Folder that stores files for the different session ids
        # Full file path
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # Make sure the folder exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence is a generic sequence, similar to list / tuple
        all_messages = list(self.messages)      # Existing message list
        all_messages.extend(messages)           # Merge new and existing into one list

        # Persist the data to a local file.
        # A class instance written to a file would just be a blob of bytes,
        # so for convenience we convert each BaseMessage into a dict and write it as JSON.
        # The official message_to_dict: a single message object (BaseMessage) -> dict
        # new_messages = []
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]
        # Write the data to the file
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property       # The @property decorator exposes the messages method as an attribute
    def messages(self) -> list[BaseMessage]:
        # File contents: list[dict]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)    # Returns list[dict]
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
