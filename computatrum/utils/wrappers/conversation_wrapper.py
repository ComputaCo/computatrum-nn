from dataclasses import dataclass
import threading
import time
import gymnasium as gym

from computatrum.utils.wrappers.env_wrapper import EnvWrapper




class ConversationWrapper(EnvWrapper):

    messages: list[Message] = []
    new_message: Message = None
    new_message_lock: threading.Lock = threading.Lock()

    def __init__(self, env, gui=False, stt=False, tts=False):
        super().__init__(env)
        self.gui = gui
        self.stt = stt
        self.tts = tts

    def reset(self):
        self.messages.empty()
        # maybe show gui
        if self.gui:
            if self._gui:
                self._gui.close()
                del self._gui
            self._gui = # TODO make simple gui
        # maybe start stt listener
        if self.stt:
            # TODO start stt listener
        return self.env.reset()
    
    def step(self, action):
        if 'text_out' in action:
            self.output_message(Message('agent', action['text_out'], time.time()))
        observation, reward, done, info = self.env.step(action)
        with new_message_lock:
            observation['text_in'] = new_message.text
            new_message = None
        return observation, reward, done, info
    
    def output_message(self, message):
        self.messages.append(message)
        if self.gui:
            import tkinter as tk
            # TODO update gui
        if self.tts and message.role == 'agent':
            # TODO use tts
            pass
    
    def show_gui(self):
        from flask import Flask, request, jsonify
        import threading
        from typing import Callable

        class WebServer:
            def __init__(self):
                self.app = Flask(__name__)
                self.new_message_listeners = []

                self.app.add_url_rule('/post', 'post', self._post, methods=['POST'])

            def start(self, address, port):
                self.server_thread = threading.Thread(target=self._run, args=(address, port))
                self.server_thread.start()

            def stop(self):
                self.app.shutdown()

            def post(self, role, msg):
                for listener in self.new_message_listeners:
                    listener(role, msg)

            def _run(self, address, port):
                self.app.run(host=address, port=port, threaded=True)

            def _post(self):
                data = request.get_json()
                role = data.get('role')
                msg = data.get('msg')
                self.post(role, msg)
                return jsonify({'status': 'success'})
        
        def message_listener(role, msg):
            print(f"{role}: {msg}")

        web_server = WebServer()
        web_server.new_message_listeners.append(message_listener)
        web_server.start('localhost', 5000)