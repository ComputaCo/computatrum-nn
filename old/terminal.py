from multiprocessing import Process
import os
import subprocess

from computatrum.utils.try_all import try_all


class Terminal:

    _process: Process

    def __init__(self):
        pass

    def read(self):
        self._process.stdout.readline()

    def write(self, text):
        self._process.std

    def start(self):
        def _start(command):
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True,
            )

        if os.name == "posix":  # Unix/Linux/MacOS
            term = os.environ.get("TERM_PROGRAM")
            try:
                try_all(
                    _start,
                    list=(
                        # chatGPT: "platform independant solution to find the default terminal emulator / command prompt ?"
                        (["open -a iTerm.app"] if term == "iTerm.app" else [])
                        + [
                            term,
                            os.environ.get("TERM"),
                            os.environ.get("TERMINAL"),
                            # https://superuser.com/questions/1153988/find-the-default-terminal-emulator
                            "x-terminal-emulator",
                            "urxvt",
                            "rxvt",
                            "termit",
                            "terminator",
                            os.environ.get("TERM"),
                        ]
                    ),
                )
            except FileNotFoundError:
                raise NotImplementedError(
                    'Cannot find terminal emulator. Please set the "TERM" environment variable.'
                )
        elif os.name == "nt":  # Windows
            # Default command prompt
            _start("cmd.exe")
        else:
            # Unsupported platform
            raise NotImplementedError("Unsupported platform")

    def stop(self):
        self._process.kill()
