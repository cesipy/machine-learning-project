import datetime

FILENAME = "logs/log_"

class Logger: 
    def __init__(self):
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        self.filename = FILENAME + today + ".txt"

    def log(self, message: str) -> None:
        timestamp: str = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry: str = f"{timestamp}- {message}\n"

        with open(self.filename, "a") as f:
            f.write(log_entry)