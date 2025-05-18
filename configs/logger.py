import logging
import os


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.logger.setLevel(logging.DEBUG)

            log_path = os.path.join('logs', 'my_app.log')

            file_handler = logging.FileHandler(log_path)
            # for prinout to console
            console_handler = logging.StreamHandler()

            file_handler.setLevel(logging.DEBUG)
            console_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            cls._instance.logger.addHandler(file_handler)
            cls._instance.logger.addHandler(console_handler)

        return cls._instance
    
    def get_logger(self):
        """Return logger property of the instance"""

        return self.logger
