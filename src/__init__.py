import json
import numpy as np
import logging

from src.configs.logging_config import config


def get_logger():
    root = logging.getLogger()
    root.setLevel(config['level'])

    handler = logging.StreamHandler(config['stream_handler'])
    handler.setLevel(config['level'])
    formatter = logging.Formatter(config['format'])
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


logger = get_logger()

logger.info('VQA research logger is init')


def set_log_level(level: str = 'INFO'):
    global logger
    if level.lower() == 'info':
        config['level'] = logging.INFO
    elif level.lower() == 'debug':
        config['level'] = logging.DEBUG
    else:
        pass
    logger = get_logger()


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float16)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
