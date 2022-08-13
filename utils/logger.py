import logging
import platform

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def getLogger(name=None, level=logging.INFO):
  logger = logging.getLogger("Main" if name is None else name)
  logger.setLevel(level)
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter("[%(name)s]:%(levelname)s:%(message)s"))
  handler.setLevel(level)
  logger.addHandler(handler)
  return logger

# Initialize Root Logger
getLogger()
for h in logging.root.handlers:
  logging.root.removeHandler(h)

LOGGER = logging.getLogger("Main")


