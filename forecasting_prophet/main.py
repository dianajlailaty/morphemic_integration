import logging
from forecasting import prophetv2

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

e = prophetv2.Prophet()
#try:
#    e.start()
#except KeyboardInterrupt:
#    e.stop()
e.start()
while True:
	pass