import logging
import sys

logging.basicConfig(stream=sys.stdout
                    , level=logging.DEBUG
                    , format='%(asctime)s ---- %(message)s'
                    , datefmt='%m/%d/%Y %I:%M:%S')
