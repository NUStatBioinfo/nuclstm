import os
import redis

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'nucleosomes'
    SESSION_TYPE = 'redis'
    SESSION_REDIS = redis.from_url('localhost:6379')
    CACHE_TYPE = 'simple'