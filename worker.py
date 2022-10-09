import os
import redis
from rq import Worker, Queue, Connection
import urllib.parse


#redis_url = os.getenv('REDISTOGO_URL')
#urlparse.uses_netloc.append('redis')
#url = urlparse.urlparse(redis_url)
listen = ['high', 'default', 'low']
#conn = redis.Redis(host='ec2-44-195-118-205.compute-1.amazonaws.com', port=12399, db=0, password='pe2d47b0a31228934cdd18a04a1fb80cbcb533a7fa630ed9511f5be3c7f2af97')
#listen = ['high', 'default', 'low']

#redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

#conn = redis.from_url("redis://:pe2d47b0a31228934cdd18a04a1fb80cbcb533a7fa630ed9511f5be3c7f2af97b@ec2-44-195-118-205.compute-1.amazonaws.com:12399")

#redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
#redis = Redis()
#conn = redis.from_url(redis_url)

#redis_url = os.getenv('REDISTOGO_URL')

#urllib.parse.uses_netloc.append('redis')
#url = urllib.parse.urlparse(redis_url)
#conn = Redis(host=url.hostname, port=23070, db=0, password=url.password)

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)


if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
        