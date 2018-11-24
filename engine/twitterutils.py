from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
from multiprocessing import Process
from datetime import datetime
import time, json, os, yaml
import subprocess as sb
import tweepy

root = os.getenv("PICTETROOT")

try :
    f = open(root+"/cfg.yml")
    config = yaml.load(f)
    print "Configuration:",config
except Exception as e: 
    print "Config file is not good"
    print(e)

consumer_key    = "uZEo4lMBcPClUMdPxMFynEAVT"
consumer_secret = "fW4l7eVPEl9mkPQPRNURJru7nYqduIQNTTMHzr3jMDxdT9tsi1"
access_token    = "1036140012943892480-l7h8Buu80n8fO9cchNRk3YQ87BSLVB"
access_secret   = "W4Ik5njuh8jscD6tdIEdRn3QyECJpFYslXELkBO2KgFCa"
auth = OAuthHandler(consumer_key, consumer_secret) 
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

shared = { 'file' : None, 'data' : [], 'time' : time.time() }

class TwitterListener(StreamListener):
 
    def on_data(self, data):
        try:
            shared['file'].write(data)
            if (time.time() - shared['time']) < config['tweet_time']:
                return True
            else :
                shared['file'].close()
                return False
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True

### Get the numeber of hits in the last N seconds
### The number can be tuned in cfg.yml 
def getTwitterCounts(name,surname) :
    
    fname = "{name}_{surname}_twitter.json".format(name=name,surname=surname)
    shared['file'] = open(fname, 'w')
    shared['time'] = time.time()

    print "Running Twitter streaming for %s seconds" % config['tweet_time']
    twitter_stream = Stream(auth, TwitterListener())
    
    ### Need to run it in a subprocess because is no hits are found,
    ### which is common for non famous people, the on_data is never calles
    ### and therefore the listener never stops. This enforces a timeout.
    p = Process(target=twitter_stream.filter, kwargs={'track':[surname]})
    p.start()
    p.join(timeout=float(config['tweet_time']))
    p.terminate()

    hits = open(fname).readlines()
    return len(hits)

### Get the numeber of follower if the user exists
def getTwitterFollowers(name,surname) :
    
    try:
    
        user=api.get_user(name+surname)
        print "Twitter user exists!"
        #for k,v in user.__dict__.iteritems() :
        #    print k, v
        return user.followers_count

    except Exception:
        print "Twitter user does not exist"
        return 0



if __name__ == "__main__" :

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("surname")
    args = parser.parse_args()

    print "Calculating number of hits..."
    print "Found", getTwitterCounts(args.name,args.surname), "hits"

    print "Calculating followers..."
    print "Found", getTwitterFollowers(args.name,args.surname), "followers"

