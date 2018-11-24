import urllib, urllib2, re, requests, sys, json
from engineutils import cleanData

def parseGoogle(name,surname,midname="",country="",nhits=10) :

    querytxt = '{name} {midname} {surname} {country}'.format(
                name    = name, 
                midname = midname,
                surname = surname,
                country = country ).replace("\\s+"," ")

    #queryurl = "https://cse.google.com/cse"
    queryurl = "https://www.googleapis.com/customsearch/v1?"

    f = { 
          'cx'     : '001410688560193712145:zdxhgvqdo_o',
          'key'    : 'AIzaSyDVFkjqS8GeqzlxzRDtMgwFxQoL1wq8yOw',
          'lr'     : 'lang_en',
          #'rights' : 'cc_publicdomain',
          'num'    : nhits,
          'q'      : querytxt 
        }

    print "Searching on Google"
    print querytxt

    myurl = queryurl+urllib.urlencode(f)
    req = requests.get(myurl)

    results = json.loads(req.text)["items"]

    fulldata = ""       ## Put all data together
    nok = 0
    for r in results :

        try :
            response = urllib2.urlopen(r["link"])
            #headers = response.info()
            data     = response.read().decode('utf-8')
            if "<body" in data : nok+=1
            fulldata += data+"\n" 
        except : 
            pass
            #print "Skipping, something is wrong"
    
    print "Analysed", nok, "links"
    return cleanData(fulldata)




