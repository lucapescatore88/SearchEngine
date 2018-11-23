###from lib.google_search_results import GoogleSearchResults
import urllib, urllib2, re, requests, sys, json
from engineutils import cleanData


def parseGoogle(name,surname,midname="",nhits=10) :

    querytxt = '{name} {midname} {surname}'.format(
                name    = name, 
                midname = midname,
                surname = surname )

    #queryurl = "https://cse.google.com/cse"
    queryurl = "https://www.googleapis.com/customsearch/v1?"

    f = { 
          'cx'  : '001410688560193712145:zdxhgvqdo_o',
          'key' : 'AIzaSyDVFkjqS8GeqzlxzRDtMgwFxQoL1wq8yOw',
          'q'   : querytxt 
        }

    print "Searching on Google"
    print querytxt

    myurl = queryurl+urllib.urlencode(f)
    print myurl
    req = requests.get(myurl)

    results = json.loads(req.text)["items"]

    #query = GoogleSearchResults({"q": querytxt})
    #results = query.get_dictionary()
    #if "organic_results" in results : 
    #    results = results["organic_results"][:nhits] ## Just take first 10
    #else :
    #    print "Google search failed! Skipping..."
    #    return ""

    fulldata = ""       ## Put all data together
    for r in results :

        print "Opening", r["link"]
        try :
            response = urllib2.urlopen(r["link"])
            #headers = response.info()
            data     = response.read().decode('utf-8')
            print data
            fulldata += data+"\n" 
            #print data
        except : 
            print "Skipping, something is wrong"
    
    fulldata = cleanData(fulldata)
    #print fulldata
    #print "\n\n\n"

    return fulldata




