from engineutils import cleanData, config, resroot
import urllib, urllib2, re, requests, sys, json
import pandas as pd

def parseGoogle(name,surname,midname="",country="",nhits=config['n_google_links']) :

    querytxt = '{name} {midname} {surname} {country}'.format(
                name    = name, 
                midname = midname,
                surname = surname,
                country = country ).replace("\\s+"," ")

    queryurl = "https://www.googleapis.com/customsearch/v1?"

    f = { 
          'cx'     : '001410688560193712145:zdxhgvqdo_o',
          'key'    : 'AIzaSyDVFkjqS8GeqzlxzRDtMgwFxQoL1wq8yOw',
          'lr'     : 'lang_en',
          #'rights' : 'cc_publicdomain',
          'num'    : nhits,
          'q'      : querytxt 
        }

    myurl = queryurl+urllib.urlencode(f)
    req = requests.get(myurl)
    results = json.loads(req.text)

    if 'items' not in results :
        print "Sorry you finished your Google searches for today"
        return ""
    results = results["items"]

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
    
    print "Analysed", nok, "links"
    return cleanData(fulldata)


if __name__ == "__main__" :

    from engineutils import getPeopleData, trainparser
    import pickle
    
    args = trainparser.parse_args()

    def dummyParseGoogle(name,surname) :
        return {'name' : name, 'surname' : surname,
                'googletext' : parseGoogle(name,surname) }

    googledata = getPeopleData("GoogleData",args.trainfile,
                        myfunction=dummyParseGoogle,
                        usebackup=args.usebackup,
                        save=True)

    entries = []
    for key,dat in googledata.iteritems() :

        d = {}
        d['isPol'] = key[2]
        d['isFam'] = key[3]
        d.update(dat)
        #print dat['name'], dat['surname'], len(dat['googletext'])
        entries.append(d)

    df = pd.DataFrame.from_dict(entries)
    pickle.dump(df,open(resroot+"GoogleDF.pkl","w"))
    print "Done! The DataFrame is in ", resroot+"GoogleDF.pkl"



