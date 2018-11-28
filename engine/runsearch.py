from engineutils import convertCountry, resroot, config, internet_off
from googleutils import parseGoogle
from wikiutils import parseWiki
import pickle 

from checkFamous import isFamous, getFamousFeatures
from checkPolitician import scorePolitician

def printResult(data) :
    with open(resroot+"print_template.txt") as tmp :
        print tmp.read().format(**data)

def backupEntry(name,surname,data) :
    bkfile = open(resroot+"backup.pkl")
    bk = pickle.load(bkfile)
    bkfile.close()
    bk[(name,surname)] = data
    bkfile = open(resroot+"backup.pkl","w")
    bk = pickle.dump(bk,bkfile)
    bkfile.close()

def runSearch(name,surname,midname="",country="") :

    if config['usebackup'] or internet_off():
        with open(resroot+"backup.pkl") as file :
            bk = pickle.load(file)
            if (name,surname) in bk.keys() :
                out = bk[(name,surname)]
                printResult(out)
                return out
    if internet_off() : 
        print "Sorry there is no internet and I have no backup... can't do much..."
        return {}

    country_code, country_name = convertCountry(country)

    out = {'name' : name, 'surname' : surname, 'midname' : midname, 'country' : country}

    ## Parsing Wikipedia page
    print "Searching info on Wikipedia"
    info = parseWiki(name,surname,midname,country_name)
    out.update(info)

    ## Getting google page to see if he it a polititian
    print "Now doing some serious NLP on Google to see if a politician"
    googleout = parseGoogle(name,surname,midname,country_name)
    scorePol = scorePolitician(googleout)
    out["isPolitician"] = (scorePol > config['isPolitician_prob_threshold'])

    print "Now doing some ML to understand if famous"
    features = getFamousFeatures(name,surname,
                                isPolitician = scorePol,
                                country      = out["country"],
                                money        = out["money"])
    out["isFamous"] = isFamous(features)

    ### Make a backup
    backupEntry(name,surname,out)

    ### Print out results
    printResult(out)

    return out
    

if __name__ == '__main__' :

    from parser import parser
    args = parser.parse_args()

    print "\n\n"+"-"*40
    print "Welcome to a very useful search engine"
    print "Now searching info about '%s %s'" % (args.name, args.surname)
    print "-"*40+"\n"
    runSearch(name    = args.name.replace("\\s+",""),
              surname = args.surname.replace("\\s+",""),
              midname = args.midname.replace("\\s+",""),
              country = args.country.replace("\\s+",""))




