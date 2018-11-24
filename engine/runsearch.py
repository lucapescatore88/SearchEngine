from engineutils import convertCountry
from googleutils import parseGoogle
from wikiutils import parseWiki

from checkFamous import isFamous, isFamousVoting, getFamousFeatures
from checkPolitician import isPoliticianSimple, isPolitician

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)

def runSearch(name,surname,midname="",country="") :

    country_code, country_name = convertCountry(country)

    out = {'name' : name, 'surname' : surname, 'midname' : midname, 'country' : country}

    ## Parsing Wikipedia page
    print "Searching Wikipedia"
    info, fulltext = parseWiki(name,surname,midname,country_name)
    out.update(info)

    ## Getting google page to see if he it a polititian
    print "Now doing some serious NLP to see if a politician"
    googleout = parseGoogle(name,surname,midname,country_name)
    out["Politician"] = bool(isPoliticianSimple(googleout+fulltext))
    #out["Politician"] = True

    print "Now doing some ML to understand if famous"
    
    features = getFamousFeatures(name,surname,
                                isPolitician = out["Politician"],
                                country      = out["country"],
                                money        = out["money"],
                                job          = out["profession"])
    out["Famous"] = isFamous(features)
    #out["Famous"] = True

    for k,v in out.iteritems() :
        print "-",k,":"
        print "     ", v

    return out
    

if __name__ == '__main__' :

    from parser import parser
    args = parser.parse_args()

    print "\n\n"+"-"*40
    print "Welcome to a very useful search engine"
    print "Now searching info about '", args.name, args.surname+"'"
    print "-"*40+"\n"
    runSearch(name    = args.name.replace("\\s+",""),
              surname = args.surname.replace("\\s+",""),
              midname = args.midname.replace("\\s+",""),
              country = args.country.replace("\\s+",""))




