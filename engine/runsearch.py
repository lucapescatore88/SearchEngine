from engineutils import convertCountry
from googleutils import parseGoogle
from wikiutils import parseWiki

from checkFamous import isFamous, isFamousVoting
from checkPolitician import isPoliticianSimple, isPolitician

import warnings
warnings.simplefilter("ignore")

def runSearch(name,surname,midname="",country="") :

    #print type(name), type(name), type(name), type(country)
    if not isinstance(name,str) or not isinstance(surname,str) :
        print "All inputs need to be strings"
        return {}
    if  not isinstance(midname,str) or not isinstance(country,str) :
        print "All inputs need to be strings"
        return {}

    country_code, country_name = convertCountry(country)

    out = {'name' : name, 'surname' : surname, 'midname' : midname, 'country' : country}

    ## Parsing Wikipedia page
    print "Searching Wikipedia"
    info, fulltext = parseWiki(name,surname,midname,country_name)
    out.update(info)

    ## Getting google page to see if he it a polititian
    print "Now doing some serious NLP to see if a politician"
    #googleout = parseGoogle(name,surname,midname,country_name)
    #out["Politician"] = bool(isPoliticianSimple(googleout+fulltext))
    out["Politician"] = True

    print "Now doing some ML to understand if famous"
    
    #features = getFamousFeatures(name,surname,out["Politician"],
    #                             out["country"],out["money"],out["profession"])
    #out["Famous"] = isFamous(features)

    out["Famous"] = True

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
    runSearch(args.name,args.surname,midname=args.midname,country=args.country)




