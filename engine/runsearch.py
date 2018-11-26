from engineutils import convertCountry, resroot
from googleutils import parseGoogle
from wikiutils import parseWiki

from checkFamous import isFamous, isFamousVoting, getFamousFeatures
from checkPolitician import isPoliticianSimple, isPolitician

def runSearch(name,surname,midname="",country="") :

    country_code, country_name = convertCountry(country)

    out = {'name' : name, 'surname' : surname, 'midname' : midname, 'country' : country}

    ## Parsing Wikipedia page
    print "Searching info on Wikipedia"
    info = parseWiki(name,surname,midname,country_name)
    out.update(info)

    ## Getting google page to see if he it a polititian
    print "Now doing some serious NLP on Google to see if a politician"
    googleout = parseGoogle(name,surname,midname,country_name)
    out["isPolitician"] = bool(isPoliticianSimple(googleout))

    print "Now doing some ML to understand if famous"
    features = getFamousFeatures(name,surname,
                                isPolitician = out["isPolitician"],
                                country      = out["country"],
                                money        = out["money"],
                                job          = out["profession"])
    out["isFamous"] = isFamous(features)

    with open(resroot+"print_template.txt") as tmp :
        print tmp.read().format(**out)

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




