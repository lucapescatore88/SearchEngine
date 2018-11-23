from wikiutils import parseWiki
from googleutils import parseGoogle
import warnings
warnings.simplefilter("ignore")

def runSearch(name,surname,midname="") :

    out = {}

    ## Parsing Wikipedia page
    #info, fulltext = parseWiki(name,surname,midname)
    #out.update(info)

    ## Getting google page to see if he it a polititian
    print "Now doing some serious NLP to see if he is a Politician"
    googleout = parseGoogle(name,surname,midname)
    #out["Politician"] = isPoliticianSimple(googleout+fulltext)
    out["Politician"] = True

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
    runSearch(args.name,args.surname)




