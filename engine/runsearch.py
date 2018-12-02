from engineutils import convertCountry, resroot, internet_off, loadConfig
from checkPolitician import PoliticianChecker
from checkFamous import FamousChecker
from wikiutils import WikiParser
import pickle, sys, os 

def printResult(data) :
    if data['money'] == -1 : data['money'] = "N/A "
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

class Search :

    def __init__(self,name,surname,midname="",country="") :
        
        self.config  = loadConfig()
        self.name    = name
        self.surname = surname
        self.midname = midname
        self.country = country

    def run(self) :

        #print "Searching ",name,surname
        if self.config['usebackup'] or internet_off():
            with open(resroot+"backup.pkl") as file :
                bk = pickle.load(file)
                if (self.name,self.surname) in bk.keys() :
                    out = bk[(self.name,self.surname)]
                    printResult(out)
                    return out
        if internet_off() : 
            print "Sorry there is no internet and I have no backup... can't do much..."
            return {}
    
        country_code, country_name = convertCountry(self.country)
    
        out = {'name' : self.name, 'surname' : self.surname, 'midname' : self.midname, 'country' : country_name}
    
        ## Parsing Wikipedia page
        print "Searching info on Wikipedia"
        wiki = WikiParser(self.name,self.surname,self.midname,self.country,self.config)
        out.update(wiki.parse())
    
        ## Test if politician
        print "Now doing some serious NLP to see if a politician"
        #googleout = parseGoogle(name,surname,midname,country_name)

        polCheck = PoliticianChecker(self.config)
        scorePol = polCheck.scorePolitician(out)
        out["isPolitician"] = (scorePol > self.config['isPolitician_prob_threshold'])
        
        ## Test if famous
        print "Now doing some ML to understand if famous"
        famCheck = FamousChecker(self.config)
        famCheck.getFamousFeatures(self.name,self.surname,
                                    isPolitician = scorePol,
                                    country      = out["country"],
                                    money        = out["money"])
        out["isFamous"] = famCheck.isFamous()
    
        ### Make a backup
        backupEntry(self.name,self.surname,out)
    
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
    search = Search(name    = args.name.replace("\\s+",""),
                    surname = args.surname.replace("\\s+",""),
                    midname = args.midname.replace("\\s+",""),
                    country = args.country.replace("\\s+",""))
    search.run()




