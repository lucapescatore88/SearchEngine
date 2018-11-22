#from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import wikipedia
from engineutils import *
wikipedia.set_lang("en")

def runSearch(name,surname,midname="") :

    query = '{name} {midname} {surname}'.format(
                name    = args.name, 
                midname = args.midname,
                surname = args.surname )

    pages = wikipedia.search(query)

    mainpage = pages[0] 
    if(args.name in mainpage and args.surname in mainpage) :
        page = wikipedia.page(mainpage)
    else :
        return "Something is wrong... no Wiki page found"

    req = requests.get(page.url)
    html = req.text
    soup = BeautifulSoup(html);

    f = open("log","w")
    f.write(html.encode('utf-8'))
    #f.write(req.text.encode('utf-8'))
    f.close()
    
    ### Remember to ass midname to search options
    midname, bio = findWikiBiography(soup,args.name,args.surname)
    out = {
        'bio'        : bio, 'midname' : midname,
        'profession' : findWikiProfession(soup),
        'bday'       : findWikiBirthDay(soup),
        'money'      : findWikiNetWorth(soup),
        'nation'     : findWikiNationality(soup) 
    }

    return out
    

if __name__ == '__main__' :

    from engineutils import parser
    args = parser.parse_args()
    runSearch(args.name,args.surname)




