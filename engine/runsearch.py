#from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import wikipedia
from engineutils import *
wikipedia.set_lang("en")

def runSearch(name,surname,midname="") :

    query = '{name} {midname} {surname}'.format(
                name    = name, 
                midname = midname,
                surname = surname )

    pages = wikipedia.search(query)

    mainpage = pages[0] 
    if(name in mainpage and surname in mainpage) :
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
    midname, bio = findWikiBiography(soup,name,surname)
    profs = findWikiProfession(soup)
    if profs is not None :
        print profs
        profs = ' '.join(profs)
    out = {
        'bio'        : bio, 
        'midname'    : midname,
        'profession' : profs,
        'bday'       : findWikiBirthDay(soup),
        'money'      : findWikiNetWorth(soup),
        'nation'     : findWikiNationality(soup) 
    }

    return out
    

if __name__ == '__main__' :

    from engineutils import parser
    args = parser.parse_args()
    runSearch(args.name,args.surname)




