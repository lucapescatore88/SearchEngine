import re
from HTMLParser import HTMLParser
hparse = HTMLParser()

def findWikiProfession(soup) :

    tags = ["Occupation","Profession"]
    professions = []
    for tr in soup.findChildren("tr") :
        
        found = False
        for th in tr.findChildren("th") :
            if th.string in tags : 
                found = True 
                break

        if not found : continue
        #print "----------------------------------------"
        #print "Found Profession tag!"
        #print tr

        for li in tr.findChildren("li") :
            links =  li.findChildren("a")
            prof = li.string
            if len(links)>0 : prof = links[0].string
            professions.append(hparse.unescape(prof))
        
        if len(professions)==0 : 
            data = tr.findChildren("td")[0].__str__()
            data = hparse.unescape(data)
            data = re.sub(r"(?i)<br/>","",data)
            data = re.sub(r"(?i)<[/]?a.*?>","",data)
            data = re.sub(r"(?i)<[/]?[ibp]>","",data)
            data = re.sub(r"(?i)<[/]?tr>","",data)
            data = re.sub(r"(?i)<[/]?td>","",data)
            data = re.sub(r"(?i)<br/>","",data)
            data = re.sub(r"<td class=\"role\">","",data)
            print data
            professions.append(data)
        break
    return professions

def findWikiNationality(soup) :

    for tr in soup.findChildren("tr") :
        
        found = False
        for th in tr.findChildren("th") :
            if th.string == "Nationality" : 
                found = True 
                break

        if not found : continue
        print "Found Nationality tag!"
        td = tr.findChildren("td")[0]
        return td.string

        break


def findWikiBirthDay(soup) :

    for el in soup.findChildren("span",{'class':'bday'}) :
        return el.text

    names, bio = findWikiBiography(soup,name,surname)
    groups = re.match("(?i)\\(.*?born.*?(\\d{4}).*?\\)")
    if len(groups)>0 :
        print groups
        return groups[0] 

    return "No date found... sorry"

#from forex_python.converter import CurrencyRates
#c = CurrencyRates()
from babel import Locale
#c.get_symbol('GBP')
#for s in Locale.currency_symbols :
#    print s

def findWikiNetWorth(soup) :

    money = None
    for tr in soup.findChildren("tr") :
        
        found = False
        for th in tr.findChildren("th") :
            if th.string == "Net worth" : 
                found = True 
                break

        if not found : continue
        print "Found Net Worth tag!"
        td = tr.findChildren("td")[0]
        a = tr.findChildren("a",{'class':'mw-redirect'})[0]
        #print tr
        print tr.__str__()
        #print a.string
        content = tr.__str__().lower()
        money = float(re.findall("\d+\.\d+",content)[0])
        if 'billion' in content :
            money *= 1000
        #elif 'millions' in content : pass
        break

    return money
        #l = Locale.parse('de_DE')
        #c.convert('USD', 'USD', 10)

def findWikiBiography(soup,name,surname) :

    bio = None
    names = ""

    ## Biography always contains a <b> tag containin the person name
    ## This also contains the full name! So can obtain the middle name
    for p in soup.findChildren("p") :
        found = False
        for b in p.findChildren("b") :
            if name in b.string and surname in b.string :
                bio = p
                names = b.string.replace(name,"")
                names = names.replace(surname,"")
                names = names.lstrip().rstrip()
                found = True
        if found : break

    if bio is None : return None, None
    
    print "Midnames = ", names

    bio = re.sub(r"(?i)<[/]?[ibp]>","",bio.__str__())   # Remove p,i,b tags
    bio = re.sub(r"(?i)<sup.*?/sup[ ]*>","",bio)        # Remove citations
    bio = re.sub(r"(?i)<[/]?a.*?>","",bio)              # Remove links (but not their text)
    bio = re.sub(r"(?i)<span.*?/span[ ]*>","",bio)
    bio = re.sub(r"(?i)<small.*?/small[ ]*>","",bio)
    bio = hparse.unescape(bio)
    print bio

    return names, bio




