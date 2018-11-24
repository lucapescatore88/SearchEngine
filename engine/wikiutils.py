from engineutils import loadCurrencies, cleanData
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup
import wikipedia, requests
from wikiutils import *
import pandas as pd
import re, os

## Initial lodings

wikipedia.set_lang("en")
hparse = HTMLParser()
currency_df = loadCurrencies()


### Find the Profession

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
        print "Found Profession tag!"
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
            professions.append(data)
        break
    return professions


### Find the Nationality

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


### Find the Date of Birth

def findWikiBirthDay(soup,name,surname) :

    for el in soup.findChildren("span",{'class':'bday'}) :
        return el.text

    names, bio = findWikiBiography(soup,name,surname)
    groups = re.match("(?i)\\(.*?born.*?(\\d{4}).*?\\)")
    if len(groups)>0 :
        #print groups
        return groups[0] 

    return "No date found... sorry"


### Find how rich they are

from forex_python.converter import CurrencyRates
conv = CurrencyRates()

def getNetWorthTag(soup) :

    for tr in soup.findChildren("tr") :
        for th in tr.findChildren("th") :
            if th.string == "Net worth" : 
                return tr


def findWikiNetWorth(soup) :

    money = None
    tr = getNetWorthTag(soup)

    if tr is not None : print "Found Net Worth tag!"
    
    content = tr.__str__().lower().decode("utf8")
    money = float(re.findall("\d+\.\d+",content)[0])    ## Getting value
    if 'billion' in content :
        money *= 1000

    #print content
    curr = None
    for row in currency_df.iterrows() :

        if row [1]['Symbol'] in content :
            curr = row[1]['Alphabetic Code']
        elif row[1]['Alphabetic Code'] in content :
            curr = row[1]['Alphabetic Code']

        if curr is not None :
            money = conv.convert(curr,"USD",money) 
            #print money, "M USD"
            break

    return money


## Find the biography 

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
    
    bio = re.sub(r"(?i)<[/]?[ibp]>","",bio.__str__())   # Remove p,i,b tags
    bio = re.sub(r"(?i)<sup.*?/sup[ ]*>","",bio)        # Remove citations
    bio = re.sub(r"(?i)<[/]?a.*?>","",bio)              # Remove links (but not their text)
    bio = re.sub(r"(?i)<span.*?/span[ ]*>","",bio)
    bio = re.sub(r"(?i)<small.*?/small[ ]*>","",bio)
    bio = hparse.unescape(bio)

    return names, bio

def parseWiki(name,surname,midname="",country="") :

    query = '{name} {midname} {surname} {country}'.format(
                name    = name, 
                midname = midname,
                surname = surname
                country = country ).replace("\\s+"," ")

    pages = wikipedia.search(query)

    mainpage = pages[0] 
    if(name in mainpage and surname in mainpage) :
        page = wikipedia.page(mainpage)
    else :
        return "Something is wrong... no Wiki page found"

    req = requests.get(page.url)
    soup = BeautifulSoup(req.text)

    ### Just a log of the page for testing purposes
    #f = open("log","w")
    #f.write(req.text.encode('utf-8'))
    #f.close()

    ### Remember to add midname to search options
    midname, bio = findWikiBiography(soup,name,surname)
    profs = findWikiProfession(soup)
    if profs is not None :
        profs = ' '.join(profs)
    
    out = {
        'bio'        : bio, 
        'midname'    : midname,
        'profession' : profs,
        'bday'       : findWikiBirthDay(soup,name,surname),
        'money'      : findWikiNetWorth(soup),
        'country'    : findWikiNationality(soup) 
    }
    return out, cleanData(req.text)


