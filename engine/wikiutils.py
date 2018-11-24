from engineutils import loadCurrencies, cleanData, country_df
from HTMLParser import HTMLParser
from unidecode import unidecode
from bs4 import BeautifulSoup
import wikipedia, requests
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
        print "Found Profession tag!"

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

def findNationality(text) :

    #cleantext = text.encode('ascii',errors='ignore')
    cleantext = text
    cleantext = re.sub(r'\0xe2|\0xc3'," ",cleantext).lower()
    cleantext = re.sub("\\s+"," ",cleantext).lower()
    nationalities = country_df['Nationality'].tolist() ## List of nationalities to compare against
    nationalities.append("ENGLISH")
    countries = country_df['Name'].tolist() ## List of countries to compare against

    foundnat, foundcount = None, None
    for nat in nationalities :
        if nat.lower() in cleantext :
            foundnat = nat
    for country in countries :
        if country.lower() in cleantext :
            foundcount = country

    if foundnat == "ENGLISH" : foundnat = "BRITISH"

    ## If both are found give precedence to the nationality
    if foundnat is not None : return foundnat
    if foundcount is not None : 
        countryinfo = country_df.loc[country_df['Name'] == foundcount]
        return countryinfo['Nationality'].values[0]


def findWikiNationality(soup,bio=None,fulltext=None) :

    ## Find in infobox
    for tr in soup.findChildren("tr") :
        
        found = False
        for th in tr.findChildren("th") :
            if th.string == "Nationality" : 
                found = True 
                break

        if not found : continue
        print "Found Nationality tag!"
        td = tr.findChildren("td")[0]
        if td.string.upper() == "ENGLISH" : return "BRITISH"
        return str(td.string)

        break

    ## Find in biography

    if bio is not None :
        print "Searching for nationality in biography"
        nat = findNationality(bio)
        if nat is not None : return nat

    ## Find in full Wiki page: just returns first nation or nationality found
    ## Could add scoring to return most represented nationality 
    if fulltext is not None :
        print "Searching for nationality in full Wiki page"
        return findNationality(fulltext)

### Find the Date of Birth

def findWikiBirthDay(soup,bio=None) :

    ## Find in infobox
    for el in soup.findChildren("span",{'class':'bday'}) :
        return el.text

    ## Find in biography
    if bio is not None :

        print "Looking for birthday in biography"
        groups = re.match("(?i)\(.*?born.*?(\d{4}).*?\)",bio)
        if groups is not None and len(groups)>0 :
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

    money = 0.
    tr = getNetWorthTag(soup)

    if tr is None : return money
    print "Found Net Worth tag!"
    
    content = tr.__str__().lower().decode("utf8")
    money = float(re.findall("\\d+\\.\\d+",content)[0])    ## Getting value
    if 'billion' in content :
        money *= 1000

    curr = None
    for ir,row in currency_df.iterrows() :

        if row['Symbol'] in content :
            curr = row['Alphabetic Code']
        elif row['Alphabetic Code'] in content :
            curr = row['Alphabetic Code']

        ### Convert to US dollars
        if curr is not None :
            money = conv.convert(curr,"USD",money) 
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

    print "Now searching Wikipedia"
    query = '{name} {midname} {surname} {country}'.format(
                name    = name, 
                midname = midname,
                surname = surname,
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
        profs = ', '.join(profs)
    
    out = {
        'bio'        : bio, 
        'midname'    : midname,
        'profession' : profs,
        'bday'       : findWikiBirthDay(soup,bio),
        'money'      : findWikiNetWorth(soup),
        'country'    : findWikiNationality(soup,bio,req.text)
    }
    return out, cleanData(req.text)


