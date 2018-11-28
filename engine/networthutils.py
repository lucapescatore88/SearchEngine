from engineutils import NAval, convertCountry, convertToUSD
from bs4 import BeautifulSoup
import urllib2, re


def parseNetWorth(name,surname,data) :

    site = "https://www.celebritynetworth.com"
    urls = [site+"/richest-celebrities/actors/{name}-{surname}-net-worth/",
            site+"/richest-businessmen/richest-billionaires/{name}-{surname}-net-worth/",
            site+"/richest-politicians/{name}-{surname}-net-worth/"]
    
    resp = None
    for url in urls :
        try :
            link = url.format(name=name.lower(),surname=surname.lower())
            resp = urllib2.urlopen(link)
            break
        except :
            continue

    if resp is None : return {}
    res = resp.read().decode('utf-8')
    soup = BeautifulSoup(res)

    if data['money'] == -1 :
        netwdiv = soup.findChildren("div",{'class':['meta_row','networth']})
        if len(netwdiv) > 0 :
            for val in netwdiv[0].findChildren("div",{'class':['value']}) :
                money = float(re.findall("\\d+[\\.]*?\\d+]*?",val.text)[0])
                if "billions" in val.text.lower() :
                    money *= 1000
                data['money'] = int(money)
                money = convertToUSD(money,val.text)
                break

    if data['country'] == NAval :
        for span in soup.findChildren("span",{'property':['v:country-name']}) :
            country = span.text.replace("of America","")
            code, data['country'] = convertCountry(country)

    if data['profession'] == NAval :
        for span in soup.findChildren("span",{'property':['v:role']}) :
            data['profession'] = span.text

    return data



