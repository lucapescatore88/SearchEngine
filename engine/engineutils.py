import re, os
import pandas as pd

dataroot = os.getenv("PICTETROOT")+"/resources/"

### Load currency data

def loadCurrencies() :

    from babel.numbers import format_currency

    currency_df = pd.read_csv(dataroot+"currencies.csv")
    symbols = []
    
    for row in currency_df.iterrows():
        #print row[1]['Alphabetic Code'],
        symbol = row[1]['Alphabetic Code']
        try : 
            s = format_currency(1, row[1]['Alphabetic Code'],locale='en_US')
            symbol = s.replace("1.00","")
        except : pass
        symbols.append(symbol)

    currency_df['Symbol'] = pd.Series(symbols)
    return currency_df

def loadCountries() :

    country_df = pd.read_csv(dataroot+"countries.csv")
    return country_df

def cleanData(data) :

    data = data.replace("\n","")
    data = re.sub(r'(?i)<[/]?[ibp]>',"",data)                 # Remove p,i,b tags but not content
    data = re.sub(r'(?i)<[/]?html.*?>',"",data)               # Remove html tags
    data = re.sub(r'(?i)<[/]?body.*?>',"",data)               # Remove body tags
    data = re.sub(r'(?i)<[/]?div.*?>',"",data)                # Remove div tags
    data = re.sub(r'(?i)<[ ]*?head.*?/head[ ]*>',"",data)     # Remove full head tag with its content
    data = re.sub(r'(?i)<[ ]*?img.*?/>',"",data)              # Remove full img tags
    data = re.sub(r'(?i)<[ ]*?footer.*?/footer[ ]*>',"",data) # Remove full footer tag with its content
    data = re.sub(r'(?i)<[ ]*?script.*?/script[ ]*>',"",data) # Remove full script tag with its content
    data = re.sub(r'(?i)<[ ]*?form.*?/form[ ]*>',"",data)     # Remove full form tag with its content
    data = re.sub(r'(?i)<!--.*?-->',"",data)                  # Remove comments
    data = re.sub(r'(?i)<[ ]*?sup.*?/sup[ ]*>',"",data)       # Remove citations
    data = re.sub(r'(?i)<[/]?a.*?>',"",data)                  # Remove links (but not their text)
    data = re.sub(r'(?i)<[/]?br>',"\n",data)                  # change <br> to \n
    data = re.sub(r'<.*?!DOCTYPE.+?html.*?>',"\n",data)       # Remove Doctype

    return data
