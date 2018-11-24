import pandas as pd
import re, os

dataroot = os.getenv("PICTETROOT")+"/resources/"

### Function to load the DB with currency codes and add symbols

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


### Function to load the DB with country codes

def loadCountries() :

    df = pd.read_csv(dataroot+"countries.csv")
    df['A3'] = df['A3'].apply(lambda x: re.sub("\\s+","",x))
    df['A2'] = df['A2'].apply(lambda x: re.sub("\\s+","",x))
    df['Name'] = df['Name'].apply(lambda x: re.sub("\\s+"," ",x).lstrip().rstrip().upper())
    df["Code"] = df["Code"].apply(pd.to_numeric)
    #print df.dtypes

    return df

country_df = loadCountries()

### Function to clean webpage data

def cleanData(data) :

    data = data.replace("\n","")

    data = re.sub(r'(?i)<[ ]*?head.*?/head[ ]*>'," ",data)     # Remove full head tag with its content
    data = re.sub(r'(?i)<[ ]*?style.*?/style[ ]*>'," ",data)   # Remove full style tag with its content
    data = re.sub(r'(?i)<[ ]*?svg.*?/svg[ ]*>'," ",data)       # Remove full svg tag with its content
    data = re.sub(r'(?i)<[ ]*?img.*?/>'," ",data)              # Remove full img tags
    data = re.sub(r'(?i)<[ ]*?footer.*?/footer[ ]*>'," ",data) # Remove full footer tag with its content
    data = re.sub(r'(?i)<[ ]*?script.*?/script[ ]*>'," ",data) # Remove full script tag with its content
    data = re.sub(r'(?i)<[ ]*?form.*?/form[ ]*>'," ",data)     # Remove full form tag with its content
    data = re.sub(r'(?i)<[ ]*?sup.*?/sup[ ]*>'," ",data)       # Remove citations
    
    #data = re.sub(r'<.*?!DOCTYPE.+?html.*?>',"\n",data)        # Remove Doctype
    #data = re.sub(r'(?i)<[/]?[ibp]>',"",data)                 # Remove p,i,b tags but not content
    #data = re.sub(r'(?i)<[/]?html.*?>',"",data)               # Remove html tags
    #data = re.sub(r'(?i)<[/]?body.*?>',"",data)               # Remove body tags
    #data = re.sub(r'(?i)<[/]?div.*?>',"",data)                # Remove div tags
    #data = re.sub(r'(?i)<!--.*?-->',"",data)                  # Remove comments
    #data = re.sub(r'(?i)<[/]?a.*?>',"",data)                  # Remove links (but not their text)

    data = re.sub(r'(?i)<[/]?br>',"--br--",data)               # Spare br rags removal
    data = re.sub(r'<[^>]*>'," ",data)                         # Remove all remaining tags but not their conent
    data = re.sub(r'--br--',"\n",data)                         # Change <br> to \n

    return data


### This function detects if the input is a A2, A3 or code country number
### Returns a (A3,country name) tuple
def convertCountry(code) :

    ### If it is a number
    if str(code).isdigit() :
        for row in country_df.iterrows() :
            if int(row[1]["Code"]) == int(code) :
                return (row[1]["A3"],row[1]["Name"])

    if isinstance(code,str) :
        code = code.upper().replace("\\s+"," ").lstrip().rstrip()
    else :
        print "The input must be a number or a string"
        return (None,None)

    ### If it is a 2 char string. Assume it is a A2 code.
    if len(code)==2 :
        for row in country_df.iterrows() :
            if row[1]["A2"] == code :
                return (row[1]["A3"],row[1]["Name"])        

    ### If it is a 3 char string. Assume it is a A3 code.
    elif len(code)==3 :
        print "It's a A3"
        for row in country_df.iterrows() :
            if row[1]["A3"] == code :
                return (row[1]["A3"],row[1]["Name"])      

    ### If it is a > 3 char string. Assume it is the full name.
    elif len(code)>3 :
        for row in country_df.iterrows() :
            if code == row[1]["Name"] :
                return (row[1]["A3"],row[1]["Name"])    

    else :
        print "I'm sorry I didn't find the country", code+"."
        print "Only A2, A3 or numeric ISO 3166 codes are usable."
        print "You can try with the full name too but you have to type it"
        print "as it is in the DB. So be careful."

    return (None,None)








