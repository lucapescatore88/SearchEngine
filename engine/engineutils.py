import re, os, sys, yaml
import pandas as pd
import pickle

### Generic paths

root    = os.getenv("PICTETROOT")
resroot = root+"/resources/"

### Open config file and make it available via imports
config = {}
try :
    f = open(root+"/cfg.yml")
    config = yaml.load(f)
    print "Configuration:",config
except Exception as e: 
    print "Config file is not good"
    print(e)
    sys.exit()

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)

### Function to load the DB with currency codes and add symbols

def loadCurrencies() :

    from babel.numbers import format_currency

    currency_df = pd.read_csv(resroot+"currencies.csv")
    symbols = []
    
    for ir,row in currency_df.iterrows():
        #print row[1]['Alphabetic Code'],
        symbol = row['Alphabetic Code']
        try : 
            s = format_currency(1, row['Alphabetic Code'],locale='en_US')
            symbol = s.replace("1.00","")
        except : pass
        symbols.append(symbol)

    currency_df['Symbol'] = pd.Series(symbols)
    return currency_df


### Function to load the DB with country codes

def loadCountries() :

    df = pd.read_csv(resroot+"countries.csv")
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

    if code == "" : return (None,code)

    ### If it is a number
    if str(code).isdigit() :
        for ir,row in country_df.iterrows() :
            if int(row["Code"]) == int(code) :
                return (row["A3"],row["Name"])

    #print code, type(code)
    if isinstance(code,str) :
        code = code.upper().replace("\\s+"," ").lstrip().rstrip()
    else :
        print "The input must be a number or a string"
        return (None,None)

    ### If it is a 2 char string. Assume it is a A2 code.
    if len(code)==2 :
        for ir,row in country_df.iterrows() :
            if row["A2"] == code :
                return (row["A3"],row["Name"])   

    ### If it is a 3 char string. Assume it is a A3 code.
    if len(code)==3 :
        print "It's a A3"
        for ir,row in country_df.iterrows() :
            if row["A3"] == code :
                return (row["A3"],row["Name"])   

    ### If it is a > 3 char string. Assume it is the full name.
    if len(code)>3 :
        for ir,row in country_df.iterrows() :
            if code == row["Name"] :
                return (row["A3"],row["Name"])

        ## Check if a nationality was given, otherwise assume country name
        for ir,row in country_df.iterrows() :
            if code == row["Nationality"] :
                return (row["A3"],row["Name"])

    else :
        print "I'm sorry I didn't find the country", code+"."
        print "Only A2, A3 or numeric ISO 3166 codes are usable."
        print "You can try with the full name too but you have to type it"
        print "as it is in the DB. So be careful."

    return (None,None)

def countryCode(text) :

    A3, name = convertCountry(text)
    if A3 is not None :
        countryinfo = country_df.loc[country_df['A3']==A3]
        return countryinfo['Code'].values[0]
    return -1


### Function to run over a list of people and apply a function, supports backup
def getPeopleData(hname,peoplefile,myfunction=None,nobackup=False) :

    data = pd.read_csv(peoplefile)

    ### To avoid running searches all the time can use backup
    backupfile = resroot+"/"+hname+"_backup.pkl"
    backup   = {}

    if os.path.exists(backupfile) and not nobackup:
        backup = pickle.load(open(backupfile))
        print "Loaded from saved data"
        print backup.keys()
    
    for ir,row in data.iterrows() :
        name         = row["name"]
        surname      = row["surname"]
        isPolitician = row["politician"]
        isFamous     = row["famous"]

        try :
            key = (name,surname,isPolitician,isFamous)
            if key in backup.keys() :
                print hname+": Info for", name, surname, "already in store, use --nobackup to avoid using it"
                curfeatures = backup[key]
            else :
                print hname+": Getting info for", name, surname
                curfeatures = myfunction(name,surname)
            
            #print curfeatures
            backup[key] = curfeatures
            if len(backup) > 0 : pickle.dump(backup,open(backupfile,"w"))
            
        except :
            continue
    
    return backup

from argparse import ArgumentParser
trainparser = ArgumentParser()
trainparser.add_argument("--trainfile",default=resroot+"people.csv",
    help="The name of the csv file with names of politicians and not")
trainparser.add_argument("--nobackup",action="store_true",
    help="If present it will try look for a backup file and use it if exists")

