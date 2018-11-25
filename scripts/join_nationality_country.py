import pandas as pd
import re

countryinfo = pd.read_csv("countries_orig.csv")
natinfo = pd.read_csv("nationalities.csv")

nationalities = []
for ic,crow in countryinfo.iterrows() :

    cname = re.sub("\\s+"," ",crow["Name"]).lstrip().rstrip().upper()

    found = False 
    for ni,nrow in natinfo.iterrows() :
        nname = re.sub("\\s+"," ",nrow["Country"]).lstrip().rstrip().upper()
        
        if cname == nname :
            nat = re.sub("\\s+"," ",nrow["Demonym"]).lstrip().rstrip().upper()
            nationalities.append(nat)
            found = True
            break

    if not found : nationalities.append(cname)

countryinfo["Nationality"] = nationalities
countryinfo.to_csv("countries.csv",header=True)

