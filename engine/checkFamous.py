from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid, GridSearchCV
from twitterutils import getTwitterCounts, getTwitterFollowers
from engineutils import country_df, countryCode, resroot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from googleutils import parseGoogle
from xgboost import XGBClassifier
from wikiutils import parseWiki
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os, re

modelXGfile     = resroot+"XG_famous_model.pkl"
modelVotingfile = resroot+"voting_famous_model.pkl"

#trained_XGmodel     = pickle.load(open(modelXGfile))
#trained_Votingmodel = pickle.load(open(modelVotingfile))

### Trains a XGboost model to classify famous
def trainFamousModel(features,labels) :

    ### Do some hyper-parameter scanning to oprimise the performance
    param_cands_XG = [
        { 'learning_rate': [0.2,0.5,1.0,1.2,1.5], 'gamma': [0.,0.3,0.4,0.5,1.0], 'max_depth': [2,3,4,5,6,7,8] }
        #{ 'learning_rate': [0.5], 'gamma': [.0], 'max_depth': [4] }
        ]

    modelXG = GridSearchCV(estimator=XGBClassifier(), param_grid=param_cands_XG,cv=5)

    ## Just test with no gridsearch
    #modelXG = XGBClassifier(learning_rate=0.2,gamma=0.4,max_depth=4)
    #modelXG.fit(features,labels)

    bestmodel = modelXG.best_estimator_
    scores    = cross_val_score(bestmodel, features, labels, cv=5, scoring='accuracy')

    pickle.dump(model,open(modelXGfile,"w"))
    print "XGBoost model score : ", np.mean(scores)
    return bestmodel


### Trains a voting model with 4 models inside to classify famous
def trainFamousVotingModel(features,labels) :

    ### Train 3 models. N.B.: To do it properly should do a GridScan for each one but no time

    modelAda = AdaBoostClassifier(n_estimators=100)
    modelAda.fit(features,labels)

    modelXG = XGBClassifier(learning_rate=0.2,gamma=0.4,max_depth=4)
    modelXG.fit(features,labels)

    modelRF = RandomForestClassifier(n_estimators=100, max_depth=4)
    modelRF.fit(features,labels)


    ### Get an idea of how each one score singularly
    scoresAda = cross_val_score(modelAda, features, labels, cv=10, scoring='accuracy')
    scoresXG  = cross_val_score(modelXG,  features, labels, cv=10, scoring='accuracy')
    scoresRF  = cross_val_score(modelRF,  features, labels, cv=10, scoring='accuracy')

    print "AdaBoost score: ", np.mean(scoresAda)
    print "XGBoost score: ", np.mean(scoresXG)
    print "Random forest score: ", np.mean(scoresRF)

    series = [('Ada',modelAda),('XG',scoresXG),('RF',scoresRF)]
    model  = VotingClassifier(estimators=series, voting='soft', n_jobs=4)
    ### "soft" option weights the vote for the accuracy of each
    
    scores    = cross_val_score(model, features, labels, cv=5, scoring='accuracy')

    pickle.dump(model,open(modelVotingfile,"w"))
    print "Voting model score : ", np.mean(scores)
    return model


### Utility to get correlations bertween the scores of various classifiers
### The input much be a list of tuples ('clf name',clf)
def getClfsCorr(clfs,test) :
        
    series = []
    for nameclf,clf in clfs.iteritems() :
        series.append( pd.Series(clf.predict(test), name=nameclf) )

    ensemble_results = pd.concat(series,axis=1)

    g = sb.heatmap(ensemble_results.corr(),annot=True)
    plt.savefig("CLFsCorr.pdf")
    plt.cla()


def isFamous(features) :

    #return int(trained_XGmodel(features) > 0.5)
    return True

def isFamousVoting(features) :

    #return int(trained_Votingmodel(features) > 0.5)
    return True


## Define One Hot Encoder for countries
country_ohe = OneHotEncoder(handle_unknown='ignore')
country_le = LabelEncoder()

unknown = pd.DataFrame(dict({'Code':-1}),index=[0])
country_df['Code'] = country_df['Code'].append(unknown,ignore_index=True)
country_lab_encoded = country_le.fit_transform(country_df['Code'])
country_ohe.fit(country_lab_encoded.reshape(len(country_lab_encoded), 1))

def oneHotCountryCode(code) :

    codedf = pd.DataFrame(dict({'Code':code}), index=[0])
    label_encoded = country_le.transform(codedf)
    return country_ohe.transform(label_encoded.reshape(len(label_encoded), 1))


### Parameters are already available quantities, so they won't be recalculated

columns=['isPolitician','twitterCounts','country','money','nTwitFollowers']

def getFamousFeatures(name,surname,isPolitician=None,country=None,money=None,job=None) :

    fdict = {
        'isPolitician'  : isPolitician,
        'twitterCounts' : getTwitterCounts(name,surname),
        'country'       : country,
        #'job'           : None,
        'money'         : money,
        "nTwitFollowers": getTwitterFollowers(name,surname)
    }

    ### Take care of doing necessary conversions
    if country is not None :
        fdict['country'] = countryCode(country)

    ### Recalculate quantities if they are missing
    if fdict['isPolitician'] is None :
        googleout = parseGoogle(name,surname)
        fdict['isPolitician'] = isPoliticianSimple(googleout+fulltext)
    
    if fdict['country'] is None or fdict['money'] is None :

        info, fulltext = parseWiki(name,surname)
        if country is None :
            fdict['country'] = countryCode(info['country'])
        #if job is None :
        #    firstjob = info['profession'].split(",")[0]
        #   fdict['job'] = firstjob
        if money is None :
            fdict['money'] = info["money"]

    
    ### Process data so that the model can read it
    df = pd.DataFrame(fdict,index=[0])

    ## Convert country into One Hot Encoding
    #print "Encoding"
    encoded = oneHotCountryCode(df['country'])
    df = pd.concat([df, encoded], axis=1)
    #df.drop('country')

    return df


if __name__ == "__main__" :

    import warnings
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    from googleutils import parseGoogle
    from argparse import ArgumentParser
    import pickle

    parser = ArgumentParser()
    parser.add_argument("--trainfile",default=resroot+"people.csv",
        help="The name of the csv file with names of politicians and not")
    parser.add_argument("--data",default=None,
        help="Pickle file where data is saved")
    args = parser.parse_args()

    data = pd.read_csv(args.trainfile)
    features  = pd.DataFrame(columns=columns)
    labels    = pd.DataFrame(columns=['Famous'])
    
    backup = {}

    ### To avoid running searches all the time can use backup
    if args.data is not None and os.path.exists(args.data) :
        backup = pickle.load(open(args.data))
        print "Loaded from saved data"
        print backup.keys()
    
    for ir,row in data.iterrows() :
        name         = row["name"]
        surname      = row["surname"]
        isPolitician = row["politician"]
        isFamous     = row["famous"]

        try :
            print "Getting info for", name, surname
            key = (name,surname,isPolitician,isFamous)
            if key in backup.keys() :
                curfeatures = backup[key]
            else :
                curfeatures = getFamousFeatures(name,surname)
            
            features.append(curfeatures, ignore_index=True)
            labels.append(pd.DataFrame(dict({"Famous":int(isFamous)}), index=[0]), ignore_index=True)

            print "Done! Saving info"
            backup[key] = curfeatures
            if len(backup) > 0 : pickle.dump(backup,open("backup_famous.pkl","w"))
        
        except :
            continue
    
    #if len(backup) > 0 : pickle.dump(backup,open("backup_famous.pkl","w"))
    #print features.head()
    #print labels.head()
    #sys.exit()

    trainFamousModel(features,labels)
    trainFamousVotingModel(features,labels)

