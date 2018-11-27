from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from engineutils import country_df, countryCode, resroot, saveDataWithPrediction
from twitterutils import getTwitterCounts, getTwitterFollowers
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import os, re, pickle
import seaborn as sb
import pandas as pd
import numpy as np

modelXGfile         = resroot+"XG_famous_model.pkl"
modelVotingfile     = resroot+"voting_famous_model.pkl"
fulldffile          = resroot+"FullDF.pkl"
trained_XGmodel     = pickle.load(open(modelXGfile))
trained_Votingmodel = pickle.load(open(modelVotingfile))


### Trains a XGboost model to classify famous
def trainFamousModel(features,labels) :

    ### Do some hyper-parameter scanning to oprimise the performance
    param_cands_XG = [
        { 'learning_rate': [0.2,0.5,1.0,1.2,1.5], 'gamma': [0.,0.3,0.4,0.5,1.0], 'max_depth': [2,3,4,5,6,7,8] }
        ]

    modelXG = GridSearchCV(estimator=XGBClassifier(objective="reg:logistic"), param_grid=param_cands_XG,cv=5)

    ## Just test with no gridsearch
    #modelXG = XGBClassifier(learning_rate=0.2,gamma=0.4,max_depth=4)
    
    modelXG.fit(features,labels)

    bestmodel = modelXG.best_estimator_
    scores    = cross_val_score(bestmodel, features, labels, cv=5, scoring='accuracy')

    print "Best parameters: "
    print modelXG.best_params_
    print "XGBoost score      : {:.2f}%".format(np.mean(scores)*100)

    #with open(modelXGfile,"w") as of :
    #    pickle.dump(modelXG.best_estimator_,of)
    saveDataWithPrediction("XGScored",features,modelXG.best_estimator_,labels,"isFamous")

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

    print "AdaBoost score     : {:.2f}%".format(np.mean(scoresAda)*100)
    print "XGBoost score      : {:.2f}%".format(np.mean(scoresXG)*100)
    print "Random forest score: {:.2f}%".format(np.mean(scoresRF)*100)

    clfs = [('Ada',modelAda),('XG',modelXG),('RF',modelRF)]
    model  = VotingClassifier(estimators=clfs, voting='soft', n_jobs=4)
    model.fit(features,labels)
    ### "soft" option weights the vote for the accuracy of each
    
    getClfsCorr(clfs,features)
    scores = cross_val_score(model, features, labels, cv=5, scoring='accuracy')

    with open(modelVotingfile,"w") as of :
        pickle.dump(modelXG,of)
    saveDataWithPrediction("votingScored",features,model,labels,"isFamous")
    print "Voting model score : {:.2f}%".format(np.mean(scores)*100)
    return model


### Utility to get correlations bertween the scores of various classifiers
### The input much be a list of tuples ('clf name',clf)
def getClfsCorr(clfs,test) :
        
    series = []
    for nameclf,clf in clfs :
        series.append( pd.Series(clf.predict(test), name=nameclf) )

    ensemble_results = pd.concat(series,axis=1)

    g = sb.heatmap(ensemble_results.corr(),annot=True)
    plt.savefig("CLFsCorr.pdf")
    plt.cla()


def isFamous(features) :

    pred = trained_XGmodel.predict(features)
    return pred[0] > 0.5

def isFamousVoting(features) :

    pred = trained_Votingmodel.predict(features)
    return pred.values[0] > 0.5

## Define One Hot Encoder for countries
## Hadles both int and string labelled categories 
class CountryOneHotEncoder :

    def __init__(self,df,colname) :

        self.colname = colname
        self.ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')
        self.le  = LabelEncoder()

        lab_encoded = self.le.fit_transform(df[colname])
        self.ohe.fit(lab_encoded.reshape(len(lab_encoded), 1))

        self.colnames = [colname+"_"+str(cl) for cl in self.le.classes_]

    def encodeOneHot(self,df,colname=None) :

        if colname is None : colname = self.colname
        lab_encoded = self.le.transform(df[colname])
        oh_encoded  = self.ohe.transform(lab_encoded.reshape(len(lab_encoded), 1))

        return pd.DataFrame(oh_encoded, columns=self.colnames)

country_ohe = CountryOneHotEncoder(country_df,'Code')


### Parameters are already available quantities, so they won't be recalculated

def getFamousFeatures(name,surname,isPolitician=None,country=None,money=None,job=None) :

    fdict = {
        'scorePolSimple': isPolitician,
        'TweetCounts'   : getTwitterCounts(name,surname),
        'TweetFollow'   : getTwitterFollowers(name,surname),
        'country'       : country,
        'money'         : money
    }
    print fdict

    ### Take care of doing necessary conversions
    if country is not None :
        fdict['country'] = countryCode(country)

    ### Recalculate quantities if they are missing
    if fdict['scorePolSimple'] is None :
        googleout = parseGoogle(name,surname)
        fdict['scorePolSimple'] = isPoliticianSimple(googleout)
    
    if fdict['country'] is None or fdict['money'] is None :

        info, fulltext = parseWiki(name,surname)
        if country is None :
            fdict['country'] = countryCode(info['country'])
        if money is None :
            fdict['money'] = info["money"]

    ### Process data so that the model can read it
    df = pd.DataFrame(fdict,index=[0])

    ## Convert country into One Hot Encoding
    oh_encoded = country_ohe.encodeOneHot(df,'country')
    df = pd.concat([df, oh_encoded], axis=1)
    df.drop(columns=['country'])

    with open(resroot+"NameFeatures.pkl") as of:
        df = df[pickle.load(of)]

    return df


if __name__ == "__main__" :

    print "It will use data from ", resroot+"WikiDF.pkl"
    if not os.path.exists(resroot+"WikiDF.pkl") :
        print "Please run 'python engine/wikiutils.py' to get data for training"
        sys.exit()
    if not os.path.exists(resroot+"TwitterDF.pkl") :
        print "Please run 'python engine/wikiutils.py' to get data for training"
        sys.exit()

    wikidata  = pickle.load(open(resroot+"WikiDF.pkl"))
    poldata   = pickle.load(open(resroot+"NLP_simple_out.pkl"))
    tweetdata = pickle.load(open(resroot+"TwitterDF.pkl"))

    common = ['name','surname','isPol','isFam']
    mydata = pd.merge(wikidata, tweetdata, left_on=common, right_on=common, how='inner')
    mydata = pd.merge(mydata, poldata, left_on=common, right_on=common, how='inner')
    mydata['country'] = mydata['country'].apply(lambda x : countryCode(x))
    mydata['money'] = mydata['money'].apply( lambda x : -1 if isinstance(x,str) else x )
    mydata['scorePolSimple'] = mydata['scorePolSimple'].fillna(0).replace(np.inf, 0)
    pickle.dump(mydata,open(fulldffile,"w"))
    
    features = mydata[['scorePolSimple','TweetCounts','TweetFollow','country','money']]
    oh_encoded = country_ohe.encodeOneHot(features,'country')
    features = pd.concat([features, oh_encoded], axis=1)
    features.drop(columns=['country'])

    labels   = mydata['isFam']

    with open(resroot+"NameFeatures.pkl","w") as of:
        pickle.dump(features.columns,of)
    trainFamousModel(features,labels)
    trainFamousVotingModel(features,labels)






