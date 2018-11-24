from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid, GridSearchCV
from xgboost import XGBClassifier
import maplotlib.pyplot as plt
import seaborn as sb
import numpy as np

resroot         = root+"/resources/"
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
    modelRF.fit(f_scaled,l)


    ### Get an idea of how each one score singularly
    scoresAda = cross_val_score(modelAda, features, labels, cv=10, scoring='accuracy')
    scoresXG  = cross_val_score(modelXG,  features, labels, cv=10, scoring='accuracy')
    scoresRF  = cross_val_score(modelRF,  features, labels, cv=10, scoring='accuracy')

    print "AdaBoost score: ", np.mean(scoresAda)
    print "XGBoost score: ", np.mean(scoresXG)
    print "Random forest score: "np.mean(scoresRF)

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

    return int(trained_XGmodel(features) > 0.5)

def isFamousVoting(features) :

    return int(trained_Votingmodel(features) > 0.5)




