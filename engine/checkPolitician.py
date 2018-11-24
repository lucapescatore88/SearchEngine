from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from engineutils import config, resroot
from googleutils import parseGoogle
from nltk.corpus import stopwords
import string, math, os, sys
import pandas as pd
import yaml, pickle
import numpy as np

modelfile = resroot+"NLP_polititian_model.pkl"
mapfile   = resroot+"NLP_polititian_wordmap.pkl"

trained_model    = pickle.load(open(modelfile))
trained_word_map = pickle.load(open(mapfile))

### Splits words, simplifies them (lower case, remove stopwords, lemmatize)
def prepareNLPData(data) :

    punct = list(string.punctuation)    ## Punctuation to remove
    stopw = stopwords.words('english')  ## Stopwords to remove

    ## Divide words, and remove punctuation and stopwords
    tokens = word_tokenize(data)
    filtered_words = [w.lower() for w in tokens if w.lower() not in stopw]
    filtered_words = [w for w in filtered_words if w not in punct]

    ## Lemmatise words to limit phasespace: e.g dogs --> dog, doing --> do 
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatised = [wordnet_lemmatizer.lemmatize(t) for t in filtered_words]

    return lemmatised


### Creates a dictionary mapping words to indices
def getWordMap(toks,word_map = {}) :

    ### In case we need to add to a previous dictionary
    cur_index = len(word_map)

    ### Build vectors of words
    for tok in toks :
        if tok not in word_map :
            word_map[tok] = cur_index
            cur_index+=1

    return word_map


### Given lemmatised tokens and a wordmap created a vector
def tokensToVector(tokens, word_map, label = None) :
    
    dim = len(word_map)
    if label is not None : dim += 1

    x = np.zeros(dim)
    for t in tokens :
        x[word_map[t]] += 1
    x = x / x.sum()             ## Normalise to get word frequency

    if label is not None : x[-1] = label
    return x


### Given a list texts applies lemmatisation and creates the map
def makeDataSet(people,word_map={}) :

    ## Build map from words to vector index
    toks = []
    for pol in people :
        lemms = prepareNLPData(pol)
        word_map = getWordMap(lemms,word_map)
        toks.append(lemms)
    
    return word_map, toks


### Function to train a NLP model for classifying politicians 
def trainNLPModel(politicians,normals) :

    word_map, pol_toks = makeDataSet(politicians,word_map_simple)
    word_map, norm_toks = makeDataSet(normals,word_map)
    
    N = len(pol_toks) + len(norm_toks)
    data = np.zeros((N,len(word_map)+1))
    i = 0
    for toks in pol_toks :
        data[i,:] = tokensToVector(toks, word_map, label = 1)
        i +=1
    for toks in norm_toks :
        data[i,:] = tokensToVector(toks, word_map, label = 0)
        i +=1

    ### Shuffle to avoid using only politicians or only normals for training
    np.random.shuffle(data)

    features = data[:,:-1]
    labels   = data[:,-1]

    ntest = int(-0.2 * len(data))
    features_train = features[:ntest,]
    labels_train = labels[:ntest,]
    features_test  = features[ntest:,]
    labels_test  = labels[ntest:,]

    model = LogisticRegression()
    model.fit(features_train,labels_train)

    pickle.dump(model,open(modelfile,"w"))
    pickle.dump(word_map,open(mapfile,"w"))
    print "Classification rate", model.score(features_test,labels_test)
    #model.score(features_train,labels_train)

    return model


### Given a text return if it is a politician 
### N.B.: Uses a pretrained model
### N.B.: Threshold can be changed in cfg.yml
def isPolitician(person) :

    word_map, alltoks = makeDataSet([person],trained_word_map)
    data = np.zeros((len(toks),len(word_map)))
    i = 0
    for toks in alltoks :
        data[i,:] = tokensToVector(toks, word_map, label = None)
        i +=1
    
    #Code to check the weight given to each word
    #for w,i in word_map.iteritems() :
    #    weight = model.coef_[0][i]
    #    if(abs(weight)>thr) :
    #        print word, weight
    return int(trained_model.predict(data) > config['nlp_thr'])


### Calculates cosigne between to vectors as a similarity measure (could use np.dot)
def cosvec(v1,v2) :
    
    if len(v1) != len(v2) :
        return "len(v1) = "+str(len(v1))+" len(v2) = "+str(len(v2))+" They must be equal!"
        return 0

    xy, xx, yy = 0, 0, 0
    for i in range(len(v1)) :
        xx += v1[i]*v1[i]
        yy += v2[i]*v2[i]
        xy += v1[i]*v2[i]
        
    if xx == 0 or yy == 0 :
        print "One of the two vectors is empty!!"
        return 0

    return float(xy) / (math.sqrt(xx)*math.sqrt(yy))


### Get the simplified map for these few political words to test
def getTestMap() :

    politics_words = ["politics","election","decision","minister","senator","president",
                        "parliament","congress","vote","nation","party","leader","idea",
                        "democratic","war","welfare","internaitonal","constitution",
                        "institution","state","head","military","governament",
                        "tyranny","federal","global","corruption","power"]

    lemms             = prepareNLPData(' '.join(politics_words))
    word_map_simple   = getWordMap(lemms)
    #print lemms
    #print word_map_simple

    return lemms, word_map_simple

### Make the test objects globally available
lemms_simple, word_map_simple = getTestMap()


### Returns average score on a list of texts
def scoreSimpleNLP(people) :

    word_map, alltoks = makeDataSet(people,word_map_simple)
    testvector = tokensToVector(lemms_simple, word_map, label = None) 
    
    vecs   = [ tokensToVector(toks, word_map, label = None) for toks in alltoks ]
    scores = [ cosvec(testvector,vec) for vec in vecs ]

    return np.mean(scores)


### Returns true if the score of the simple model passes a threshold
### N.B.: Threshold was pretrained and can be changed in cfg.yml
def isPoliticianSimple(person) :
    return int( scoreSimpleNLP([person]) > config['simple_nlp_thr'])


### Function to train the simplified model
def trainSimpleNLPModel(politicians,normals) :

    word_map, pol_toks = makeDataSet(politicians,word_map_simple)
    word_map, norm_toks = makeDataSet(normals,word_map)
    
    testvector = tokensToVector(lemms_simple, word_map, label = None) 
    pol_vecs   = [ tokensToVector(toks, word_map, label = None) for toks in pol_toks ]
    norm_vecs  = [ tokensToVector(toks, word_map, label = None) for toks in norm_toks ]

    pol_scores  = [ cosvec(testvector,vec) for vec in pol_vecs ]
    norm_scores = [ cosvec(testvector,vec) for vec in norm_vecs ]
        
    ## The treshold will be the middle point between the two averages
    thr = (np.mean(pol_scores) + np.mean(norm_scores))/2.
    print "Mean Politics = ", np.mean(pol_scores)
    print "Mean Normal   = ", np.mean(norm_scores)
    print "Threshold     = ", thr
    return thr



if __name__ == "__main__" :

    import warnings
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    from googleutils import parseGoogle
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--trainfile",default=resroot+"people.csv",
        help="The name of the csv file with names of politicians and not")
    parser.add_argument("--data",default=None,
        help="Pickle file where data is saved")
    parser.add_argument("--simple", action="store_true",
        help="Will train the simple model instead of the LogisticRegression that is trained by default")
    args = parser.parse_args()

    data = pd.read_csv(args.trainfile)
    politicians = []
    normals     = []
    
    backup = {}

    ### To avoid running searches all the time can use backup
    if args.data is not None and os.path.exists(args.data) :
        backup = pickle.load(open(args.data))
        print "Loaded from saved data"
        print backup.keys()
    
    for ir,row in data.iterrows() :
        name         = row["name"]
        surname      = row["surname"]
        isPolititian = row["polititian"]

        ### Get data, from the net or from backup
        try :
            if (name,surname) in backup :
                out = backup[(name,surname)]
            else :
                out = parseGoogle(name,surname)
        
            if isPolititian == 1 : politicians.append(out)
            else : normals.append(out)

            if out != "" : backup[(name,surname)] = out
            if len(backup) > 0 : pickle.dump(backup,open("backup.pkl","w"))
            
        except :
            continue

    if args.simple : trainSimpleNLPModel(politicians,normals)
    else : trainNLPModel(politicians,normals)


