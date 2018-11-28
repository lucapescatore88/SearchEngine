from engineutils import resroot
import pandas as pd
import pickle

modelfile  = resroot+"NLP_politician_model.pkl"
mapfile    = resroot+"NLP_politician_wordmap.pkl"

trained_model    = pickle.load(open(modelfile))
trained_word_map = pickle.load(open(mapfile))

#Function to check the weight given to each word
def checkWeights(word_map, model):

    recs = []
    for w,i in word_map.iteritems() :
        recs.append( {"word": w, "coeff" : model.coef_[0][i] })
    df = pd.DataFrame(recs)
    print "Most politics related words"
    print df.sort_values('coeff', ascending=False).head(30)
    print "Least politics related words"
    print df.sort_values('coeff').head(30)

checkWeights(trained_word_map,trained_model)
