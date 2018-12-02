from engineutils import res
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import argparse
import pickle
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--simple",action="store_true")
args = parser.parse_args()

simple = ""
file = res.RESOURCES+"NLP_out.pkl"
var = 'scorePol'
if args.simple :
    simple = "simple"
    file = res.RESOURCES+"NLP_simple_out.pkl"
    var = 'scorePolSimple'

data = pickle.load(open(file))
#print data.head()

scale = (data[[var]].max() - data[[var]].min())
shift = data[[var]].min()
print "Scale and shift", scale, shift
data[var] = data[var].apply( lambda x: (x-shift)/scale )

sb.set()
data = data.fillna(0)
dataPol   = data.loc[data['isPol']==1,var]
dataNoPol = data.loc[data['isPol']==0,var]

sb.distplot(dataNoPol,kde=False,bins=30)
sb.distplot(dataPol,kde=False,bins=30)
plt.legend(['Normals (mean = %.2f)' % dataNoPol.mean(),
    'Politicians (mean = %.2f)' % dataPol.mean()], loc='best');
plt.yscale('log')
plt.savefig(res.PLOTS+"NLPout"+simple+".pdf")
plt.clf()

eff, rej = [], []
cuts = np.linspace(0,1,100)
totPol = float(len( dataPol.values ))
totNoPol = float(len( dataNoPol.values ))
mindist = 100
bestcut, besteff = -1, -1
for c in cuts :
    eff.append( len( dataPol.loc[dataPol>c].values ) / totPol )
    rej.append( len( dataNoPol.loc[dataNoPol<c].values ) / totNoPol )
    dist = (1 -eff[-1])**2 +(1-rej[-1])**2
    if dist < mindist : 
        mindist = dist
        bestcut = c
        besteff = eff[-1]

print "The best cut is:", bestcut, "with efficiency", besteff
plt.plot(eff,rej)
plt.xlabel('Efficiency')
plt.ylabel('Rejection')
plt.savefig(res.PLOTS+"NLP_ROC"+simple+".pdf")
plt.clf()




