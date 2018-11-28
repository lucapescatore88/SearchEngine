from engineutils import res
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pickle
import sys


file = res.RESOURCES+"NLP_simple_out.pkl"
file = res.RESOURCES+"NLP_out.pkl"
var = 'scorePolSimple'
var = 'scorePol'

data = pickle.load(open(file))
print data.head()

sb.set()
data = data.fillna(0)
dataPol   = data.loc[data['isPol']==1,var]
dataNoPol = data.loc[data['isPol']==0,var]

sb.distplot(dataPol,kde=False,bins=30)
sb.distplot(dataNoPol,kde=False,bins=30)
plt.legend(['Normals (mean = %.2f)' % dataNoPol.mean(),
    'Politicians (mean = %.2f)' % dataPol.mean()], ncol=2, loc='best');
plt.yscale('log')
plt.savefig(res.PLOTS+"NLPout_seaborn.pdf")
plt.clf()

plt.hist(dataPol, normed=True, alpha=0.5)
plt.hist(dataNoPol, normed=True, alpha=0.5)
plt.savefig(res.PLOTS+"NLPout.pdf")
plt.clf()

eff, rej = [], []
cuts = np.linspace(data[[var]].min(),data[[var]].max(),100)
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
plt.savefig(res.PLOTS+"NLP_ROC.pdf")
plt.clf()




