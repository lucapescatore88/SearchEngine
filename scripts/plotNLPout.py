from engineutils import root, resroot
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pickle
import sys

plots = root+"/plots/"

data = pickle.load(open(resroot+"NLP_simple_out.pkl"))

#print data.head()
data = data.fillna(0)
dataPol   = data.loc[data['isPol']==1,'scorePolSimple']
dataNoPol = data.loc[data['isPol']==0,'scorePolSimple']

sb.distplot(dataPol,kde=False)
sb.distplot(dataNoPol,kde=False)
plt.legend(['Politicians (mean = %.2f)' % dataPol.mean(),'Normals (mean = %.2f)' % dataNoPol.mean()], ncol=2, loc='best');
plt.savefig(plots+"NLPout_seaborn.pdf")
plt.clf()

plt.hist(data.loc[data['isPol']==1,'scorePolSimple'], normed=True, alpha=0.5)
plt.hist(data.loc[data['isPol']==0,'scorePolSimple'], normed=True, alpha=0.5)
plt.savefig(plots+"NLPout.pdf")
plt.clf()

eff = []
rej = []
cuts = np.linspace(data[['scorePolSimple']].min(),data[['scorePolSimple']].max(),100)
totPol = float(len( dataPol.values ))
totNoPol = float(len( dataNoPol.values ))
mindist = 100
bestcut = -1
besteff = -1
for c in cuts :
    eff.append( len( dataPol.loc[data['scorePolSimple']>c].values ) / totPol )
    rej.append( len( dataNoPol.loc[data['scorePolSimple']<c].values ) / totNoPol )
    dist = (1 -eff[-1])**2 +(1-rej[-1])**2
    if dist < mindist : 
        mindist = dist
        bestcut = c
        besteff = eff[-1]

print "The best cut is:", bestcut, "with efficiency", besteff
plt.plot(eff,rej)
plt.xlabel('Efficiency')
plt.ylabel('Rejection')
plt.savefig(plots+"NLP_ROC.pdf")
plt.clf()




