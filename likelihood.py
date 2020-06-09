### likelihood ratio test
import math
import numpy as np
from scipy.stats import logistic
from scipy.optimize import minimize
from scipy.stats import chi2

# 參數 p q r
# H1: general model: 變動 p q , r=(1-q-p)
# H0: restrict model: 分別固定 p q r, 另兩個參數相等 
    # e.g. test q,r : (p = behaviorMean[i][col] q,r = (1-p)/2)

def negloglikelihood(prob, modeltype = "typical", data = [], parameterChoose = None):
#     scenario = 0   # 0: more / 3: same / 6: less
#     parameterChoose = 0   # 0: buy / 1: no trade / 2: sell
##      -->(0,1):compare whether p and q is equal
##       --> gengeral model: vary q r to decide p  (???)
##       --> restrict model: vary r to decide p and q
    data_=np.array(data)
    print(prob)
    prob_ = np.zeros(3)
    ### general model unfinished(but basically every general models are the same)
    if modeltype == "general":
        decidedP = 1-prob[0]-prob[1]
        print(decidedP)
        if parameterChoose == (0,1):
            prob = np.insert(prob,2,0)
        elif parameterChoose == (1,2):
            prob=np.insert(prob,0,0)
        elif parameterChoose == (0,2):
            prob=np.insert(prob,1,0)
        else:
            print("parameterChoose set error!")

        for i in range(0,3):
            if i in parameterChoose:
                prob_[i] = prob[i]
            else:
                prob_[i] = decidedP
    elif modeltype == "restrict":
        decidedP = (1-prob)/2
        print(decidedP)
        for i in range(0,3):
            if i in parameterChoose:
                prob_[i]=decidedP
            else:
                prob_[i]=prob
    elif modeltype == "typical":
        prob_ = prob
    else:
        print("modeltype not support!")
    print(prob_)
    negloglikelihood = -(data_[0:3] * np.log(prob_)).sum()
    
    return negloglikelihood

def likelihood_ratio_test(clusterBehaviorMean, clusterBehaviorCount, cluster=1, scenario=0, parameterChoose=(0)):
    gen_probability = [clusterData[cluster][scenario+parameterChoose[0]],clusterData[cluster][scenario+parameterChoose[1]]] # initial guess
    if parameterChoose == (0,1):
        rst_probability = behaviorMean[cluster][scenario+2]
    elif parameterChoose == (0,2):
        rst_probability = behaviorMean[cluster][scenario+1]
    elif parameterChoose == (1,2):
        rst_probability = behaviorMean[cluster][scenario+0]
    else:
        print("parameterChoose out of range")
    countN = [clusterBehaviorCount[cluster][scenario],clusterBehaviorCount[cluster][scenario+1],clusterBehaviorCount[cluster][scenario+2]]

    gen_parameter = minimize(fun = negloglikelihood, x0 = gen_probability, args = ("general",countN,parameterChoose), method='TNC',bounds=((0,1),(0,1)))
    print(gen_parameter.x)
    gen_estimated = gen_parameter.x.tolist()
    if parameterChoose == (0,1):
        gen_estimated.insert(2,1-sum(gen_estimated))
    elif parameterChoose == (0,2):
        gen_estimated.insert(1,1-sum(gen_estimated))
    elif parameterChoose == (1,2):
        gen_estimated.insert(0,1-sum(gen_estimated))
    else:
        print("error")
    print(gen_estimated)
    # gen_loglikelihood=negloglikelihood(gen_estimated, "typical", countN)
    gen_loglikelihood = gen_parameter.fun
    print(gen_loglikelihood)
    print("finish")

    rst_parameter = minimize(fun = negloglikelihood, x0 = rst_probability, args = ("restrict", countN, parameterChoose), method = 'TNC')
    print(rst_parameter.x)
    rst_estimated = rst_parameter.x.tolist()
    if parameterChoose == (0,1):
        paste = (i-rst_estimated[0])/2
        rst_estimated.insert(0,paste)
        rst_estimated.insert(1,paste)
    elif parameterChoose == (0,2):
        paste = (i-rst_estimated[0])/2
        rst_estimated.insert(0,paste)
        rst_estimated.insert(2,paste)
    elif parameterChoose == (1,2):
        paste = (i-rst_estimated[0])/2
        rst_estimated.insert(1,paste)
        rst_estimated.insert(2,paste)
    else:
        print("error")
    rst_loglikelihood = rst_parameter.fun
    G = 2 * (rst_loglikelihood - gen_loglikelihood)
    p_value = chi2.sf(G, 1)
    
    return p_value