## temp (model construct)
# (parameterChoose=(), correspondingRatio=[], assignedParameter=[])-->(model_parameter,df)
# parameterChoose: 指定的參數或要比較的參數
# correspondingRatio: 這組scenario本身的參數
# assignedParameter(optional): 給定的參數值 
import math
import numpy as np
from scipy.stats import logistic
from scipy.optimize import minimize
from scipy.stats import chi2
def generate_model_parameter(parameterChoose=(), correspondingRatio=[], count=[], assignedParameter=[]):
    if len(parameterChoose) == 3:  ## parameterChoose=(0,1,2)指定所有參數 --> 不用跑 minimize
        df = 0
        model_parameter = assignedParameter
        temp_parameter = model_parameter
        negtiveLogLikelihood = negloglikelihood(model_parameter, "typical", count)

    elif len(parameterChoose) == 2: ## parameterChoose為要比較是否相等的兩個參數 
        df = 1
        if parameterChoose == (0,1):
            model_parameter = correspondingRatio[2]
        elif parameterChoose == (0,2):
            model_parameter = correspondingRatio[1]
        elif parameterChoose == (1,2):
            model_parameter = correspondingRatio[0]
        else:
            print("parameterChoose error!")

        temp_estimate = minimize(fun = negloglikelihood, x0 = model_parameter, args = ("equal", count, parameterChoose), method = 'TNC')
        print(temp_estimate.x)
        temp_parameter = temp_estimate.x.tolist()
        if parameterChoose == (0,1):
            paste = (1-temp_parameter[0])/2
            temp_parameter.insert(0,paste)
            temp_parameter.insert(1,paste)
        elif parameterChoose == (0,2):
            paste = (1-temp_parameter[0])/2
            temp_parameter.insert(0,paste)
            temp_parameter.insert(2,paste)
        elif parameterChoose == (1,2):
            paste = (1-temp_parameter[0])/2
            temp_parameter.insert(1,paste)
            temp_parameter.insert(2,paste)
        else:
            print("error")
        negtiveLogLikelihood = temp_estimate.fun
        print("finish")        

    elif len(parameterChoose) == 1: ## parameterChoose為給定的參數 
        df = 1
        if parameterChoose == (0,): ## 指定 p
            model_parameter = correspondingRatio[1] ## 變動 q,相依自動產生 r
        elif parameterChoose == (1,): ## 指定 q
            model_parameter = correspondingRatio[0] ## 變動 p, 相依自動產生 r
        elif parameterChoose == (2,): ## 指定 r
            model_parameter = correspondingRatio[0] ## 變動 p, 相依自動產生 q
        else:
            print("parameterChoose error!")
        print(model_parameter)
        temp_estimate = minimize(fun = negloglikelihood, x0 = model_parameter, args = ("assigned",count,parameterChoose),method = 'TNC',bounds=[(0,1)])
        print(temp_estimate.x)
        temp_parameter = temp_estimate.x.tolist()
        if parameterChoose == (0,):
            temp_parameter.insert(0,assignedParameter)
            temp_parameter.insert(2,1-sum(temp_parameter))
        elif parameterChoose == (1,):
            temp_parameter.insert(1,assignedParameter)
            temp_parameter.insert(2,1-sum(temp_parameter))
        elif parameterChoose == (2,):
            temp_parameter.insert(1,assignedParameter)
            temp_parameter.insert(1,1-sum(temp_parameter))
        else:
            print("error")
        print(temp_parameter)
        # gen_loglikelihood=negloglikelihood(gen_estimated, "typical", countN)
        negtiveLogLikelihood = temp_estimate.fun
        print(negtiveLogLikelihood)
        print("finish")        

    elif len(parameterChoose) == 0: ## 跑 minize 應會與本身參數相同
        ### parameterChoose 的 conflict 還未修正
        df = 2
        model_parameter = [correspondingRatio[0], correspondingRatio[1]] ## 變動 p q, 相依自動產生 r

        temp_estimate = minimize(fun = negloglikelihood, x0 = model_parameter, args = ("typical",count,(0,1)), method='TNC',bounds=[(0,1),(0,1)])
        print(temp_estimate.x)
        temp_parameter = temp_estimate.x.tolist()
        temp_parameter.insert(2,1-sum(temp_parameter))
        # if parameterChoose == (0,1):
            # temp_parameter.insert(2,1-sum(temp_parameter))
        # elif parameterChoose == (0,2):
            # temp_parameter.insert(1,1-sum(temp_parameter))
        # elif parameterChoose == (1,2):
            # temp_parameter.insert(0,1-sum(temp_parameter))
        # else:
            # print("error")
        print(temp_parameter)
#         nloglikelihood=negloglikelihood(temp_estimate, "typical", countN)
        negtiveLogLikelihood = temp_estimate.fun
        print(negtiveLogLikelihood)
        print("finish")
    else:
        print("parameterChoose error!")

    return temp_parameter, df, negtiveLogLikelihood

class toLikelihood:
    ###    UNFINISH !
    def __init__(self, ratio, count, parameterChoose,assignedValue = None):
        self.ratio = ratio # ratio: 這組scenario本身的行為分配
        # estimatedParameter: 根據條件估計出來的分配
        # df: 這個條件下的 degree of freedom
        # negtiveLogLikelihood: 根據估計出來的分配(estimatedParameter)得到的negtive lod likelihood
        self.estimatedParameter, self.df, self.negtiveLogLikelihood = generate_model_parameter(parameterChoose = parameterChoose, correspondingRatio = ratio, count = count, assignedParameter=assignedValue)

def likelihood_ratio_test(alternative_hypothesis, null_hypothesis, df=0):

    df = alternative_hypothesis.df-null_hypothesis.df
    print(df)
    G = 2*(null_hypothesis.negtiveLogLikelihood-alternative_hypothesis.negtiveLogLikelihood)
    print(G)
    p_value = chi2.sf(G, df)
    print(p_value) ## if < .05, the two behavior are not the same. else if > .05, the two behavior are the same.
#     print('{:.5f}'.format(p_value))
    
    return G, p_value

#  temp(adjuster negtive log likelihood)
#  modeltype: typical(沒有任何條件)/equal(任兩個參數相等)/assigned(指定某個參數)
def negloglikelihood(prob, modeltype = "typical", data = [], parameterChoose = None, assignedValue=None):
    data_=np.array(data)
    print(prob)
    prob_ = np.zeros(3)

    if modeltype == "typical":
        if len(prob)==3:
            prob_ = prob
        elif len(prob)==2:
            decidedP = 1-prob[0]-prob[1]
            if parameterChoose == (0,1):
                prob = np.insert(prob,2,decidedP)
            elif parameterChoose == (1,2):
                prob=np.insert(prob,0,decidedP)
            elif parameterChoose == (0,2):
                prob=np.insert(prob,1,decidedP)
            else:
                print("parameterChoose set error!")
            prob_ = prob
        else:
            print("not support")
            
    elif modeltype == "equal":
        decidedP = (1-prob)/2
        print(decidedP)
        for i in range(0,3):
            if i in parameterChoose:
                prob_[i]=decidedP
            else:
                prob_[i]=prob
    elif modeltype == "assigned":
        prob_[parameterChoose] = assignedValue
        print(prob_)
        if parameterChoose == (0,): ## 指定 p
            prob_[1] = prob  ## 變動 q,相依自動產生 r
            prob_[2] = 1-prob_[0]-prob_[1]
        elif parameterChoose == (1,): ## 指定 q
            prob_[0] = prob ## 變動 p, 相依自動產生 r
            prob_[2] = 1-prob_[0]-prob_[1]
        elif parameterChoose == (2,): ## 指定 r
            prob_[0] = prob ## 變動 p, 相依自動產生 q
            prob_[1] = 1-prob_[0]-prob_[2]
        print(prob_)
        
    else:
        print("modeltype not support!")
    print(prob_)
    negloglikelihood = -(data_[0:3] * np.log(prob_)).sum()
    
    return negloglikelihood
