# -CKD-Screening-Tool
ABSTARCT
Based on the given patient data, the objective of this case study is to create an easy-to-use screening tool on a digital platform to identify patients at risk for Chronic Kidney Disease (CKD). Furthermore, we perform this by creating and implementing a multivariate logistic regression model to predict and see if high-risk patients can be identified using easily obtainable patient data. 
Chronic Kidney Disease (CKD) is a progressive condition that explains the gradual loss of kidney function over time. The two main causes of CKD medically known to be are diabetes and high blood pressure or hypertension. Other factors that can be positively correlated with the onset of CKD are Cardiovascular disease (CVD), family history of kidney disease, age, and race (especially Pacific Islanders, African-American/Black, and Hispanics). Treatments can help slow down the progressive nature of CKD; however, this disease cannot be cured. This indicates the importance of recognizing the early onset of CKD and intervening as soon as possible to halt the progress and damage that CKD can pose. 

METHODOLOGY
The dataset for the case study consists of responses for specifically designed questionnaire from 8819 individuals, aged 20 years or older taken between 1999-2000 and 2001-2002 during a survey across various states in USA. The dataset is divided into two sets 1) Training set with 6000 observations in which 33 variables along with the status of CKD is provided. 2) Validation set consisting of 2819 observations with same set of variables in which the CKD has to be predicted. The predictions were to be made based on – 1) a statistical model and  2)a screening tool. The predictions for the statistical model were made using logistic regression. The data driven screening tool with calculated risk scores which was based on the logistic regression model used 6 of the 8 variables and had an AUC score of 0.82.
The in-sample data consisting of 6000 observations was divided into one training (3000) and two test sets (1500 each). 

MISSING DATA
Our dataset consists of 8819 responses against 33 attributes (8819 x 33) 291027 individual responses are to be recorded. But only 283285 are recorded and 7742 records are missing (which is about 2.6 % of the data set). Four dummy variables have been created for Race group (Black, White, Hispanic and others).

IMPUTATION
To deal with the missing data here, MICE package has been used with mean imputation so that the overall mean will not be affected. Each of the three data sets – train, test1 and test2 were imputed separately.

VARIABLE SELECTION
Attribute selection methods are used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model. 
Based on correlation, LASSO regression and p-values of the logistics regression we created two models:
Model 1: Age, Weight, Height, Hypertension, Diabetes, CVD, Female, Race(white) 
Model 2: Age, Waist, Hypertension, Diabetes, CVD, Female, Race(white). 
Finally, the AIC value and several accuracy metrics such as TPR, FPR, f-measure, accuracy hlped zero it down on model 1. Although model 2 had an accuracy value of 91%, the other metrics were way better and also a better performance indicator for a model, especially when the goal is to reduce the number of false positives.

THRESHOLD SELECTION
In order to convert probability values to actual predictions – 0’s and 1’s; a threshold value is slected. For a given threshold value if the probability value of CKD is greater than the threshold value, we predict the individual has CKD.
The threshold value for the logistic regression output was 0.2. This value was chosen between the 3rd quantile and half of the maximum values in the probabilities predicted by the Logistic Regression. The model performance was found to be optimal at a probability cut-off value of 0.2. This value was also in line with the ultimate goal of reducing the false positive rates.

LIMITATIONS
The white race group was overrepresented in this data set, while the Hispanic race group were underrepresented.
The model does its best to accurately predict individuals at higher for CKD; however, it does not account for outliers.
The sample presented in this data set was not a random sample, the findings cannot be representative for the whole population. A random sample is necessary for an unbiased representation of the population.


 



