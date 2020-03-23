library('dplyr')
## Step 1  - Read in Data
data=read.csv("casestudydata.csv")
names(data)
data = data[,-1] #remove ID column

#Explore the data
summary(data)
out_sample=which(is.na(data$CKD)==1)
data_out=data[out_sample,]   ## the ones without a disease status
data_in=data[-out_sample,]   ## the ones with a disease status



## Step 2  - Missing Data
#install required packages 
install.packages('VIM')
library(VIM)
install.packages('mice')
library(mice)

#summary of in-sample data with missing values
summary(data_in)
dim(data_in)

#omit missing values
#data_in=na.omit(data_in)
## Step 3 and 4 - Correlation with missing data eliminated 
#cor(data_in)
#summary(data_in)

#dummyfying racegroup and caresource
data_new=model.matrix(~-1+Racegrp,data=data_in)
summary(data_new)
data_in=data_in[,-c(4:8)] #remove medically irrelavant variables 
data_in=cbind(data_in,data_new)
data_in = select(data_in, -c( Racegrp))
names(data_in)

#corrlation and visualization wihtout missing values
a = cor(data_in)
str(data_in)
names(data_in)
library(corrplot)
corrplot(a, method = "circle", type = 'upper')

#divide data into train, test1 and test2 and impute each of them separately
set.seed(123) #set seed to make the sample reporducible
samp = floor(0.5*nrow(data_in))  #floor returns largest int not > than the no.
id = sample(seq_len(nrow(data_in)), size = samp) 
train = data_in[id,]
test  = data_in[-id,]
dim(train)
dim(test)

samp_1 = floor(0.5*nrow(test))
id_1 = sample(seq_len(nrow(test)), size = samp_1) 
test1 = test[id_1,]
test2  = test[-id_1,]
dim(test2)
dim(test1)


#impute missing values - MICE package
#train
imputed_train = mice(train, m = 10, maxit = 20)
complete_train = complete(imputed_train, 10)

write.csv(complete_train, "train.csv")

#test1
imputed_test1 = mice(test1, m = 10, maxit = 20)
complete_test1 = complete(imputed_test1, 10)

write.csv(complete_test1, "test1.csv")

#test2
imputed_test2 = mice(test2, m = 10, maxit = 20)
complete_test2 = complete(imputed_test2, 10)

write.csv(complete_test2, "test2.csv")


#removed multicollinear and other variables-SBP,Fam.hypertension,Fam.CVD,Stroke,Obese,total.chol(dervied variable)
complete_train = select(complete_train, -c(Fam.Hypertension, Fam.CVD, Stroke, Total.Chol, Obese, Racegrp, Dyslipidemia, SBP))
complete_test1 = select(complete_test1, -c(Fam.Hypertension, Fam.CVD, Stroke, Total.Chol, Obese, Racegrp, Dyslipidemia, SBP))
complete_test2 = select(complete_test2, -c(Fam.Hypertension, Fam.CVD, Stroke, Total.Chol, Obese, Racegrp, Dyslipidemia, SBP))

#LASSO regression
library(glmnet)
x <- model.matrix(CKD~., complete_train)
y2 <- ifelse(complete_train$CKD == "pos", 1, 0)

y2 <- complete_train$CKD
y2 <- as.matrix(data.frame(y2))

# Find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(x, y2, alpha = 1, family = "binomial")
# Fit the final model on the training data
model <- glmnet(x, y2, alpha = 1, family = "binomial",
                lambda = 0.005)
plot(cv.lasso, label = TRUE)

cv.lasso$lambda.min
cv.lasso$lambda.1se
summary(model)

# Display regressio coefficients
coef(model, s= 0.005 )
summary(model)


##Logistic Regression 
#MODEL1
final_data = model.matrix(~-1+Age+Weight+Height+Female+Hypertension+Diabetes+CVD
                          +Racegrpwhite+CKD,data = complete_train)

final_data <- data.frame(final_data)

model_final=glm(CKD~.,family="binomial",data=final_data)
summary(model)

#MODEL 2 
final_data_1 = model.matrix(~-1+Age+Waist+Female+Hypertension+Diabetes+CVD
                            +Racegrpwhite+CKD,data = complete_train)

final_data_1 <- data.frame(final_data_1)

model_1=glm(CKD~.,family="binomial",data=final_data_1)
summary(model)


##TESTING THE DATA####
test_1 <- read.csv('test1.csv')
dim(test_1)
data_test_1 = model.matrix(~-1+Age+Weight+Height+Female+Hypertension+Diabetes+CVD
                               +Racegrpwhite, data = test_1)

data_test_1 <- data.frame(data_test_1)


#PREDICT
model_predict <- predict(model_final, newdata = data_test_1, type = "response")
summary(model_predict)


##predictions <- rep("CKD", dim(final_data_test)[1])
library(ROCR)
?ROCR
data(ROCR.simple)
pred <- prediction(ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = rainbow(10))

summary(model_predict)
data("ROCR.simple")
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
plot(perf)
abline(a=0, b=1)
text(0.8, 0.2, labels = paste("AUC = ", round(auc,3)))
title("ROC CURVE")



##ACCURACY##
summary(model_predict)

classify=ifelse(model_predict>0.2,1,0)  
#summary(classify)  
#classify
#write.csv()
c_accuracy(test_1$CKD,classify) 
c_accuracy

c_accuracy=function(actuals,classifications){
  df=data.frame(actuals,classifications);
  
  
  tp=nrow(df[df$classifications==1 & df$actuals==1,]);        
  fp=nrow(df[df$classifications==1 & df$actuals==0,]);
  fn=nrow(df[df$classifications==0 & df$actuals==1,]);
  tn=nrow(df[df$classifications==0 & df$actuals==0,]); 
  
  
  recall=tp/(tp+fn)
  precision=tp/(tp+fp)
  accuracy=(tp+tn)/(tp+fn+fp+tn)
  tpr=recall
  fpr=fp/(fp+tn)
  fmeasure=2*precision*recall/(precision+recall)
  ckddollars=1300*tp-100*(fp)
  churndollars=1600*tp-100*(fp+tp)
  scores=c(recall,precision,accuracy,tpr,fpr,fmeasure,ckddollars,churndollars)
  names(scores)=c("recall","precision","accuracy","tpr","fpr","fmeasure","ckd profit","churn profit")
  
  #print(scores)
  return(scores);
}
## Function above Built by Matthew J. Schneider
##  actuals and classifications should be 0 (no) or 1 (yes) only 

model_predict <- data.frame(model_predict)
model_predict


predict_LR <- rep('no CKD', length(model_predict))
predict_LR[model_predict>= 0.5] <- 'CKD'
table(predict_LR, final_data_test$CKD)


library(caret)
confusionMatrix(final_data_test$CKD, predict_LR)


main <- plot(roc(final_data_test$CKD, model_predict), print.auc = TRUE, col = "blue")

