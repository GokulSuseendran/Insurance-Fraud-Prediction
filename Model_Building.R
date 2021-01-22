"""
	@script-author: Jyothi Tom, Shekhina Neha, Gokul S, Aishwarya Lakshmi, Srija S, Robin Wilson
	@script-description: R Code to clean the data and train the Logistic Regression Model  
  	@script-details: Written in RStudio
"""

library(DescTools)    #For Cramer's V
library(dummies)      #For creating dummy variables
library(ROCR)       #Library to plot ROC
library(Metrics)    #Library to calculate AUC
library(dplyr)

#Importing the data 
insurance <- read.csv("C:/Users/GOKUL SUSEENDRAN/Documents/Insurance_Fraud_Prediction-main/insurance_claims.csv")

#****************************************************MISSING VALUES***************************************************
#Get column names having '?' values
null_cols = unique(names(insurance)[which(insurance == '?', arr.ind=T)[, "col"]]);null_cols

#Check the relevance of the columns with '?' values using:
#1. Proportion of '?'
sapply(insurance[,null_cols], function(x) length(x[x=="?"])/nrow(insurance))
 #2. Chi-sq Test
for (i in null_cols){
  coll = insurance[,i][which(insurance[,i] !='?', arr.ind=T)];coll
  fraud = insurance$fraud_reported[which(insurance[,i]!='?', arr.ind=T)]
  cat(sprintf("***Chi-sq test results between Fraudulent Cases Reported and %s***\n",i))
  print(chisq.test(coll,fraud))}

#Since it's seen that these columns have no relevant effect on 'fraud_reported', drop them
#Other irrelevant columns were laos dropped
insurance[,c(null_cols,"policy_number", "incident_year", "insured_zip", "policy_bind_date", "incident_date",
             "incident_location", "incident_city", "policy_day", "policy_month", "policy_year")]<-NULL

#Exported the cleaned data to be used in the UI for plots
write.csv(insurance, "C:\\Users\\jyoth\\Desktop\\SEM 2\\Bhogle Sir\\R Proj\\Insurance\\Final\\cleaned_data.csv")

#*************************************************NUMERICAL FEATURES*************************************************

#MUlticollinearity
num<-NULL
num <- c(split(names(insurance),sapply(insurance, function(x) paste(class(x), collapse=" ")))$integer, split(names(insurance),sapply(insurance, function(x) paste(class(x), collapse=" ")))$numeric)
num<-num[!(num %in% c("incident_hour_of_the_day", "auto_year"))]
num
corr1 <- cor(insurance[, num]);corr1     #Computing the correlation between the numerical variables
write.csv(corr1, "C:\\Users\\jyoth\\Desktop\\SEM 2\\Bhogle Sir\\R Proj\\Insurance\\Final\\corr1.csv")


#PCA, considering numerical variables with high correlation
corr_nums<- c("months_as_customer","age","Avg_capital_loss","capital_gains","Avg_capital_gains","total_claim_amount","injury_claim","property_claim","vehicle_claim" )
insurance.pca<-prcomp(insurance[corr_nums],center = TRUE, scale. = TRUE)
summary(insurance.pca)
insurance.pca
var = insurance.pca$sdev^2
prop_var <- var/sum(var)
cumsum(prop_var)
plot(cumsum(prop_var))  
screeplot(insurance.pca, type="lines")
biplot(insurance.pca,scale=0)
insurance[, corr_nums] <- NULL
#First 5 PCs explain 06% of the variance. Hence, only those were selected
insurance <- cbind(insurance, insurance.pca$x[,1:5])
names(insurance)
#To double-check, if multicollinearity problem is fixed
num<-num[!(num %in% c("insured_zip", "incident_hour_of_the_day", "auto_year",corr_nums))]
num <- c(num, "PC1", "PC2", "PC3", "PC4", "PC5")
num
corr2 <- cor(insurance[, num]);corr2     #Computing the correlation between the numerical variables
write.csv(corr2, "C:\\Users\\jyoth\\Desktop\\SEM 2\\Bhogle Sir\\R Proj\\Insurance\\Final\\corr2.csv")

#ANOVA to check the effect of numeric features on the target variable
for (i in num){
  anova_res <- aov(insurance[,i]~insurance$fraud_reported, data = insurance)
  cat(sprintf('\nAnova Table: fraud_reported and %s\n',i))
  print(summary(anova_res))
  cat("----------------------------------------------------------------\n")
}
nums_final<-c("umbrella_limit","PC1")  #Since only these variables had an effect on 'fraud_reported'

#*************************************************CATEGORICAL FEATURES***********************************************

#Categorical features
others = names(insurance)[!(names(insurance) %in% c(num,"fraud_reported"))]; others  

#Chi-Sq Test of Independence and Cramer's V computed for all the dummy variables and the target variable
cats <- c()
for (i in others){
  cat(sprintf("\nChi-sq test results between \'fraud_reported\' and \"%s\" \n",i))
  p <- chisq.test(insurance[,i],insurance$fraud_reported)
  print(p)
  if(p$p.value<0.07){
    cats<-c(cats,i)}
  cat(sprintf("Cramer's V: %s\n", CramerV(insurance[,i],insurance$fraud_reported)))
  cat("\n------------------------------------------------------------------------------\n")
}
cats     #Relevant categorical features 


#Creating dummy variables
categ<-dummy.data.frame(insurance,names=cats, sep=":", all=FALSE)   #Creating dummy variables from the categorical variables 
names(categ)  
# final<-c(names(categ), nums_final, 'fraud_reported');final       #Final variables

insurance[,!(names(insurance) %in% c(nums_final, "fraud_reported"))]<-NULL  #Removing the other columns
insurance<- cbind(categ, insurance)     #Final cleaned dataset with the required features and the target variable
names(insurance)                        #Final variables - after preprocessing and EDA

str(insurance)                          #Data type of the columns in a dataframe
#Changing the data type of 'fraud_reported' into factor and labeling it as 0:No, 1:Yes
insurance$fraud_reported <- factor(insurance$fraud_reported, labels = c(0,1))   
dim(insurance)       #Shape: (1000,43)

#****************************************************MODEL BUILDING**************************************************

names(insurance)
train<- insurance[1:(0.75*nrow(insurance)),]                  #Training data containing 75% of the entire dataset
test<- insurance[(0.75*nrow(insurance)):nrow(insurance),]     #Test data containing 25% of the entire dataset

#Initial Logistic Regression Model with all the variables
lr<-glm(fraud_reported~., data = train, family=binomial(link="logit"))   
summary(lr)
  
prob <- predict(lr, test, type = "response")        
pred <- ifelse(prob > 0.5, 1, 0)               #Predicted classes

#Plot ROC Curve
pr <- prediction(pred,test$fraud_reported)
perf <- performance(pr, measure="tpr", x.measure="fpr")
auc(test$fraud_reported,pred)     #0.8494395
plot(perf) > auc(test$fraud_reported,pred)    #ROC Curve


#Improving the model by selected only a few variables 

final<-c("insured_hobbies:camping","insured_hobbies:chess","insured_hobbies:cross-fit","incident_severity:Major Damage","fraud_reported")
train1<-train[,final]
train1 =train1 %>%
  rename('camping' = 'insured_hobbies:camping',
         'chess' = 'insured_hobbies:chess',
         'cross_fit'= "insured_hobbies:cross-fit",
         'major_damage' = 'incident_severity:Major Damage')
lr1<-glm(fraud_reported~., data = train1, family=binomial(link="logit"))  #Final Logistic Regression Model
summary(lr1)

test1 <- test[,final] 
test1 =test1 %>%
  rename('camping' = 'insured_hobbies:camping',
         'chess' = 'insured_hobbies:chess',
         'cross_fit'= "insured_hobbies:cross-fit",
         'major_damage' = 'incident_severity:Major Damage')

prob1 <- predict(lr1, test1, type = "response")        
pred1 <- ifelse(prob1 > 0.5, 1, 0)               #Predicted classes

#Plot ROC Curve
pr1 <- prediction(pred1,test$fraud_reported)
perf1 <- performance(pr1, measure="tpr", x.measure="fpr")
auc(test$fraud_reported,pred1)     #0.9087746
plot(perf1) > auc(test$fraud_reported,pred1)    #ROC Curve

#Saving the model; Loaded in the UI 
saveRDS(lr1, file = "C:/Users/GOKUL SUSEENDRAN/Documents/Insurance_Fraud_Prediction-main/insurance_claims.csv/LR.rda")
