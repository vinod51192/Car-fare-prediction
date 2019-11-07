
# Cab Fare Prediction 

rm(list = ls())
setwd("G:/EdwisorAssignments/Cab Fare Prediction")

# #loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','xgboost','stats','plyr','dplyr',"corrgram","DataCombine")


install.packages("DMwR")
install.packages(c('devtools','curl'))
library('devtools')
#load Packages
lapply(x, require, character.only = TRUE)
rm(x)


# loading datasets
train = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("test.csv", na.strings = c(" ", "", "NA"))
test_pickup_datetime = test["pickup_datetime"]
# Structure of data
str(train)
str(test)
summary(train)
summary(test)
head(train,5)
head(test,5)

#############                              Exploratory Data Analysis                    #######################
# Changing the data types of variables
train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count=round(train$passenger_count)

### Removing values which are not within desired range(outlier) depending upon basic understanding of dataset.

# 1.Fare amount has a negative value, which doesn't make sense. A price amount cannot be -ve and also cannot be 0. 
#Also 2 observations have values 54343 and 4343 which is not possible for cab fare.So we will remove these fields.

train$fare_amount[order(-train$fare_amount)]
train[which(train$fare_amount < 1 ),]

train = train[-which(train$fare_amount < 1 ),]
train = train[-which(train$fare_amount > 1000 ),]

summary(train)

#2.Passenger_count variable
count(train[which(train$passenger_count >6),])
count(train[which(train$passenger_count <1),])

# so 20 observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's check them.
train[which(train$passenger_count >6 ),]
train[which(train$passenger_count < 1 ),]

# We will remove these observations which are above 6 value because a cab cannot hold these number of passengers.
train = train[-which(train$passenger_count < 1 ),]
train = train[-which(train$passenger_count > 6),]

# 3.Latitudes range from -90 to 90.Longitudes range from -180 to 180.Removing which does not satisfy these ranges

nrow(train[which(train$pickup_longitude >180 ),])
nrow(train[which(train$pickup_longitude < -180 ),])
nrow(train[which(train$pickup_latitude > 90 ),])
nrow(train[which(train$pickup_latitude < -90 ),])
nrow(train[which(train$dropoff_longitude > 180 ),])
nrow(train[which(train$dropoff_longitude < -180 ),])
nrow(train[which(train$dropoff_latitude < -90 ),])
nrow(train[which(train$dropoff_latitude > 90 ),])

# There's only one outlier which is in variable pickup_latitude.So we will remove it with nan.
# Also we will see if there are any values equal to 0.

nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
# there are values which are equal to 0. we will remove them.
train = train[-which(train$pickup_latitude > 90),]
train = train[-which(train$pickup_longitude == 0),]
train = train[-which(train$dropoff_longitude == 0),]

################Pickup_Datetime#####

train$pickup_datetime=as.Date(as.character(train$pickup_datetime)) 

str(train)

# Make a copy
df=train
# train=df

#############                        Missing Value Analysis                  #############
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
missing_val

################ Bar graph of passenger count #############################################
PassengerCount=ggplot(data = train, aes(x =passenger_count)) + ggtitle("passenger count") + geom_bar()
gridExtra::grid.arrange(PassengerCount)

unique(train$passenger_count)
unique(test$passenger_count)

train[,'passenger_count'] = factor(train[,'passenger_count'], labels=(1:6))
test[,'passenger_count'] = factor(test[,'passenger_count'], labels=(1:6))

# 1.For Passenger_count:
# Actual value = 1
# Mode = 1
# KNN = 1
train$passenger_count[1000]
train$passenger_count[1000] = NA

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Mode Method
getmode(train$passenger_count)
train$passenger_count[is.na(train$passenger_count)] = getmode(train$passenger_count)
sum(is.na(train$passenger_count))

train= knnImputation(train,k=4)

# 2.For fare_amount:
# Actual value = 10.5,
# Mean = 11.366,
# Median = 8.5,
# KNN = 10.833

train$fare_amount[1000]
train$fare_amount[1000]= NA

# Mean Method
mean(train$fare_amount, na.rm = T)

#Median Method
median(train$fare_amount, na.rm = T)

# kNN Imputation
train = knnImputation(train, k = 3)
train$fare_amount[1000]
train$fare_amount[1000]= NA

train$passenger_count[1000]
sapply(train, sd, na.rm = TRUE)

sum(is.na(train))
str(train)
summary(train)

df1=train
# train=df1
#####################                        Outlier Analysis                 ##################

# We Will do Outlier Analysis only on Fare_amount just for now and we will do outlier analysis after feature engineering laitudes and longitudes.
# Boxplot for fare_amount

pl1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Removing the outliers
vals = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out
summary(vals)

train = train[which(!train$fare_amount %in% vals),]

#lets check the NA's
sum(is.na(train$fare_amount))

#Imputing with KNN
train = knnImputation(train,k=3)

# lets check the missing values
sum(is.na(train$fare_amount))
str(train)
df2=train

##Histogram##


# train=df2
##################                   Feature Engineering                       ##########################
# 1.Feature Engineering for timestamp variable
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time
train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))# Monday = 1
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$pickup_hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
pickup_time_test = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time_test,"%H"))

sum(is.na(train))# there were 5 'na' in pickup_datetime which created na's in above feature engineered variables.
train = na.omit(train) # we will remove that 1 row of na's

train = subset(train,select = -c(pickup_datetime,pickup_date))
test = subset(test,select = -c(pickup_datetime,pickup_time_test))

# 2.Calculate the distance travelled using longitude and latitude
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}
# Using haversine formula to calculate distance fr both train and test
train$dist = haversine(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$dist = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)

# We will remove the variables which were used to feature engineer new variables
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

str(train)
summary(train)

df3=train
#train=df3
# Data Visualization #
train$pickup_Duration = as.character(train$pickup_hour)

train$pickup_Duration[train$pickup_Duration %in% c("00","01","02","03")] = "Midnight"
train$pickup_Duration[train$pickup_Duration %in% c("04","05","06","07")] = "Early Morning"
train$pickup_Duration[train$pickup_Duration %in% c("08","09","10","11","12")] = "Morning"
train$pickup_Duration[train$pickup_Duration %in% c("13","14","15","16")] = "Afternoon"
train$pickup_Duration[train$pickup_Duration %in% c("17","18","19","20")] = "Evening"
train$pickup_Duration[train$pickup_Duration %in% c("21","22","23","24")] = "Night"

train$pickup_Duration=as.factor(train$pickup_Duration)

TimeBar = ggplot(data = train, aes(x = pickup_Duration))+ geom_bar()+ ggtitle("Time of day" )
gridExtra::grid.arrange(TimeBar)

scat1 = ggplot(data = train, aes(y =fare_amount, x = pickup_Duration)) + ggtitle("Fare as per time") + geom_point() + ylab("Fare") + xlab("Time Slot")
gridExtra::grid.arrange(scat1)

scat2 = ggplot(data = train, aes(y =fare_amount, x = dist)) + ggtitle("Fare as per time") + geom_point() + ylab("Fare") + xlab("Time Slot")
gridExtra::grid.arrange(scat2)

################                             Feature selection                 ###################
numeric_index = sapply(train,is.numeric) #selecting only numeric

numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)
#Correlation analysis for numeric variables
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#ANOVA for categorical variables with target numeric variable

#aov_results = aov(fare_amount ~ passenger_count * pickup_hour * pickup_weekday,data = train)
aov_results = aov(data = train,fare_amount ~ passenger_count + pickup_hour + pickup_weekday + pickup_mnth + pickup_yr)

summary(aov_results)

# pickup_weekdat has p value greater than 0.05 
train = subset(train,select=-pickup_weekday)

#remove from test set 
test = subset(test,select=-pickup_weekday)

##################################             Feature Scaling         ################################################
#Normality check
# qqnorm(train$fare_amount)
# histogram(train$fare_amount)

#Normalisation
df4=train
#train=df4

print('dist')
train[,'dist_norm'] = (train[,'dist'] - min(train[,'dist']))/
  (max(train[,'dist'] - min(train[,'dist'])))

train[,'Fare_norm'] = (train[,'fare_amount'] - min(train[,'fare_amount']))/
  (max(train[,'fare_amount'] - min(train[,'fare_amount'])))


summary(train)

scat3 = ggplot(data = train, aes(y =Fare_norm, x = dist_norm)) + ggtitle("Fare as per time") + geom_point() + ylab("Fare") + xlab("Time Slot")
gridExtra::grid.arrange(scat3)

scat3 = ggplot(data = train, aes(y =fare_amount, x = dist_norm)) + ggtitle("Fare as per time") + geom_point() + ylab("Fare") + xlab("Time Slot")
gridExtra::grid.arrange(scat3)

# #check multicollearity
# library(usdm)
# vif(train[,-1])
# 
# vifcor(train[,-1], th = 0.9)

##Scatter plot#
scat2 = ggplot(data = train, aes(y =fare_amount, x = dist)) + ggtitle("distance and fare") + geom_point() + ylab("Fare") + xlab("Time Slot")
gridExtra::grid.arrange(scat2)

#################### Splitting train into train and validation subsets ###################
set.seed(1000)
tr.idx = createDataPartition(train$fare_amount,p=0.75,list = FALSE) # 75% in trainin and 25% in Validation Datasets
train_data = train[tr.idx,]
test_data = train[-tr.idx,]

train=train[-6]

rmExcept(c("test","train","df",'df1','df2','df3','test_data','train_data','test_pickup_datetime'))
rmExcept(c('df3'))
###################Model Selection################
#Error metric used to select model is RMSE

#############            Linear regression               #################
lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)
str(train_data)
plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data[,2:6])

qplot(x = test_data[,1], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],lm_predictions)
# mae           mape 
# 3.5303114   0.4510407  


#############                             Decision Tree            #####################

Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[,2:6])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)
# mae        mape 
# 1.8981592  0.2241461 


#############                             Random forest            #####################
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,2:6])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)
# mae            mape 
# 1.9053850  0.2335395