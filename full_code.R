library(ggplot2)
library(tidyr)
library(dplyr)
library(knitr)
library(caret)
library(corrgram)
library(pROC)
library(plyr)
library(caretEnsemble)
library(doMC)
registerDoMC(cores = 4)

##############################################################################################

# A function to train multiple models on the same training data
train_models <- function(data, methods){
  
  t <- list(
    obs= data %>% dplyr::select(-TARGET,-CONTRACT_KEY),
    class= (data %>% dplyr::select(TARGET))[,1]
  )

  set.seed(1234556)
  models <- caretList(
    x= t$obs,
    y= t$class,
    metric="ROC",
    preProcess=c( "center", "scale"),
    tuneList = list(
      rf= caretModelSpec(method="rf")
    ),
    continue_on_fail = T,
    trControl = trainControl(method = "repeatedcv",
                             number=5,
                             repeats = 1 ,
                             classProbs = TRUE,
                             savePredictions = "final",
                             summaryFunction= twoClassSummary,
                             allowParallel = TRUE
    )
  )
  
  return(models)
}

# A function to train a single model with a grid search on the params
train_model_single_tune <- function(data, method, grid){
  
  t <- list(
    obs= data %>% dplyr::select(-TARGET,-CONTRACT_KEY),
    class= (data %>% dplyr::select(TARGET))[,1]
  )

  set.seed(1234556)
  model <- train(
    x= t$obs,
    y= t$class,
    preProcess=c( "center", "scale"),
    method=method,
    metric="ROC",
    trControl = trainControl(method="LGOCV", number=1, classProbs = T, summaryFunction = twoClassSummary, allowParallel = TRUE),
    tuneGrid = grid
  )
  
  return(model)
}

# A function to train a certain model with certain params
train_model_no_tune <- function(data, method, grid){
  
  t <- list(
    obs= data %>% dplyr::select(-TARGET,-CONTRACT_KEY),
    class= (data %>% dplyr::select(TARGET))[,1]
  )

  set.seed(1234556)
  model <- train(
    x= t$obs,
    y= t$class,
    preProcess=c( "center", "scale"),
    method=method,
    trControl = trainControl(method="none",allowParallel = TRUE),
    tuneGrid = grid
  )
  
  return(model)
}


# A function that takes the model and the test data and prepares a submission
predict_and_create_submission <- function(model, test_data){
  
  test_data_without_contract_key <- test_data %>% dplyr::select(-CONTRACT_KEY)
  CONTRACT_KEYS <- (test_data %>% dplyr::select(CONTRACT_KEY))[,1] %>% as.factor
  predictions <- predict(model, test_data_without_contract_key)
  
  predictions <- predictions %>% sapply(function(x){return(ifelse(x == "X0", 0,1))})
  
  submission <- data.frame(
    CONTRACT_KEY = CONTRACT_KEYS,
    PREDICTED_TARGET = predictions
  )
  
  write.csv(submission,quote=FALSE,row.names=FALSE, file= paste("submissions/", Sys.time(), ".csv", sep=""))
  
  return(submission)
}

#############################################################################
if(FALSE){
  #ONLY USED ONCE TO CREATE THE USAGE ESTIMATE COLUMN OF THE SIXTH MONTH
  calculate_month_estimate <- function(u1,u2,u3,u4,u5){
    d <- data.frame(
      x = (1:5),
      y =c(u1,u2,u3,u4,u5)
    )
    p <- predict(lm(y~x, data=d), data.frame(x=c(6)))[[1]]
    return(p)
  }
  
  train_es <- train_data %>% rowwise() %>% do(ES=calculate_month_estimate(.$X206_USAGE, .$X207_USAGE, .$X208_USAGE, .$X209_USAGE,.$X210_USAGE)) %>% summarise(ES=ES)
  train_es <- data.frame(ES = train_es$ES %>% unlist)
  write.csv(train_es, quote=FALSE,row.names=FALSE, file="./data/train_estimated.csv")
  
  test_es <- test_data %>% rowwise() %>% do(ES=calculate_month_estimate(.$X206_USAGE, .$X207_USAGE, .$X208_USAGE, .$X209_USAGE,.$X210_USAGE)) %>% summarise(ES=ES)
  test_es <- data.frame(ES = test_es$ES %>% unlist)
  write.csv(test_es, quote=FALSE,row.names=FALSE, file="./data/test_estimated.csv")
}
#######################################################################################

# Reading all the needed datasets

#roaming_data <- read.csv("data/roaming_monthly.csv")
contract_data <- read.csv("data/contract_clean.csv")
test_data <- read.csv("data/test.csv")
train_data <- read.csv("data/train.csv")
daily_aggregate <- read.csv("data/daily_aggregate.csv")
train_estimate_data <- read.csv("data/train_estimated.csv")
train_data$ES <- train_estimate_data$ES
test_estimate_data <- read.csv("data/test_estimated.csv")
test_data$ES <- test_estimate_data$ES
rm(train_estimate_data)
rm(test_estimate_data)

# Manipulating Daily Aggregate
daily_aggregate_man <- daily_aggregate %>% group_by(CONTRACT_KEY, CALL_DATE_KEY) %>% summarise_each(funs(sum), TOTAL_CONSUMPTION) %>% mutate(TOTAL_CONSUMPTION=TOTAL_CONSUMPTION/1024/1024) %>% mutate(CALL_DATE_KEY=paste("USAGE_DAY_", CALL_DATE_KEY, sep="")) %>% spread(CALL_DATE_KEY, TOTAL_CONSUMPTION, fill=0)

# Daily Predict
#daily_predict <- read.csv("data/predict_daily.csv")

######################################################################################

# A function that takes the data and do the feature engineering with hot one encoding of
# the categorical values (DEPRECATED)
prepare_dataset_one_hot <- function(data, is_train) {
  ret_data <- data
  
  ret_data <- inner_join(ret_data, contract_data, by="CONTRACT_KEY") 
  
  ret_data <- ret_data %>% dplyr::select(-X, -HANDSET_NAME, -RATE_PLAN, -VALUE_SEGMENT)
  ret_data <- model.matrix(~. -1, data=ret_data, fullRank=F) %>% as.data.frame
  
  # FIXING THE AGE
  ret_data[ret_data$AGE == 99,]$AGE <- (ret_data[ret_data$AGE != 99,]$AGE %>% median)
  
  
  # Adding Some cols
  ret_data <- ret_data %>% mutate(USAGE_MEAN=(X206_USAGE+X207_USAGE+X208_USAGE+X209_USAGE+X210_USAGE)/5)
  
  if(is_train){
    
    ret_data <- ret_data %>% mutate(TARGET=as.factor(TARGET))
    levels(ret_data$TARGET) <- make.names(levels(factor(ret_data$TARGET)))
    
    # Drop UNNEEDED COLUMNS
    ret_data <- ret_data %>% dplyr::select(-ends_with("_SESSION_COUNT"))
    #ret_data <- ret_data %>% dplyr::select(-starts_with("SESSIONS_"))
    #ret_data <- ret_data %>% dplyr::select(-matches("X2"))
    
    
    # Removing almost zero variance variables
    #nzv <- nearZeroVar(ret_data, names=TRUE)
    #nzv <- nzv [! nzv %in% c("TARGET")]
    #ret_data <- ret_data %>% dplyr::select(-one_of(nzv))
    
    # Removing highly correlated features
    #descrCor <- cor(ret_data %>% dplyr::select(-TARGET))
    #highlyCorDescr <- findCorrelation(descrCor, cutoff = .90, names=TRUE)
    #ret_data <- ret_data %>% dplyr::select(-one_of(highlyCorDescr))
    
    # Removing linearly corelated values
    #lc <- findLinearCombos(ret_data %>% dplyr::select(-TARGET))
    
    #if(!is.null(lc$remove)){
    #  ret_data <- ret_data[, -lc$remove]
    #}
  }
  
  return(ret_data)
}


# The function responsible for the feature engineering work
# Each part is commented

prepare_dataset <- function(data, is_train) {
  ret_data <- data
  
  # Join the contract data of the users
  ret_data <- inner_join(ret_data, contract_data, by="CONTRACT_KEY") 
  
  # Merge the featured daily aggregated data.frame
  ret_data <- left_join(ret_data, daily_aggregate_man , by="CONTRACT_KEY") 
  
  # FIXING THE AGE
  ret_data[ret_data$AGE == 99,]$AGE <- (ret_data[ret_data$AGE != 99,]$AGE %>% median)
  
  # Adding Some cols ( Will be descriped in a seperate document)
  ret_data <- ret_data %>% mutate(USAGE_MEAN=(X206_USAGE+X207_USAGE+X208_USAGE+X209_USAGE+X210_USAGE)/5)
  ret_data <- ret_data %>% mutate(USAGE_DIFF=X210_USAGE-X206_USAGE)
  
  ret_data <- ret_data %>% mutate(USAGE_FARGHAL=as.factor(ES > 0.9*USAGE_MEAN + 500))
  ret_data <- ret_data %>% mutate(USAGE_VARIANCE=(((X206_USAGE - USAGE_MEAN)^2 + (X207_USAGE - USAGE_MEAN)^2 + (X208_USAGE - USAGE_MEAN)^2 + (X209_USAGE - USAGE_MEAN)^2 + (X210_USAGE - USAGE_MEAN)^2)/5)^0.5)
  
  ret_data <- ret_data %>% mutate(USAGE_DAY_6337=ifelse(is.na(USAGE_DAY_6337), X210_USAGE/30,USAGE_DAY_6337))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6338=ifelse(is.na(USAGE_DAY_6338), X210_USAGE/30,USAGE_DAY_6338))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6339=ifelse(is.na(USAGE_DAY_6339), X210_USAGE/30,USAGE_DAY_6339))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6340=ifelse(is.na(USAGE_DAY_6340), X210_USAGE/30,USAGE_DAY_6340))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6341=ifelse(is.na(USAGE_DAY_6341), X210_USAGE/30,USAGE_DAY_6341))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6342=ifelse(is.na(USAGE_DAY_6342), X210_USAGE/30,USAGE_DAY_6342))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6343=ifelse(is.na(USAGE_DAY_6343), X210_USAGE/30,USAGE_DAY_6343))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6344=ifelse(is.na(USAGE_DAY_6344), X210_USAGE/30,USAGE_DAY_6344))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6345=ifelse(is.na(USAGE_DAY_6345), X210_USAGE/30,USAGE_DAY_6345))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6346=ifelse(is.na(USAGE_DAY_6346), X210_USAGE/30,USAGE_DAY_6346))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6347=ifelse(is.na(USAGE_DAY_6347), X210_USAGE/30,USAGE_DAY_6347))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6348=ifelse(is.na(USAGE_DAY_6348), X210_USAGE/30,USAGE_DAY_6348))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6349=ifelse(is.na(USAGE_DAY_6349), X210_USAGE/30,USAGE_DAY_6349))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6350=ifelse(is.na(USAGE_DAY_6350), X210_USAGE/30,USAGE_DAY_6350))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6351=ifelse(is.na(USAGE_DAY_6351), X210_USAGE/30,USAGE_DAY_6351))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6352=ifelse(is.na(USAGE_DAY_6352), X210_USAGE/30,USAGE_DAY_6352))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6353=ifelse(is.na(USAGE_DAY_6353), X210_USAGE/30,USAGE_DAY_6353))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6354=ifelse(is.na(USAGE_DAY_6354), X210_USAGE/30,USAGE_DAY_6354))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6355=ifelse(is.na(USAGE_DAY_6355), X210_USAGE/30,USAGE_DAY_6355))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6356=ifelse(is.na(USAGE_DAY_6356), X210_USAGE/30,USAGE_DAY_6356))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6357=ifelse(is.na(USAGE_DAY_6357), X210_USAGE/30,USAGE_DAY_6357))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6358=ifelse(is.na(USAGE_DAY_6358), X210_USAGE/30,USAGE_DAY_6358))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6359=ifelse(is.na(USAGE_DAY_6359), X210_USAGE/30,USAGE_DAY_6359))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6360=ifelse(is.na(USAGE_DAY_6360), X210_USAGE/30,USAGE_DAY_6360))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6361=ifelse(is.na(USAGE_DAY_6361), X210_USAGE/30,USAGE_DAY_6361))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6362=ifelse(is.na(USAGE_DAY_6362), X210_USAGE/30,USAGE_DAY_6362))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6363=ifelse(is.na(USAGE_DAY_6363), X210_USAGE/30,USAGE_DAY_6363))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6364=ifelse(is.na(USAGE_DAY_6364), X210_USAGE/30,USAGE_DAY_6364))
	ret_data <- ret_data %>% mutate(USAGE_DAY_6365=ifelse(is.na(USAGE_DAY_6365), X210_USAGE/30,USAGE_DAY_6365))

  ret_data <- ret_data %>% mutate(USAGE_DAY_EXPECTED=(USAGE_DAY_6365 - USAGE_DAY_6337)/30 + USAGE_DAY_6365)
	
	ret_data <- ret_data %>% mutate(USAGE_PER_DAY=((USAGE_DAY_6337+ USAGE_DAY_6338+ USAGE_DAY_6339+ USAGE_DAY_6340+ USAGE_DAY_6341+ USAGE_DAY_6342+ USAGE_DAY_6343+ USAGE_DAY_6344+ USAGE_DAY_6345+ USAGE_DAY_6346+ USAGE_DAY_6347+ USAGE_DAY_6348+ USAGE_DAY_6349+ USAGE_DAY_6350+ USAGE_DAY_6351+ USAGE_DAY_6352+ USAGE_DAY_6353+ USAGE_DAY_6354+ USAGE_DAY_6355+ USAGE_DAY_6356+ USAGE_DAY_6357+ USAGE_DAY_6358+ USAGE_DAY_6359+ USAGE_DAY_6360+ USAGE_DAY_6361+ USAGE_DAY_6362+ USAGE_DAY_6363+ USAGE_DAY_6364+ USAGE_DAY_6365)/30))
  
  ret_data$RATE_PLAN <- ret_data$RATE_PLAN %>% as.character %>% sapply(function(x) ifelse(grepl("enterprise|business", tolower(x)), "Business", x)) %>% as.factor
  
  ret_data <- ret_data %>% mutate(USAGE_DIFF_POS=as.factor(USAGE_DIFF > 0))
  ret_data <- ret_data %>% mutate(USAGE_DAY_DIFF=USAGE_DAY_6365 - USAGE_DAY_6337)
  
  # Dropping some columns that's not important for our model.
  if(is_train){
    
    ret_data <- ret_data %>% filter(USAGE_MEAN < quantile(ret_data$USAGE_MEAN, c(0.99))[[1]])
    
    es_99_quantile <- quantile(ret_data$ES, c(0.99))[[1]]
    es_1_quantile <- quantile(ret_data$ES, c(0.01))[[1]]
    ret_data <- ret_data %>% filter(ES < es_99_quantile)
    ret_data <- ret_data %>% filter(ES > es_1_quantile)
    
    ret_data <- ret_data %>% filter(USAGE_PER_DAY < quantile(ret_data$USAGE_PER_DAY, c(0.99))[[1]])
    ret_data <- ret_data %>% filter(USAGE_DAY_EXPECTED < quantile(ret_data$USAGE_DAY_EXPECTED, c(0.99))[[1]])
    ret_data <- ret_data %>% filter(USAGE_DIFF > quantile(ret_data$USAGE_DIFF, c(0.01))[[1]])
    ret_data <- ret_data %>% filter(USAGE_DAY_6365 < quantile(ret_data$USAGE_DAY_6365, c(0.99))[[1]])
    
    ret_data <- ret_data %>% mutate(TARGET=as.factor(TARGET))
    levels(ret_data$TARGET) <- make.names(levels(factor(ret_data$TARGET)))
    
    # Drop UNNEEDED COLUMNS
    #ret_data <- ret_data %>% dplyr::select(-ends_with("_SESSION_COUNT"))
    #ret_data <- ret_data %>% dplyr::select(-starts_with("SESSIONS_"))
    ret_data <- ret_data %>% dplyr::select(-matches("X20"))
    ret_data <- ret_data %>% dplyr::select(-num_range("USAGE_DAY_",6337:6364))
    ret_data <- ret_data %>% dplyr::select(-USAGE_PER_DAY)
    
    # Removing almost zero variance variables
    #nzv <- nearZeroVar(ret_data, names=TRUE)
    #nzv <- nzv [! nzv %in% c("TARGET")]
    #ret_data <- ret_data %>% dplyr::select(-one_of(nzv))
    
    # Removing highly correlated features
    #descrCor <- cor(ret_data %>% dplyr::select(-TARGET))
    #highlyCorDescr <- findCorrelation(descrCor, cutoff = .90, names=TRUE)
    #ret_data <- ret_data %>% dplyr::select(-one_of(highlyCorDescr))
    
    # Removing linearly corelated values
    #lc <- findLinearCombos(ret_data %>% dplyr::select(-TARGET))
    
    #if(!is.null(lc$remove)){
    #  ret_data <- ret_data[, -lc$remove]
    #}
  }
  
  return(ret_data)
}

# A function that divides the training data into a train dataset and a validation one. Then applies
# the feature engineering function on each part along with the test set.
divide_train_data <- function(){
  set.seed(Sys.time())
  parts <- createDataPartition(train_data$TARGET, p=0.75, list=F)
  validate_data <- train_data[-parts,]
  train_data <- train_data[parts,]
  rm(parts)
  
  train_transformed_data <- prepare_dataset(train_data, TRUE)
  test_transformed_data <- prepare_dataset(test_data, FALSE)
  test_transformed_data <- test_transformed_data[, names(test_transformed_data) %in% (train_transformed_data %>% names)]
  
  validate_transformed_data <- prepare_dataset(validate_data, FALSE)
  validate_transformed_data <- validate_transformed_data[, names(validate_transformed_data) %in% (train_transformed_data %>% names)]
  validate_transformed_data <- validate_transformed_data %>% mutate(TARGET=as.factor(TARGET))
  levels(validate_transformed_data$TARGET) <- make.names(levels(factor(validate_transformed_data$TARGET)))
  
  
  return(list(
    train = train_transformed_data,
    test = test_transformed_data,
    validate = validate_transformed_data
  ))
}

#A function that applies the feature engineering function on the whole training set along with the test
# set. Used for the final submission training.
prepare_full_data_train <- function(){
  set.seed(Sys.time())
  
  train_transformed_data <- prepare_dataset(train_data, TRUE)
  test_transformed_data <- prepare_dataset(test_data, FALSE)
  
  test_transformed_data <- test_transformed_data[, names(test_transformed_data) %in% (train_transformed_data %>% names)]
  
  return(list(
    train = train_transformed_data,
    test = test_transformed_data
  ))
}

##########################################################################

# Runs a kind of manual cross validation.
for(i in (1:5)){
  data <- divide_train_data()
  nb_model <- train_model_no_tune(data$train, "nb", expand.grid(fL=0, usekernel=T, adjust=1))
  print(nb_model$method)
  p <- predict(nb_model, data$validate %>% dplyr::select(-TARGET,-CONTRACT_KEY))
  print(auc(as.numeric(data$validate$TARGET), as.numeric(p)))
}


# Trains on the full dataset for the final submission
data <- prepare_full_data_train()
nb_model <- train_model_no_tune(data$train, "nb", expand.grid(fL=0, usekernel=T, adjust=1))
print(nb_model$method)
p <- predict(nb_model, data$train %>% dplyr::select(-TARGET,-CONTRACT_KEY))
print(auc(as.numeric(data$train$TARGET), as.numeric(p)))

#Prepare the submission and predict on the test data
submission <- predict_and_create_submission(nb_model, data$test)