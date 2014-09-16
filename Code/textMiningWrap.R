


# ===============================================================================
# Code for mining text in R
# 
# The entire procedure of pre-processing, stemming, document-term matrix, 
# predictive modeling, and feature selection has been wrapped into one 
# function
#
# Date: 2014-09-16 (first); 2014-09-16 (current) 
# Dependencies: tm, rJava, Snowball, RWeka, parallel
# Tested on: R 3.0.2, OSX 10.9.4
#
# Input: Text vector and target vector
# Output: Important words from text with standardized coeffiecients 
# NOTE: The output list of words is not unique; separate coefficient is provided
#       for each fold to allow cross-fold comparison
# ===============================================================================


# load data
train <- readLines("Data/SMSSpamCollection", n= -1)
train <- strsplit(train, "\t")
target <- do.call("c", lapply(1:length(train), function(x) train[[x]][1]))
target <- ifelse(target == "ham", 0, 1)
table(target)
train <- do.call("c", lapply(1:length(train), function(x) train[[x]][2]))
train <- data.frame(target, text= train, stringsAsFactors= FALSE)
write.table(train, file= "Data/SMSSpamCollection.txt", sep= "\t", row.names= FALSE)


# =========================
# Pre-process functions
# =========================

# function to create and process corpus
prepareCorpus <- function(x, vec.source= TRUE){
  require(tm)
  require(rJava)
  require(Snowball)
  if (vec.source){
    crps <- Corpus(VectorSource(x)) 
  } else{
    my.corpus <- Corpus(DataframeSource(x))
  }
  crps <- tm_map(crps, stripWhitespace)
  crps <- tm_map(crps, tolower)
  crps <- tm_map(crps, removePunctuation)
  crps <- tm_map(crps, removeWords, stopwords('english'))
  crps
}

# function to create TDM
createTDM <- function(crps, ngram.min= 1, ngram.max= 1, min.rows.pop= 100, 
                      weight= weightTf, out.sparse= TRUE){
  require(RWeka)
  require(Matrix)
  library(parallel)
  options(mc.cores= 1)
  
  # create tokenizer
  tokenizer <- function(x){
    NGramTokenizer(x, Weka_control(min= ngram.min, max= ngram.max))
  }
  tdm <- TermDocumentMatrix(crps,
                            control= list(weighting= weight, 
                                           tokenize= tokenizer))
  # remove sparse terms
  n <- ncol(tdm)
  s <- 1 - (min.rows.pop/n)
  tdm <- removeSparseTerms(tdm, sparse= s)
  print(dim(t(tdm)))
  if (out.sparse) {
    return(t(sparseMatrix(i= tdm$i, j= tdm$j, x= tdm$v, dims= c(tdm$nrow, tdm$ncol), 
                          dimnames= list(dimnames(tdm)$Terms, dimnames(tdm)$Docs))))
  } else{
    as.data.frame(t(as.matrix(tdm)))
  }
}


# =========================
# Feature importance
# =========================

findFeatures <- function(text, target, nfolds= 4, ntry= 2, 
                         vec.source= TRUE, ngram.min= 1, 
                         ngram.max= 1, min.rows.pop= 10, 
                         weight= weightTf, out.sparse= TRUE,
                         family= "binomial", alpha= 0.5){
  require(glmnet)
  require(parallel)
  words <- lapply(1:ntry, function(i){
    cat("Run", i, "of", ntry, "\n")
    set.seed(i*sample(23:456, 1))
    folds <- rep(1:nfolds, length.out= length(target))[sample(length(target), length(target))]
    words.fold <- lapply(1:nfolds, function(j){
      crps <- prepareCorpus(x= text[folds == j], vec.source= vec.source)
      tdm <- createTDM(crps, ngram.min= ngram.min, ngram.max= ngram.max, min.rows.pop= min.rows.pop, 
                       weight= weight, out.sparse= out.sparse)
      enet.1 <- cv.glmnet(tdm, target[folds == j], family= family, alpha= alpha, nfolds= 3)
      vars <- enet.1$glmnet.fit$beta[, which(enet.1$lambda == enet.1$lambda.min)]
      vars <- vars[which(vars != 0)]
      vars <- data.frame(word= names(vars), coeff= vars, row.names= NULL)
      vars
    })
    do.call("rbind", words.fold)
  })
  words <- do.call("rbind", words)
  words <- words[order(-words$coeff),]
  words
}


# =========================
# Feature importance
# =========================

imp.words <- findFeatures(text= train$text, target= train$target, nfolds= 10, ntry= 2, vec.source= TRUE, 
                          ngram.min= 1, ngram.max= 2, min.rows.pop= 10, weight= weightTf, out.sparse= TRUE,
                          family= "binomial", alpha= 0.5)


# eof