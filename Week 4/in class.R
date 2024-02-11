setwd("C:/Users/josel/OneDrive - The University of Chicago/Clases/2nd Q/Linear and non linear/4w")

dta = read.csv("binomial_regression.csv")


y <- matrix(dta$y,ncol=1)
#X <- matrix(dta[,-1],nrow = nrow(dta), ncol= (ncol(dta)-1), byrow = FALSE)

X <- as.matrix(dta[,-1])

#binomial parameters

n=1  #turns into a bernoulli

# mean_i = p_i
# so the link function would be a logit
# log(p/(1-p)) = x_i*b = eta_i
# pi = exp(x_i*b)/(1+exp(x_i*b))



b0 <- c(0,0,0,1)
tol <- 1e-5
err <- 1e10
ite <- 0


while (err>tol){
  
  eta = X %*% b0
  mu = exp(X %*% b0)/(1+exp( X %*% b0))
  
  var.y <- exp(X %*% b0)/(1+exp( X %*% b0)) * ( 1 - exp(X %*% b0)/(1+exp(X %*% b0)) ) 
  
  dudeta <- exp( X %*% b0) /((1+ exp(X %*% b0))^2)
  detadmu <- 1/(  exp(X %*% b0)/(1+exp( X %*% b0)) - (exp(X %*% b0)/(1+exp( X %*% b0)))^2 )
  
  W <- diag(as.numeric(1/var.y * dudeta^2 ),nrow=length(var.y),ncol=length(var.y))
  z <- eta + (y - mu) * detadmu
  
  b1 <- solve(t(X) %*% W %*% X) %*% (t(X) %*% W %*% z)
  
  err <- max(abs(b1-b0))
  #dta <- rbind(dta,data.frame(ite,b0,b1,err,stringsAsFactors=FALSE))
  
  b0 <- b1
  ite <- ite + 1
  
}




regdta <- data.frame(y=y,X=X[,-1])

print(glm(y~.,data=regdta,family=binomial(link = "logit"))$coefficients)
