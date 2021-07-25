# Logistic growth curve with 4 parameters
abKinetic_4 <- function(t, theta, logFlag = FALSE) {
  # log transformed parameter are better for optimization
  if (logFlag) {
    theta <- 10^theta
  }
  return(theta[4] + ((theta[1] - theta[4]) / ((1 + ((t / theta[3])^theta[2])))))
}

# Derivative of logistic growth curve 
abKinetic_4_grad <- function(t, theta, logFlag = FALSE) {
  if (logFlag) {
    theta <- 10^theta
  }
  
  out_grad <- matrix(0, nrow = length(t), ncol = 4)
  
  out_grad[,1] <- ((theta[3]/t)^(-theta[2]) + 1)^(-1)
  out_grad[,2] <-  -(theta[1] - theta[4]) * (t/theta[3])^theta[2] * log(t/theta[3]) * ((t/theta[3])^theta[2] + 1)^(-2)
  out_grad[,3] <- (theta[2] * (theta[1] - theta[4]) * (t/theta[3])^theta[2] * ((t/theta[3])^theta[2] + 1)^(-2))/theta[3]
  out_grad[,4] <-  1 - ((t/theta[3])^theta[2] + 1)^(-1)

  out_grad[which(t == 0), 2] <- 0
  
  if (logFlag) { # chain rule
    for (iParam in 1:4) {
      out_grad[,iParam] <- out_grad[,iParam] * (theta[iParam] * log(10))
    }
  }
  return(out_grad)
}

