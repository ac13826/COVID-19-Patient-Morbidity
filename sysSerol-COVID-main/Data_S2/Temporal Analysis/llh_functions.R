# Fit two groups together using a laplace noise distribution

# Log-likelihood function
llh_laplace_2group <- function(curveFun, data_A, data_B, indA, indB, indNoise, theta) {
  b <- 10^theta[indNoise]
  out <- -log(2*b)*(dim(data_A)[1] + dim(data_B)[1]) - 
    sum(abs(curveFun(data_A$t, theta[indA], logFlag = TRUE) - data_A$x))/b -
    sum(abs(curveFun(data_B$t, theta[indB], logFlag = TRUE) - data_B$x))/b
  return(out)
}

# Gradient of log-likelihood function
llh_laplace_2group_grad <- function(curveFun, curveFun_grad, data_A, data_B, 
                            indA, indB, indNoise, theta) { 
  out_grad <- rep(0, length = length(theta))
  
  # Gradient for noise parameter
  b <- 10^theta[indNoise]
  out_grad[indNoise] <- -(dim(data_A)[1] + dim(data_B)[1])/b +
    sum(abs(curveFun(data_A$t, theta[indA], logFlag = TRUE) - data_A$x))/b^2 +
    sum(abs(curveFun(data_B$t, theta[indB], logFlag = TRUE) - data_B$x))/b^2
  out_grad[indNoise] <- out_grad[indNoise]*b*log(10)
  
  # Group A
  ab_grad <- curveFun_grad(data_A$t, theta[indA], logFlag = TRUE) 
  part1 <- sign(curveFun(data_A$t, theta[indA], logFlag = TRUE) - data_A$x)
  for (iParam in 1:length(indA)) {
    out_grad[indA[iParam]] <- out_grad[indA[iParam]] - sum(ab_grad[,iParam] * part1)/b
  }
  
  # Group B
  ab_grad <- curveFun_grad(data_B$t, theta[indB], logFlag = TRUE) 
  part1 <- sign(curveFun(data_B$t, theta[indB], logFlag = TRUE) - data_B$x)
  for (iParam in 1:length(indB)) {
    out_grad[indB[iParam]] <- out_grad[indB[iParam]] - sum(ab_grad[,iParam] * part1)/b
  }
  return(out_grad)
}

