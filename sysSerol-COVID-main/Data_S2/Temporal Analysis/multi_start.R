# Randomly draws initial parameters values and performs constrained optimization
# using BFGS, returns ordered parameter vectors and log-likelhood values

# llh: log-likelihood function
# llh_grad: gradient log-likelihood function
# data: data
# n_param: number of parameters to be estimated
# n_starts: number of starts
# lb: vector of lower bounds
# ub: vector of upper bounds

multi_start <- function(llh, llh_grad, n_param, n_starts, lb, ub) {
  set.seed(123) # fix set for reproducibility
  llhs <- rep(NA, length = n_starts)
  mls <- matrix(NA, nrow = n_param, ncol = n_starts)
  
  # Multi-start optimization
  for (iStart in 1:n_starts) {
    evalFailed <- TRUE
    while (evalFailed) { 
      theta0 <- runif(1, lb[1], ub[1])
      for (iParam in 2:n_param) {
        theta0 <- c(theta0, runif(1, lb[iParam], ub[iParam]))
      }
      if (length(which(is.na(llh_grad(theta0)))) == 0) {
        evalFailed <- FALSE
      }
    }
    A <- rbind(diag(n_param), -diag(n_param))
    B <- c(-lb, ub)
    ineqCon <- list(ineqA = A, ineqB = B)
    ml <- maxLik::maxLik(llh, grad = llh_grad, start = c(theta = theta0),
                 constraint = ineqCon, method = "BFGS")
    llhs[iStart] <- ml$maximum
    mls[, iStart] <- ml$estimate
  }
  mls <- mls[, order(-llhs)]
  llhs <- llhs[order(-llhs)]

  result_MS <- list(llhs = llhs, mls = mls)
  return(result_MS)
}
