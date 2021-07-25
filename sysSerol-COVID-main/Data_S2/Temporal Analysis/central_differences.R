# Central difference for function depending on multiple parameters
# Checks each parameter and outputs the derivative vector
# Parameters:
# f: function to be evaluated
# theta: parameter values at which the gradient should be approximated via central finite differences
# h: step size for finite differences

central_differences <- function(f, theta, h) {
  fdx <- rep(NA, length = length(theta))
  for (iParam in 1:length(theta)) {
    theta_tmp1 <- theta
    theta_tmp1[iParam] <- theta_tmp1[iParam] + 0.5 *h
    theta_tmp2 <- theta
    theta_tmp2[iParam] <- theta_tmp2[iParam] - 0.5 *h
    fdx[iParam] <- (f( theta_tmp1) - f(theta_tmp2)) / h
  }
  return(fdx)
}