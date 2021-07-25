# This function returns the indices mapping the overall parameter vector which is
# estimated from the data, to the parameters used for the curve of each group
#
# indAs: list of indices indicating which parameters map to the curve parameters of the first group (A), here, these are the first parameters
# indBs: list of indices indicating which parameters map to the curve parameters of the second group (B)
# indNoises: list of indices indicating which parameter is the noise parameter, shared between the groups
#
# The list has n_models entries, with n_models being the number of possible combinations of differences in the curve parameters

get_indices_groups <- function(n_curve_param) {

  diffBs <- list() # list of inddices with differences in curve parameters, these entries thus have values between 1 and 5
  diffBs[[1]] <- c() # the first entry of the list is the model without any differences
  
  # fill the list for the remaining possible models
  count_diff <- 2
  for (ind_diff in 1:n_curve_param) { # number of different parameters, 0 is covered above
    for (ind_diff2 in 1:dim(combn(n_curve_param, ind_diff))[2]) { # all combination of differences of size ind_diff
      diffBs[[count_diff]] <- combn(n_curve_param, ind_diff)[,ind_diff2]
      count_diff <- count_diff + 1
    }
  }
  
  # map the differences to the indices of the overall parameter vector
  # (which includes all parameters that are estimated from the data)
  indAs <- list()
  indBs <- list()
  indNoises <- list()
  for (ind_model in 1:length(diffBs)) {
    indAs[[ind_model]] <- c(1:n_curve_param) # the first group always has the first 5 parameters
    tmp_ind <- c(1:n_curve_param)
    if (length(diffBs[[ind_model]]) > 0) {
      tmp_ind[diffBs[[ind_model]]] <- seq(n_curve_param + 1, (n_curve_param + length(diffBs[[ind_model]])))
    }
    indBs[[ind_model]] <- tmp_ind
    indNoises[[ind_model]] <- max(indBs[[ind_model]]) + 1 # add an additional parameter for the noise
  }
  
  indices <- list(indAs = indAs, indBs = indBs, indNoises = indNoises)
  return(indices)

}