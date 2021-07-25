get_all_AICs <- function(model_string, features) {
  n_models <- 16
  all_AICs <- matrix(NA, ncol = n_models, nrow = length(features))
  rownames(all_AICs) <- features
  for (feature in features) {
    try(
      all_results <- readRDS(file = paste("Results/results_", feature, "_", model_string, ".RDS", sep = ""))
    )
    try(
      all_AICs[feature,] <- all_results$AICs
    )
    rm(all_results)
  }
  return(all_AICs)
}