# Add colored parameters that are different in the left upper corner of the plot

annotate_parameters <- function(plt, param_names, colors) {
  for (ind in 1:length(param_names)) {
    plt <- plt + geom_text(x = 0.5, y = 0.95 - (ind - 1)*0.13,
                           label = param_names[ind], size = 2.5,
                           color = colors[[ind]])
  }
  return(plt)
}
