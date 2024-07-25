
list.of.packages <- c("data.table", "dplyr", "magrittr", "tidyverse", "plinkFile", "genio")

lapply(list.of.packages, library, character.only = TRUE)


# bed <- readBED('/Users/mmir/Downloads/bed/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bed')


# fam <- read_fam('/Users/mmir/Downloads/bed/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.fam')

# bim <- read_bim('/Users/mmir/Downloads/bed/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bim')

# df <- readBED('/Users/mmir/Downloads/bed/out.bed')


# bed_t <- as.data.frame(t(bed))



# Load necessary library
library(data.table)

# Function to convert .bed to .csv
convert_bed_to_csv <- function(input_dir, output_dir) {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Get a list of all .bed files in the input directory
  bed_files <- list.files(input_dir, pattern = "\\.bed$", full.names = TRUE)
  
  print(bed_files)
  
  # Loop through each .bed file and convert it to .csv
  for (bed_file in bed_files) {
    # Read the .bed file
    bed_data <- readBED(bed_file)
    
    # Generate the output .csv file path
    csv_file <- file.path(output_dir, paste0(basename(tools::file_path_sans_ext(bed_file)), ".csv"))
    
    # Write the data to .csv file
    fwrite(bed_data, csv_file)
    
    # Print a message indicating the file has been converted
    cat("Converted", bed_file, "to", csv_file, "\n")
  }
}

# Example usage
input_directory <- "/homes/nber/mahdimir/plink_out"
output_directory <- "/homes/nber/mahdimir/bed_files_converted_2_csv"
convert_bed_to_csv(input_directory, output_directory)


fn <- "/Users/mmir/Library/CloudStorage/Dropbox/sibs_model_data.parquet"


library(arrow)

df <- read_parquet(fn)


# Fit the OLS model
# For example, let's predict 'mpg' (miles per gallon) based on 'wt' (weight) and 'hp' (horsepower)
model <- lm(g1_plus_g2 ~ g1_minus_g2_hat, data = df)

# Summarize the model
summary(model)


# For example, let's predict 'mpg' (miles per gallon) based on 'wt' (weight) and 'hp' (horsepower)
model <- lm(g1_minus_g2 ~ g1_minus_g2_hat, data = df)

# Summarize the model
summary(model)






