library(tidyverse)
library(magrittr)
library(patchwork)

packageVersion("tidyverse") # Tidyverse 2.0.0
R.version.string # R version 4.4.2 (2024-10-31 ucrt)

# EAI Practicum 2.1 EDA and Model Selection.
# Version 2025.01

rm(list=ls()) # Clear working environment.
# data <- read_csv("~/Elements of AI/Data/FIFA.22.Original.Data/players_22.csv", locale = locale(encoding = "UTF-8"))
data <- read_csv("~/OneDrive/Documents/The Elements of Statistical Learning/Data/FIFA.22.Original.Data/players_22.csv", locale = locale(encoding = "UTF-8"))
data # Check if data are correctly loaded.
View(data)

#---- 1. Preprocess data. ----

# Add player classification to the table. 
# https://en.wikipedia.org/wiki/Association_football_positions

# There are two attributes that define player roles, each offering its own set of advantages and disadvantages. Can you identify what they are?

data %<>%
  mutate(role = case_when(str_detect(club_position, "^(CAM|CF|LAM|LF|LS|LW|RAM|RF|RS|RW|ST)$") ~ "Offense",
                          str_detect(club_position, "^(CDM|CM|LCM|LDM|LM|RCM|RDM|RM)$") ~ "Midfielders",
                          str_detect(club_position, "^(CB|LB|LCB|LWB|RB|RCB|RWB)$") ~ "Defense",
                          club_position == "GK" ~ "GK",
                          club_position == "RES" ~ "Reserve",
                          club_position == "SUB" ~ "Substitute", 
                          TRUE ~ "Unknown")) %>%
  relocate(c(role, club_position), .after = player_positions) %>%
  filter(!role %in% c("Unknown", "Reserve", "Substitute"))

data # Quick check. 
View(data)

#---- 1.1 Quick EDA for inspiration. ----

library(pheatmap)

# We will use heatmaps, which are an excellent source of information, especially when no specific idea what to model comes to mind.

set.seed(124) # Needed for experimental reproducibility. 

# Heatmaps can be useful for visualizing large datasets, but they may not always be the best choice.
# Visualizing a smaller subset of data can often be more informative.
# Therefore, create an experimental subset of data, as heatmaps are not well-suited for displaying large labeled datasets.

# Focus on the top tier players by selecting 40 players for better visualization.

data.heatmap <- data %>%
  filter(overall > 80) %>% 
  filter(role %in% c("Offense", "Midfielders", "Defense", "GK")) %>%
  slice_sample(n = 40) %>% 
  select(short_name, role, 45:73)

dim(data.heatmap) # Subset containing 31 predictors for 40 randomly chosen players. Selected players are elite players in each category.

# Data preparation.

role.labels <- data.heatmap$role
role.labels

player.names <- data.heatmap$short_name
player.names

labels <- data.frame(role.labels)
rownames(labels) <- player.names
labels <- rename(labels, Role = role.labels)
labels

# pheatmap() requires a matrix format instead of a tibble or data frame.

data.heatmap <- as.matrix(data.heatmap[,3:30]) 
data.heatmap <- t(data.heatmap)
data.heatmap

colnames(data.heatmap) <- player.names
data.heatmap # <- Final data, not exactly tidy:
#                                             * Player/observations in columns
#                                             * Skills/attributes in rows.  

# Running pheatmap().

pheatmap(data.heatmap, scale="row", 
         annotation_col = labels,
         annotation_colors=list(Role=c(Offense="#f22a2a", Midfielders="#ffdbac", Defense = "#b3cde0", GK="grey")),
         cutree_cols=4,
         color=colorRampPalette(c("navy", "white", "red"))(50))

# Among the four categories, goalkeepers can be distinctly differentiated by most attributes (except of reactions).
# Movement, acceleration, sprint speed, reactions, ball dribbling, free kick accuracy, agility, and balance are key traits for attackers.
# Conversely, interceptions, power jumping, and heading accuracy are crucial traits for defenders.

#---- 2. Examine the response variable for regression setting. ----

# We are going to regress value_eur onto one or more skills.
# walue_eur ~ skill1 + skill2 + ... + skilln

typeof(data$value_eur) 
hist(data$value_eur) # Wage_eur is a continuous variable with a highly skewed distribution towards smaller values.  

# Histograms
theme_set(theme_bw() + 
            theme(panel.grid.major = element_blank(), 
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank(),
                  axis.line = element_line(colour = "grey"))) 


create.histogram <- function(data, player_role, x_limit = 50000000) {
  data %>%
    select(role, value_eur) %>%
    filter(role == player_role) %>%
    ggplot(aes(x = value_eur)) +
    geom_histogram(aes(y = ..density..), colour = "grey", fill = "grey", binwidth = 5000) +
    geom_density(color = "#005b96", linewidth = 0.5) +
    scale_x_continuous(labels = scales::comma, limits = c(0, x_limit)) + 
    scale_y_continuous(labels = scales::comma, limits = c(0, 0.000002)) +
    labs(title = paste(player_role),
         subtitle = "Player Valuation in Euro") +
    theme(legend.position="none",
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          axis.title.y = element_blank(),
          axis.title.x = element_blank(),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(y = guide_axis(cap = "both"),
           x = guide_axis(cap = "both"))
}

hist.offense <- create.histogram(data, "Offense", max(data$value_eur)) # Create a histograms of value for offensive players and save it as an object.
hist.defense <- create.histogram(data, "Defense", max(data$value_eur))
hist.midfielders <- create.histogram(data, "Midfielders", max(data$value_eur))

# Box plots

create.boxplot <- function(data, y_attr, y_limit) {
  data %>%
    select(role, {{y_attr}}) %>%
    ggplot(aes(x = role, y = {{y_attr}} , fill = role)) +
    labs(title = "",
         subtitle = paste("Capped at:", scales::comma(y_limit / 1e6), "Million(s)")) +
    geom_boxplot(width = 0.7, 
                 outlier.shape = 1, 
                 outlier.fill = NA,
                 outlier.size = 3, 
                 outlier.alpha = 0.1,
                 outliers = TRUE, staplewidth = 0.5) +
    scale_fill_manual(values=c("#b3cde0", "lightgrey", "lightgrey","#f65858" )) +
    scale_y_continuous(labels = scales::comma, limits = c(0, y_limit)) +
    stat_summary(fun = mean, 
                 geom = "point", 
                 shape = 18, size = 3, color = "black", fill = "black") +
    stat_summary(fun = median, geom = "text", 
                 aes(label = round(..y.., 1)), vjust = -1.9, hjust = -0.1, size = 3, color = "black") +
    theme(legend.position="none",
          axis.text.x = element_text(size = 13),
          axis.text.y = element_text(size = 10),
          axis.title.y = element_blank(),
          axis.title.x = element_blank(),
          panel.border = element_blank(),
          axis.ticks.x = element_blank(),
          axis.line.x = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(y = guide_axis(cap = "both"))
}

boxplot.value_eur <- create.boxplot(data, value_eur, 16000000) # Create a boxplot and save it as an object.

# An average offensive player valuation is about 16M, defense player 12M and 14M for midfielders. 
# Response variable is highly skewed towards smaller values, definitely not gaussian, or gaussian with many outliers at least.

# Mash-up of all visualizations.
hist.offense + hist.midfielders + hist.defense + boxplot.value_eur + plot_layout(ncol = 2, nrow = 2) # Implemented by patchworks.

#---- 3. Examine how various skills influence a player's wage. ----

# Helper functions for pairs().
panel.cor <- function(x,y, digits=2, prefix="", cex.cor){
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(0,1,0,1))
  r <- abs(cor(x,y,use="complete.obs"))
  txt <- format(c(r,0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt,sep="")
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * (1 + r) / 2)
}
panel.hist <- function(x, ...){
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2],0,1.5))
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks
  nB <- length(breaks)
  y <- h$counts
  y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col="white",...)
}

# Plot multivariate data with regression splines, correlation coefficients and histograms.
data %>%
  filter(role == "Offense") %$% # Change/delete role to inspect skill relations for different player roles.
  pairs( ~ wage_eur + overall + skill_dribbling + skill_curve + skill_fk_accuracy +
           skill_long_passing + skill_ball_control,       
         upper.panel= panel.cor, 
         diag.panel=panel.hist,
         lower.panel=panel.smooth)

#---- 4. Examine predictor variables for regression setting. ----

# Independence: 19239 players represent independent observations.
# Model selection: Player wages depend on their skills, but the relationship is far from linear -> polynomial fit is the best option to minimize model error. 

# Co-linearity: All predictors are co-linear with each other and only loosely correlated with wage. 
#               To address this, selecting a single predictor/or meta-predictor will suffice: dribbling or ball control are excellent choices for offense, 
#               while long passes are ideal for defense. 
#               Methods: scatter plots, correlation matrix, variance inflation factor (model needed), condition index (model needed).

# Calculate the correlation matrix for skills vs. player roles.
# Choose predictors for co-linearity testing. Pearson's correlation will be used as the metric. 

columns <- c("wage_eur", "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control")

corSkills <- function(data, position){
  data = data
  data %>%
    filter(role == position) %>%
    select(all_of(columns)) %>%
    na.omit() %>%
    cor() # Uses Person's correlation by default. 
}

skills <- list("Offense", "Defense", "Midfielders") # Create a list of player positions.
results <- map(skills, ~ corSkills(data, .x)) # Iterate corSkill() over data.
names(results) <- skills
results # Inspect data.

# Normality: Dependent variable: Normality is not required. 
#            Predictors: Normality is not required, but skill_dribbling and skill_ball_control are normally distributed.

# The simplest method to test for normality is by visually inspecting histograms of the selected predictors' values. 
# Alternatively, the Kolmogorov-Smirnov test can be used for a more formal assessment.

normality <- function(data, position, skill) {
  data %>%
    filter(role == position) %>%
    pull({{ skill }}) %>%
    unique() %>%
    ks.test("pnorm", mean = mean(.), sd = sd(.))
}

test_normality <- function(data) {
  skills <- c("skill_dribbling", "skill_long_passing", "skill_ball_control")
  positions <- c("Offense", "Defense")
  
  results <- map(positions, function(position) {
    map(skills, function(skill) {
      normality(data, position, !!sym(skill))
    })
  })
  
  names(results) <- positions
  results <- map(results, set_names, skills)
  
  return(results)
}

test_normality(data) # Display the results. In every instance, the D value is small, and the P value is significant.

# Feature scaling: Dependent variable: Scaling is not required.
#                  Predictor variables: The skills are already scaled to the same interval. [0-100]

#---- 5. Homoscedasticity. ----

# Homoscedasticity refers to the assumption that the variance of the errors (or residuals) in a regression model is constant across all levels of the independent variables.
# To evaluate it for a regression model with `value_eur` as the dependent variable and skills as predictors, you must first construct the model.

# How to test: Residuals vs. fitted plot.
#              Residual Q-Q plot

#---- 6. Checking for outliers. ----

# Among the various skills, we are focusing on either overall or skill dribbling. Let's evaluate the performance of skill dribbling.

# Visual methods: Box plots.
#                 Scatter Plots
#                 Histograms

# Define the function
create.histogram <- function(data, player_role, y_attr) {
  
  # Calculate mean and standard deviation
  stats <- data %>%
    filter(role == player_role) %>%
    summarise(mean = mean(!!sym(y_attr), na.rm = TRUE),
              sd = sd(!!sym(y_attr), na.rm = TRUE))
  
  mean_val <- stats$mean
  sd_val <- stats$sd
  
  # Define the limits for 3 sigmas
  lower_limit <- mean_val - 3 * sd_val
  upper_limit <- mean_val + 3 * sd_val
  
  data %>%
    filter(role == player_role) %>%
    ggplot(aes(x = !!sym(y_attr))) +
    geom_histogram(aes(y = ..density..), colour = "grey", fill = "grey") +
    geom_density(color = "#005b96", size = 0.5) +
    geom_vline(xintercept = lower_limit, linetype = "dashed", color = "darkgrey") + 
    geom_vline(xintercept = upper_limit, linetype = "dashed", color = "darkgrey") +
    labs(title = paste("Player Role:", player_role),
         subtitle = paste("Distribution of", y_attr, "score.")) +
    theme(legend.position="none",
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          axis.title.y = element_blank(),
          axis.title.x = element_blank(),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(y = guide_axis(cap = "both"),
           x = guide_axis(cap = "both"))
}

hist.offense <- create.histogram(data, "Offense", "skill_dribbling") # Create a histograms of value for offensive players and save it as an object.
hist.defense <- create.histogram(data, "Defense", "skill_dribbling")
hist.midfielders <- create.histogram(data, "Midfielders", "skill_dribbling")

boxplot.value_eur <- create.boxplot(data, skill_dribbling, 100) # Create a boxplot and save it as an object.
hist.offense + hist.midfielders + hist.defense + boxplot.value_eur + plot_layout(ncol = 2, nrow = 2) # Implemented by patchworks.

# Z-scores:
# A z-score, also known as a standard score, measures how many standard deviations a data point is from the mean of the dataset. 
# It allows you to understand the position of a specific value within the distribution of the data.

# Calculate Z-scores for all players
z_scores <- scale(data$skill_dribbling, center = T, scale = T) # Convert normally distributed data to the Z-distributed data with centering and scaling.  

# Identify outliers (e.g., Z-score > 3 or < -3)
length(which(abs(z_scores) > 3)) 
length(which(abs(z_scores) > 3)) / dim(data)[1] 

# In skewed distributions, the mean and standard deviation may not be representative of the data's central tendency and spread, 
# leading to potentially misleading z-scores.

# Alternatives: IQR method
#               Cook's distances -> A fitted model is required.

#---- 7. Sample Size. ----

library(pwr)

# A rule of thumb is: Regression: 10-15 observations per predictor variable -> requirement met. 
#                     Classification: 10x more observations as there are features -> requirement met.
#                                     It is recommended that classes should be balanced, but not strictly required.                                       

# Power analysis.

# Parameters for power analysis
f2 <- 0.15  # Effect size (Cohen's f^2) [0.3=small, 0.15-medium, 0.30=large]
u <- 2      # Number of predictors
sig.level <- 0.05  # Significance level
power <- 0.80  # Desired power


# Perform power analysis
pwr.f2.test(u = u, f2 = f2, sig.level = sig.level, power = power)

# Cross validation -> model needed.

#---- 8. Linear Relationship. ----
# Scatter plots -> co-linearity testing: Correlation matrix
#                                        VIF -> model needed.
#                                        Condition index -> model needed.

# Residual plots -> model needed.
# Lack of fit testing -> model needed.
# crPlots -> model needed. [https://rkabacoff.github.io/qacReg/reference/cr_plots.html]


