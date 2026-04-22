library(tidyverse)
library(magrittr)
library(patchwork)

packageVersion("tidyverse") # Tidyverse 2.0.0
R.version.string # R version 4.4.2 (2024-10-31 ucrt)

# EAI Practicum 2.2 Predictive Analsmysis with Regression.
# Version 2025.03

rm(list=ls()) # Clear working environment.
data <- read_csv("~/OneDrive/Documents/The Elements of Statistical Learning/Data/FIFA.22.Original.Data/players_22.csv", locale = locale(encoding = "UTF-8"))
#data <- read_csv("~/OneDrive/Documents/Elements of AI/Data/FIFA.22.Original.Data/players_22.csv", locale = locale(encoding = "UTF-8"))
data # Check if data are correctly loaded.

#---- 1. Preprocess data. ----

# Add player classification to the table. 

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

#---- 2. Hypothesis formulation. -----

# Hypothesis-driven approach.

# 1. Select players playing in offense. 
# 2. Predict player's value or player overall rank from his stats.
# But from which attributes?

#---- 2.1 Choosing the right predictors: hypothesis-free approach. ---- 

# We will use heatmaps, which are an excellent source of information, especially when no specific idea comes to mind.

library(pheatmap)

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

#---- 2.2 Choosing the right predictors: hypothesis-driven approach. ----

# 2. Predict player's value or player overall rank from his stats.
#                   Examples: overall  ~  speed and dribbling
#                                         passing and vision
#                                         strength and heading accuracy
#                                         stamina and aggression
#                                         finishing and composure


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
  pairs( ~ overall + 
           movement_sprint_speed + skill_dribbling +
           attacking_short_passing + mentality_vision + 
           power_strength + attacking_heading_accuracy +
           power_stamina + mentality_aggression,
         upper.panel= panel.cor, 
         diag.panel=panel.hist,
         lower.panel=panel.smooth)

#---- 2.3 Requirement analysis.  ----

# 1. Choose predictors and dependent variable(s).

# Performance on the field: overall <- will act as a depending (target) variable.
# Player's skill set: skill_dribbling and movement_sprint_speed <- will act as predictor variables. 

# Hypothesis formulation: overall ~ skill_dribbling (+ movement_sprint_speed)

# 2. Assess whether the relationship is linear or polynomial.

# Set parameters for empty canvas.

theme_set(theme_bw() + 
            theme(panel.grid.major = element_blank(), 
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank(),
                  axis.line = element_line(colour = "grey"))) 

# Create a template for a plotting function. 

create_plot <- function(data, x, y, color_factor, plot_title) {
  data %>%
    ggplot(aes(x = {{ x }}, y = {{ y }}, colour = as_factor({{ color_factor }}), group = as_factor({{ color_factor }}))) + 
    geom_point(shape = 16, alpha = 0.1, size = 2) + 
    scale_color_manual(values = c("#b3cde0", "#f22a2a", "grey", "#ffdbac")) +
    labs(title = plot_title, 
         subtitle = "",
         x = deparse(substitute(x)),
         y = deparse(substitute(y))) + 
    scale_y_continuous(labels = scales::comma) +
    theme(legend.position="none",
          legend.title = element_blank(),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(x = guide_axis(cap = "both"),
           y = guide_axis(cap = "both"))
}

# Plot data. Save them into objects to be arranged with patchwork. sp = scatter plot  

sp.ball.dribbling <- create_plot(data %>% filter(role == "Offense"), skill_dribbling, overall, role, "Scatter Plots")
sp.sprint.speed <- create_plot(data %>% filter(role == "Offense"), movement_sprint_speed, overall, role, "")
sp.dribbling.speed <- create_plot(data %>% filter(role == "Offense"),movement_sprint_speed , skill_dribbling, role, "")

# Visualize data using patchworks. 

sp.ball.dribbling + sp.sprint.speed + sp.dribbling.speed + plot_layout(ncol = 3, nrow = 3)

# Conclusions: 

# 1. Even before constructing the model, it appears that a player's ball dribbling skill has
# linear correlation with their overall performance on the field.

# 2. sprint speed and skill dribbling are not correlated, so we can be used in multiple linear regression with interaction.

# 3. Outliers: even without proper analysis, it seems that there are no outliers, or high leverage points. 

# 4. Assess whether predictor values are normally distributed. This is not necessary for OLR, but still can be part of the analysis. 

# Create histograms.
create.histogram <- function(data, x, player_role, title, label, x_limit = 100) {
  data %>%
    filter(role == player_role) %>%
    ggplot(aes(x = {{ x }})) +
    geom_histogram(aes(y = ..density..), colour = "lightgrey", fill = "lightgrey", bins = 30) +
    geom_density(color = "navyblue", linewidth = 0.8) +
    labs(title = title, 
         x = label,
         y = "Density") +
    theme(legend.position = "none",
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(y = guide_axis(cap = "both"),
           x = guide_axis(cap = "both"))
}

hist.ball.dribbling <- create.histogram(data %>% filter(role == "Offense"), skill_dribbling, "Offense", "", "Ball Dribbling")
hist.sprint.speed <- create.histogram(data %>% filter(role == "Offense"), movement_sprint_speed, "Offense", "",  "Sprint Speed")
hist.overall <- create.histogram(data %>% filter(role == "Offense"), overall, "Offense", "Histograms", "Overall Performance")

# Create Q-Q plots. 

create.qqplot <- function(data, x, label) {
  ggplot(data, aes(sample = {{ x }})) +
    stat_qq_line(color = "lightgray") +
    stat_qq(color = "navyblue") +
    labs(title = label,
         x = "Theoretical Quantiles", 
         y = "Sample Quantiles") +
    theme(legend.position = "none",
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(y = guide_axis(cap = "both"),
           x = guide_axis(cap = "both"))
}

qq.overall <- create.qqplot(data %>% filter(role == "Offense"), overall, "Q-Q Plots")
qq.ball.dribbling <- create.qqplot(data %>% filter(role == "Offense"), skill_dribbling, "")
qq.sprint.speed <- create.qqplot(data %>% filter(role == "Offense"), movement_sprint_speed, "")

# Mash up EDA panels.

hist.overall + hist.ball.dribbling + hist.sprint.speed +
  qq.overall + qq.ball.dribbling + qq.sprint.speed + plot_layout(ncol = 3, nrow = 3)

# There are more players with smaller sprinting speed values than expected, 
# indicating the presence of a subgroup of players. It's intriguing to consider what role they might be playing at.

# Conclusions:
#             * Each attribute conforms to the normal distribution, which can be further demonstrated by calculating z-scores.
#             * The relationship between overall and ball dribbling is linear, whereas the relationship between overall and sprint speed is less so.
#             * Ball dribbling and sprint speed should be modeled both with and without interaction to determine which approach yields better results.

# 5."Independence: Since observations are represented by individual players, we can consider them fully independent.

#---- 3. Building linear regression model: overall <- ball_dribbling + sprint_speed ----
#---- 3.1 Data spending with rsample. ----

library(rsample) # package of Tidymodels
library(broom) # package of Tidymodels

set.seed(123) # For reproducibility 
split <- initial_split(data %>% filter(role == "Offense"), prop = 0.8) # 80% training, 20% testing

split 
# <Training/Testing/Total>
# <1362/341/1703>

View(split)

# Extract training and testing sets.

train.data <- training(split) # 80% goes into training.
test.data <- testing(split) # 20% goes into testing. 
dim(train.data) # 1362
dim(test.data) # 341

#---- 3.2 Building simple OLR model. ----

# Do not run model without capturing it.

lm(overall ~ skill_dribbling, data = data %>% filter(role == "Offense")) # ls() produces an object of type lm, but does not saves it.
View(lm(overall ~ skill_dribbling, data = data  %>% filter(role == "Offense")))

lm.simple <- data  %>% 
  filter(role == "Offense") %$% 
  lm(overall ~ skill_dribbling) # Multiple R-squared:  0.08902,	Adjusted R-squared:  0.0889

lm.simple
class(lm.simple) # [1] "lm"
typeof(lm.simple) # [1] "list"
summary(lm.simple)
tidy(lm.simple)

# Residuals:
# Min       1Q   Median       3Q      Max 
# -11.7780  -2.3306  -0.3306   2.0510  16.1365  <- An average real data point will lie -0.33 pts from the fitted value. - means that predicted values will be higher.

# term            estimate std.error statistic   p.value
# <chr>              <dbl>     <dbl>     <dbl>     <dbl>
# 1 (Intercept)       25.4      0.769       33.0 5.90e-185
# 2 skill_dribbling    0.638    0.0110      57.8 0        

# Residual standard error: 3.58 on 1701 degrees of freedom <- 68% of real values will lie within 3.58 pts from the fitted value. 
# Multiple R-squared:  0.6624,	Adjusted R-squared:  0.6622 <- This model explains only 66.3% of observed variance.

#---- 3.3 Building multiple OLR model. ----

lm.multiple <- data %>% 
  filter(role == "Offense")%$% 
  lm(overall ~ skill_dribbling + movement_sprint_speed) # Multiple R-squared:  0.09816,	Adjusted R-squared:  0.09792 

summary(lm.multiple)
tidy(lm.multiple)

# Residuals:
# Min      1Q  Median      3Q     Max 
# -11.243  -2.245  -0.334   1.957  15.792  <- An average real data point will lie -0.33 pts from the fitted value. - means that predicted values will be higher.

# term                  estimate std.error statistic   p.value
# <chr>                    <dbl>     <dbl>     <dbl>     <dbl>
# 1 (Intercept)            28.3      0.830       34.1  1.79e-194
# 2 skill_dribbling         0.673    0.0116      58.1  0        
# 3 movement_sprint_speed  -0.0727   0.00868     -8.37 1.14e- 16

# Residual standard error: 3.51 on 1700 degrees of freedom <- 68% of real values will lie within 3.51 pts from the fitted value. 
# Multiple R-squared:  0.6758,	Adjusted R-squared:  0.6754 <- This model explains 67% of observed variance, which is only slightly more.  

#---- 3.4 Building multiple OLR model with interaction. ----

lm.multiple.interact <- data %>% 
  filter(role == "Offense")%$% 
  lm(overall ~ skill_dribbling * movement_sprint_speed) # Multiple R-squared:  0.2004,	Adjusted R-squared:    0.2 

summary(lm.multiple.interact)
tidy(lm.multiple.interact)

# Residuals:
# Min       1Q   Median       3Q      Max 
# -10.8309  -2.2254  -0.2978   2.0392  15.8906 <- An average real data point will lie -0.29 pts from the fitted value. - means that predicted values will be higher.

# term                                  estimate std.error statistic  p.value
# <chr>                                    <dbl>     <dbl>     <dbl>    <dbl>
# 1 (Intercept)                           66.2      4.59         14.4  1.29e-44
# 2 skill_dribbling                        0.107    0.0683        1.57 1.17e- 1
# 3 movement_sprint_speed                 -0.597    0.0630       -9.47 8.63e-21
# 4 skill_dribbling:movement_sprint_speed  0.00777  0.000926      8.40 9.41e-17

# Residual standard error: 3.44 on 1699 degrees of freedom <- 68% of real values will lie within 5.3 pts from the fitted value. 
# Multiple R-squared:  0.6887,	Adjusted R-squared:  0.6882 <- This model explains 69% of observed variance.

# Summary:
# 1. The three models do not differ significantly.
# 2. The linear regression model with interaction terms explains the highest amount of variance in the data.
# 3. Omitting sprint speed slightly diminishes the model's predictive accuracy.
# 4. Each model tends to overestimate the actual values.
# 5. In all models, 70% of the predictions will have an error margin of no more than 3%.
# 6. The residuals seem to follow a normal distribution, satisfying the homoscedasticity assumption.
# 7. The Pearson correlation coefficient indicates that the target variable might have a linear dependency on the predictors, showing a positive relationship.

#---- 4. Residuals and what can we learn from them. ----

# Goal: Ensure that your model predicts effectively for both small and large values.

# You can use the following methods to achieve this:
#                                                   Homoscedasticity
#                                                   Outliers
#                                                   Normality of residuals

par(mfrow = c(2,2))

# The plot(lm.model) function in R is utilized to create diagnostic plots for a linear model object produced by the lm() function.
# These plots are useful for evaluating the validity of model assumptions and detecting any potential issues.

plot(lm.simple)

# Residuals vs Fitted: This plot helps check the linearity assumption. 
# Ideally, the residuals should be randomly scattered around the horizontal line (y = 0), indicating a good fit.

# Normal Q-Q: This plot helps check the normality assumption of the residuals. 
# If the residuals are normally distributed, the points should fall approximately along the reference line.

# Scale-Location (or Spread-Location): This plot helps check the homoscedasticity assumption (constant variance of residuals). 
# The residuals should be spread equally along the range of fitted values.

# Residuals vs Leverage: This plot helps identify influential observations. 
# Points that stand out from the rest may have a disproportionate impact on the model.

# In the context of linear models (lm), leverage refers to the influence that a particular data point has on the estimation of the regression coefficients. 
# It measures how far an independent variable deviates from its mean. High leverage points can significantly affect the fit of the model, making them important to identify and assess.

# In a linear regression model, leverage values are calculated for each data point, and they range between 0 and 1. 
# Points with high leverage have a greater potential to influence the regression line. These points are often located far from the center of the data distribution in the space of the independent variables.


#---- 5. Predict new values using one of the models. ----

# Typically, you can use a pre-generated model to predict values of a dependent variable from new observations. 
# You need to supply the new data in the same format as was used for training.

# New data for prediction
new_data <- data.frame(skill_dribbling = c(70, 80, 90))

# Predicting new data
predictions <- predict(lm.simple, newdata = new_data)

# Display predictions
print(predictions)

#---- 6. Cross-validate. ----

# 1. Inspect training and testing data.  

# Create a custom function for plotting. Quosure, in fact.

create_plot.2 <- function(data, x, y, plot_title) {
  data %>%
    ggplot(aes(x = {{ x }}, y = {{ y }})) + 
    geom_point(shape = 16, alpha = 0.1, size = 2) + 
    labs(title = plot_title,
         x = deparse(substitute(x)),
         y = deparse(substitute(y))) + 
    scale_y_continuous(labels = scales::comma) +
    theme(legend.position="none",
          legend.title = element_blank(),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(x = guide_axis(cap = "both"),
           y = guide_axis(cap = "both"))
}

# Plot data. Save them into objects to be arranged with patchwork. sp = scatter plot  

sp.training <- create_plot.2(train.data, skill_dribbling, overall, "Training Data")
sp.testing <- create_plot.2(test.data, skill_dribbling, overall, "Training Data")

# Visualize data using patchworks. 

sp.training + sp.testing + plot_layout(ncol = 2, nrow = 1)

# 2.Define the custom function for linear regression with cross-validation.

build.model <- function(data_list, predictor.col, depval.col) {
  
  model.list <- map(data_list, function(data) { 
    formula <- as.formula(paste(depval.col, "~", predictor.col)) 
    lm(formula, data = data)
  })
  
  return(model.list)
}

# This code defines a function build.model that takes three arguments: data_list, predictor_col, and dep_val_col. 
# Here's a breakdown of what the function does:

# A. Input Parameters:
#  - data_list: A list of data frames.
#  - predictor_col: The name of the predictor (independent variable) column.
#  - dep_val_col: The name of the dependent (response) variable column.

data.list <- list(train.data, test.data)
predictor.col <- "skill_dribbling"
depval.col <- "overall"

# B. Function call:
#  - The function uses map from the purrr package to iterate over each data frame in the data_list.
#  - For each data frame in the list, it constructs a formula for linear regression using the specified predictor and dependent variable columns.
#  - It then fits a linear model (lm) using the constructed formula and the data frame.

models <- build.model(data.list, predictor.col, depval.col)
View(models)

# C. Return value: The function returns a list of fitted linear models, one for each data frame in the data_list. 

# 3. Inspect model parameters.

map_df(models, coef) # Get model coefficients.
map(models, tidy) # Get model coefficients with error estimation and statistics.
map(models, summary) # Get classical summaries printed on a screen. 

create_plot.3 <- function(data, x, y, intercept, slope, plot_title) {
  data %>%
    ggplot(aes(x = {{ x }}, y = {{ y }})) + 
    geom_point(shape = 16, alpha = 0.1, size = 2) +
    geom_abline(intercept = intercept, slope = slope, color = "#b3cde0") + 
    labs(title = plot_title,
         x = deparse(substitute(x)),
         y = deparse(substitute(y))) + 
    scale_y_continuous(labels = scales::comma) +
    theme(legend.position="none",
          legend.title = element_blank(),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          panel.border = element_blank(),
          axis.line = element_line(colour = "grey"),
          axis.ticks = element_line(color = "grey")) + 
    guides(x = guide_axis(cap = "both"),
           y = guide_axis(cap = "both")) +
    annotate("text", x = Inf, y = Inf, label = paste("Slope:", round(slope, 2), "\nIntercept:", round(intercept, 2)), 
             hjust = 1, vjust = 1, size = 4, color = "black", 
             angle = 0, fontface = "italic")
}

sp.training <- create_plot.3(train.data, skill_dribbling, overall, 24.6, 0.648, "Training Data")
sp.testing <- create_plot.3(train.data, skill_dribbling, overall, 28.6, 0.593, "Training Data")

sp.training + sp.testing + plot_layout(ncol = 2, nrow = 1)

# 4. Formally compare models by extracting and analyzing residuals. 

residuals.train <- resid(models[[1]]) 
residuals.test <- resid(models[[2]]) 

calculate.rmse <- function(residuals) {
  sqrt(mean(residuals^2))
}

rsme <- map(list(residuals.train, residuals.test), ~ calculate.rmse(.))
names(rsme) <- c("Train.RSME", "Test.RSME") 
rsme

# 5. Visualize residuals. 

boxplot(residuals.train)
boxplot(residuals.test)

#---- 7. Determine the optimal degree of a polynomial with cross-validation. ----

# Our data shows a linear relationship, indicating that a polynomial fit will not outperform a simple linear fit. 
# Furthermore, multiple ordinary least squares (OLR) models do not significantly outperform simpler models. 
# However, if needed, you can follow this procedure:
  
# 1. Train several models with different polynomial degrees.
# 2. Calculate the error for each model on both training and testing data.
# 3. Plot the errors for each polynomial degree.
# 4. Identify the degree that results in the largest error drop, after which the error no longer significantly decreases.
# 5. Be aware that if the model error on test data starts to rise, it may indicate overfitting on the training data.

# By following these steps, you can determine the optimal polynomial degree for your model.


