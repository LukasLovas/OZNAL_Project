library(shiny)
library(tidyverse)
library(tidymodels)
library(ranger)
library(glmnet)
library(stringr)
library(forcats)
library(rpart)
library(kknn)

# ── PREPROCESSING (rovnaký ako v RMD) ────────────────────────────────
prepare_data <- function(path) {
  data <- read_csv(path)
  
  # Feature engineering
  data_fe <- data %>%
    mutate(
      is_electric = (fuelType == "Electricity") | (fuelType1 == "Electricity"),
      is_electric = replace_na(is_electric, FALSE),
      displ       = if_else(is_electric, 0, displ),
      cylinders   = if_else(is_electric, 0, cylinders),
      
      fuel_group = case_when(
        fuelType %in% c("Regular", "Premium", "Midgrade")    ~ "Gasoline",
        fuelType == "Diesel"                                  ~ "Diesel",
        fuelType == "Electricity"                             ~ "Electricity",
        fuelType %in% c("Premium and Electricity",
                        "Regular Gas and Electricity",
                        "Premium Gas or Electricity",
                        "Regular Gas or Electricity",
                        "Electricity and Hydrogen")          ~ "Hybrid/Electric mix",
        fuelType %in% c("Gasoline or E85", "Premium or E85") ~ "Flex-fuel/E85",
        TRUE                                                  ~ "Other"
      ),
      
      drive = case_when(
        str_detect(drive, "4-Wheel|All-Wheel") ~ "AWD/4WD",
        str_detect(drive, "Front-Wheel")       ~ "Front-Wheel Drive",
        str_detect(drive, "Rear-Wheel")        ~ "Rear-Wheel Drive",
        is.na(drive)                           ~ "Unknown",
        TRUE                                   ~ "Other"
      ),
      
      VClass = fct_lump(as.factor(VClass), n = 10, other_level = "Other"),
      
      transmission_type = case_when(
        str_detect(trany, regex("^Automatic", ignore_case = TRUE)) ~ "Automatic",
        str_detect(trany, regex("^Manual",    ignore_case = TRUE)) ~ "Manual",
        TRUE ~ "Other"
      ),
      is_cvt             = replace_na(str_detect(trany, regex("variable gear ratios", ignore_case = TRUE)), FALSE),
      n_gears            = replace_na(as.integer(str_match(trany, "(\\d+)-spd")[, 2]), 0),
      has_discrete_gears = as.integer(n_gears > 0)
    )
  
  # Columns to remove (rovnaké skupiny ako v RMD)
  cols_to_remove <- unique(c(
    "c240bDscr","c240Dscr","sCharger","rangeA","fuelType2","guzzler",
    "evMotor","atvType","tCharger","trans_dscr","startStop","mfrCode",
    "id","engId","model",
    "city08","highway08","UCity","UHighway","barrels08","co2TailpipeGpm",
    "fuelCost08","youSaveSpend","comb08U","city08U","highway08U",
    "combA08","cityA08","highwayA08","combA08U","cityA08U","highwayA08U",
    "co2","co2A","co2TailpipeAGpm","feScore","ghgScore","ghgScoreA",
    "phevCity","phevHwy","phevComb","combE","cityE","highwayE",
    "combinedCD","combinedUF","cityCD","cityUF","highwayCD","highwayUF",
    "barrelsA08","charge120","charge240","charge240b","fuelCostA08",
    "hlv","hpv","lv2","lv4","pv2","pv4","range","rangeCity","rangeCityA",
    "rangeHwy","rangeHwyA","UCityA","UHighwayA",
    "createdOn","modifiedOn","eng_dscr","baseModel",
    "trany","fuelType","fuelType1","is_electric","has_combustion_engine"
  ))
  
  data_fe %>%
    filter(!is_electric) %>%
    select(-any_of(cols_to_remove))
}

# Načítaj a priprav dáta RAZ pri štarte
vehicles_model <- prepare_data("C:/school/oznal/OZNAL_Project/data/vehicles.csv")

set.seed(123)
data_split <- initial_split(vehicles_model, prop = 0.8, strata = comb08)
train_data <- training(data_split)
test_data  <- testing(data_split)
folds      <- vfold_cv(train_data, v = 5, strata = comb08)

# Skontroluj, či stĺpce existujú
cat("Columns in train_data:\n")
print(names(train_data))

# Recipes definuj AŽ TU (nie pred prepare_data)
clean_recipe <- recipe(comb08 ~ ., data = train_data) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_impute_median(displ, cylinders) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# premenovaný knn recipe
clean_recipe <- recipe(comb08 ~ ., data = train_data) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_impute_median(displ, cylinders) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

pca_recipe <- clean_recipe %>%
  step_pca(all_numeric_predictors(), num_comp = 18)

# ── UI ────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("Fuel Efficiency Predictor – comb08"),
  
  sidebarLayout(
    sidebarPanel(
      
      # 1. Data loading
      fileInput("file", "Upload CSV", accept = ".csv"),
      actionButton("load_default", "Use default dataset"),
      hr(),
      
      # 2. Model selection
      selectInput("model_type", "Select Model",
                  choices = c("Linear Regression" = "lm",
                              "Lasso"             = "lasso",
                              "Ridge"             = "ridge",
                              "Random Forest"     = "rf",
                              "Decision Tree"     = "dt",
                              "KNN"               = "knn")),
      
      # 3. Dynamic parameters – menia sa podľa modelu
      conditionalPanel(
        condition = "input.model_type == 'lasso' || input.model_type == 'ridge'",
        sliderInput("penalty", "Penalty (lambda)",
                    min = 0.001, max = 1, value = 0.1, step = 0.001)
      ),
      conditionalPanel(
        condition = "input.model_type == 'lasso'",
        sliderInput("mixture", "Mixture (0=Ridge, 1=Lasso)",
                    min = 0, max = 1, value = 1, step = 0.1)
      ),
      conditionalPanel(
        condition = "input.model_type == 'rf'",
        sliderInput("trees", "Number of Trees",
                    min = 50, max = 500, value = 100, step = 50),
        sliderInput("mtry", "mtry (features per split)",
                    min = 2, max = 10, value = 5, step = 1) # zmeniť max = 22 ?
      ),
      conditionalPanel(
        condition = "input.model_type == 'dt'",
        sliderInput("tree_depth", "Tree Depth", min = 3, max = 15, value = 8, step = 1),
        sliderInput("min_n_dt", "Min Node Size", min = 2, max = 20, value = 8, step = 1)
      ),
      conditionalPanel(
        condition = "input.model_type == 'knn'",
        sliderInput("neighbors", "Number of Neighbors", min = 3, max = 25, value = 5, step = 1)
      ),
      
      hr(),
      # 4. Feature space
      selectInput("feature_space", "Feature Space",
                  choices = c("Full features" = "full",
                              "PCA"           = "pca")),
      conditionalPanel(
        condition = "input.feature_space == 'pca'",
        sliderInput("n_comp", "Number of PCA components",
                    min = 5, max = 22, value = 13, step = 1)
      ),
      
      actionButton("run", "Fit Model", class = "btn-primary")
    ),
    
    mainPanel(
      tabsetPanel(
        
        tabPanel("Data",
                 tableOutput("data_preview")),
        
        tabPanel("Model Results",
                 h4("Performance Metrics"),
                 tableOutput("metrics"),
                 hr(),
                 h4("Actual vs Predicted"),
                 plotOutput("actual_vs_pred")),
        
        tabPanel("Residuals",
                 plotOutput("residual_plot")),
        
        tabPanel("Feature Importance",
                 plotOutput("importance_plot"))
      )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  options(shiny.maxRequestSize = 100 * 1024^2)
  
  # Reaktívne dáta
  data_r <- reactiveVal(NULL)
  
  observeEvent(input$load_default, {
    data_r(prepare_data("C:/school/oznal/OZNAL_Project/data/vehicles.csv"))
  })
  
  observeEvent(input$file, {
    data_r(prepare_data(input$file$datapath))
  })
  
  output$data_preview <- renderTable({
    req(data_r())
    head(data_r(), 10)
  })
  
  # Fit model – spustí sa len po kliknutí "Fit Model"
  model_results <- eventReactive(input$run, {
    
    withProgress(message = "Training model...", value = 0, {
      
      incProgress(0.1, detail = "Preparing data...")
      # POUŽI globálne train_data/test_data, nie data_r()
      
      incProgress(0.3, detail = "Building recipe...")
      rec <- if (input$feature_space == "pca") {
        clean_recipe %>% step_pca(all_numeric_predictors(), num_comp = input$n_comp)
      } else {
        clean_recipe
      }
      
      incProgress(0.5, detail = "Fitting model...")
      spec <- switch(input$model_type,
                     lm    = linear_reg() %>% set_engine("lm"),
                     lasso = linear_reg(penalty = input$penalty, mixture = 1) %>% set_engine("glmnet"),
                     ridge = linear_reg(penalty = input$penalty, mixture = 0) %>% set_engine("glmnet"),
                     rf    = rand_forest(trees = input$trees, mtry = input$mtry) %>%
                       set_mode("regression") %>% set_engine("ranger", importance = "impurity"),
                     dt    = decision_tree(tree_depth = input$tree_depth, min_n = input$min_n_dt) %>%
                       set_mode("regression") %>% set_engine("rpart"),
                     knn   = nearest_neighbor(neighbors = input$neighbors) %>%
                       set_mode("regression") %>% set_engine("kknn")
      )
      
      wf  <- workflow() %>% add_recipe(rec) %>% add_model(spec)
      fit <- fit(wf, train_data)   # <-- globálne train_data
      
      incProgress(0.8, detail = "Generating predictions...")
      preds <- predict(fit, test_data) %>%   # <-- globálne test_data
        bind_cols(tibble(comb08 = test_data$comb08))
      
      incProgress(1.0, detail = "Done!")
    })
    
    showNotification("Model trained successfully!", type = "message", duration = 3)
    list(fit = fit, preds = preds, test = test_data)
  })
  
  # Metrics
  output$metrics <- renderTable({
    req(model_results())
    p <- model_results()$preds
    bind_rows(
      rmse(p, truth = comb08, estimate = .pred),
      rsq(p,  truth = comb08, estimate = .pred),
      mae(p,  truth = comb08, estimate = .pred)
    ) %>% select(.metric, .estimate) %>%
      rename(Metric = .metric, Value = .estimate)
  })
  
  # Actual vs Predicted
  output$actual_vs_pred <- renderPlot({
    req(model_results())
    model_results()$preds %>%
      ggplot(aes(x = comb08, y = .pred)) +
      geom_point(alpha = 0.3) +
      geom_abline(linetype = "dashed", color = "red") +
      labs(x = "Actual", y = "Predicted") +
      theme_minimal()
  })
  
  # Residuals
  output$residual_plot <- renderPlot({
    req(model_results())
    model_results()$preds %>%
      mutate(residual = comb08 - .pred) %>%
      ggplot(aes(x = .pred, y = residual)) +
      geom_point(alpha = 0.3) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      labs(x = "Predicted", y = "Residuals") +
      theme_minimal()
  })
  
  # Feature importance (len pre RF)
  output$importance_plot <- renderPlot({
    req(model_results())
    if (input$model_type == "rf") {
      model_results()$fit %>%
        extract_fit_parsnip() %>%
        vip::vip(num_features = 15) +
        theme_minimal()
    } else {
      model_results()$fit %>%
        extract_fit_parsnip() %>%
        vip::vip(num_features = 15) +
        theme_minimal()
    }
  })
}

# ── RUN ───────────────────────────────────────────────────────────────
shinyApp(ui, server)