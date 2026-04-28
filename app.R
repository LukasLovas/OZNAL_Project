library(shiny)
library(tidyverse)
library(tidymodels)
library(ranger)
library(glmnet)
library(stringr)
library(forcats)
library(rpart)
library(kknn)
library(vip)

# ── PREPROCESSING ─────────────────────────────────────────────────────
prepare_data <- function(path) {
  data <- read_csv(path, show_col_types = FALSE)
  
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

# ── LOAD & SPLIT DATA ─────────────────────────────────────────────────
DATA_PATH <- "C:/school/oznal/OZNAL_Project/data/vehicles.csv"

vehicles_model <- prepare_data(DATA_PATH)

set.seed(123)
data_split <- initial_split(vehicles_model, prop = 0.8, strata = comb08)
train_data <- training(data_split)
test_data  <- testing(data_split)

# ── RECIPES ───────────────────────────────────────────────────────────
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

# ── LOAD PRE-TRAINED MODELS ───────────────────────────────────────────
# Ak modely existujú, načítaj ich; inak nastav NULL
load_model_safe <- function(path) {
  if (file.exists(path)) readRDS(path) else NULL
}

pretrained_models <- list(
  "Decision Tree"       = load_model_safe("models/dt_final.rds"),
  "Random Forest"       = load_model_safe("models/rf_final.rds"),
  "KNN"                 = load_model_safe("models/knn_final.rds"),
  "Random Forest + PCA" = load_model_safe("models/pca_rf_final.rds")
)
# Odstráň NULL modely (ktoré ešte neboli natrénované)
pretrained_models <- Filter(Negate(is.null), pretrained_models)

# ── UI ────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("Fuel Efficiency Predictor – comb08"),
  
  tabsetPanel(
    
    # ════════════════════════════════════════════════════
    # TAB 1: Data Preview
    # ════════════════════════════════════════════════════
    tabPanel("Data",
             br(),
             fluidRow(
               column(4,
                      fileInput("file", "Upload CSV", accept = ".csv"),
                      actionButton("load_default", "Use default dataset",
                                   class = "btn-primary")
               ),
               column(8,
                      tableOutput("data_preview")
               )
             )
    ),
    
    # ════════════════════════════════════════════════════
    # TAB 2: Pre-trained Models
    # ════════════════════════════════════════════════════
    tabPanel("Trained Models",
             br(),
             sidebarLayout(
               sidebarPanel(
                 selectInput("pretrained_model", "Select Model",
                             choices = names(pretrained_models)),
                 hr(),
                 h5("These models were trained and tuned in the project R markdown file.")
               ),
               mainPanel(
                 h4("Performance Metrics"),
                 tableOutput("pretrained_metrics"),
                 hr(),
                 h4("Actual vs Predicted"),
                 plotOutput("pretrained_actual_vs_pred"),
                 hr(),
                 h4("Residual Plot"),
                 plotOutput("pretrained_residuals"),
                 hr(),
                 h4("Feature Importance"),
                 plotOutput("pretrained_importance")
               )
             )
    ),
    
    # ════════════════════════════════════════════════════
    # TAB 3: Custom Model (auto-update)
    # ════════════════════════════════════════════════════
    tabPanel("Custom Model",
             br(),
             sidebarLayout(
               sidebarPanel(
                 
                 h5("Any change automatically refits the model."),
                 hr(),
                 
                 # Model type
                 selectInput("model_type", "Select Model",
                             choices = c("Linear Regression" = "lm",
                                         "Lasso"             = "lasso",
                                         "Ridge"             = "ridge",
                                         "Random Forest"     = "rf",
                                         "Decision Tree"     = "dt",
                                         "KNN"               = "knn")),
                 
                 # Lasso / Ridge parameters
                 conditionalPanel(
                   condition = "input.model_type == 'lasso' || input.model_type == 'ridge'",
                   sliderInput("penalty", "Penalty (lambda)",
                               min = 0.001, max = 1, value = 0.1, step = 0.001)
                 ),
                 
                 # Random Forest parameters
                 conditionalPanel(
                   condition = "input.model_type == 'rf'",
                   sliderInput("trees", "Number of Trees",
                               min = 50, max = 500, value = 100, step = 50),
                   sliderInput("mtry", "mtry (features per split)",
                               min = 2, max = 22, value = 5, step = 1)
                 ),
                 
                 # Decision Tree parameters
                 conditionalPanel(
                   condition = "input.model_type == 'dt'",
                   sliderInput("tree_depth", "Tree Depth",
                               min = 3, max = 15, value = 8, step = 1),
                   sliderInput("min_n_dt", "Min Node Size",
                               min = 2, max = 20, value = 8, step = 1)
                 ),
                 
                 # KNN parameters
                 conditionalPanel(
                   condition = "input.model_type == 'knn'",
                   sliderInput("neighbors", "Number of Neighbors",
                               min = 3, max = 25, value = 5, step = 1)
                 ),
                 
                 hr(),
                 
                 # Feature space
                 selectInput("feature_space", "Feature Space",
                             choices = c("Full features" = "full",
                                         "PCA"           = "pca")),
                 conditionalPanel(
                   condition = "input.feature_space == 'pca'",
                   sliderInput("n_comp", "Number of PCA components",
                               min = 5, max = 22, value = 18, step = 1)
                 ),
                 
                 hr(),
                 # Status
                 uiOutput("training_status")
               ),
               
               mainPanel(
                 h4("Performance Metrics"),
                 tableOutput("metrics"),
                 hr(),
                 h4("Actual vs Predicted"),
                 plotOutput("actual_vs_pred"),
                 hr(),
                 h4("Residual Plot"),
                 plotOutput("residual_plot"),
                 hr(),
                 h4("Feature Importance"),
                 plotOutput("importance_plot")
               )
             )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  
  options(shiny.maxRequestSize = 200 * 1024^2)
  
  # ── Data loading ────────────────────────────────────────────────────
  data_r <- reactiveVal(head(vehicles_model, 100))  # default preview
  
  observeEvent(input$load_default, {
    data_r(vehicles_model)
    showNotification("Default dataset loaded.", type = "message", duration = 2)
  })
  
  observeEvent(input$file, {
    tryCatch({
      data_r(prepare_data(input$file$datapath))
      showNotification("File loaded and preprocessed.", type = "message", duration = 2)
    }, error = function(e) {
      showNotification(paste("Error loading file:", e$message), type = "error")
    })
  })
  
  output$data_preview <- renderTable({
    req(data_r())
    head(data_r(), 15)
  })
  
  # ── Pre-trained models ───────────────────────────────────────────────
  pretrained_preds <- reactive({
    req(input$pretrained_model)
    fit <- pretrained_models[[input$pretrained_model]]
    req(fit)
    predict(fit, test_data) %>%
      bind_cols(tibble(comb08 = test_data$comb08))
  })
  
  output$pretrained_metrics <- renderTable({
    p <- pretrained_preds()
    bind_rows(
      rmse(p, truth = comb08, estimate = .pred),
      rsq(p,  truth = comb08, estimate = .pred),
      mae(p,  truth = comb08, estimate = .pred)
    ) %>% select(.metric, .estimate) %>%
      rename(Metric = .metric, Value = .estimate)
  })
  
  output$pretrained_actual_vs_pred <- renderPlot({
    pretrained_preds() %>%
      ggplot(aes(x = comb08, y = .pred)) +
      geom_point(alpha = 0.3) +
      geom_abline(linetype = "dashed", color = "red") +
      labs(title = input$pretrained_model, x = "Actual comb08", y = "Predicted comb08") +
      theme_minimal()
  })
  
  output$pretrained_residuals <- renderPlot({
    pretrained_preds() %>%
      mutate(residual = comb08 - .pred) %>%
      ggplot(aes(x = .pred, y = residual)) +
      geom_point(alpha = 0.3) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      labs(x = "Predicted comb08", y = "Residuals") +
      theme_minimal()
  })
  
  output$pretrained_importance <- renderPlot({
    fit <- pretrained_models[[input$pretrained_model]]
    req(fit)
    tryCatch({
      fit %>%
        extract_fit_parsnip() %>%
        vip::vip(num_features = 15) +
        labs(title = paste("Feature Importance –", input$pretrained_model)) +
        theme_minimal()
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, "Feature importance not available for this model.", cex = 1.2)
    })
  })
  
  # ── Custom model (auto-update with debounce) ─────────────────────────
  model_inputs <- reactive({
    list(
      model_type    = input$model_type,
      feature_space = input$feature_space,
      penalty       = input$penalty,
      trees         = input$trees,
      mtry          = input$mtry,
      n_comp        = input$n_comp,
      neighbors     = input$neighbors,
      tree_depth    = input$tree_depth,
      min_n_dt      = input$min_n_dt
    )
  })
  
  # Čaká 1.5s po poslednej zmene pred refitovaním
  model_inputs_d <- debounce(model_inputs, 1500)
  
  training_running <- reactiveVal(FALSE)
  
  model_results <- reactive({
    params <- model_inputs_d()
    
    training_running(TRUE)
    on.exit(training_running(FALSE))
    
    withProgress(message = "Training model...", value = 0, {
      
      incProgress(0.2, detail = "Building recipe...")
      rec <- if (params$feature_space == "pca") {
        clean_recipe %>%
          step_pca(all_numeric_predictors(), num_comp = params$n_comp)
      } else {
        clean_recipe
      }
      
      incProgress(0.4, detail = "Defining model spec...")
      spec <- switch(params$model_type,
                     lm    = linear_reg() %>% set_engine("lm"),
                     lasso = linear_reg(penalty = params$penalty, mixture = 1) %>% set_engine("glmnet"),
                     ridge = linear_reg(penalty = params$penalty, mixture = 0) %>% set_engine("glmnet"),
                     rf    = rand_forest(trees = params$trees, mtry = params$mtry) %>%
                       set_mode("regression") %>%
                       set_engine("ranger", importance = "impurity"),
                     dt    = decision_tree(tree_depth = params$tree_depth,
                                           min_n = params$min_n_dt) %>%
                       set_mode("regression") %>% set_engine("rpart"),
                     knn   = nearest_neighbor(neighbors = params$neighbors) %>%
                       set_mode("regression") %>% set_engine("kknn")
      )
      
      incProgress(0.6, detail = "Fitting model...")
      wf  <- workflow() %>% add_recipe(rec) %>% add_model(spec)
      fit <- fit(wf, train_data)
      
      incProgress(0.9, detail = "Generating predictions...")
      preds <- predict(fit, test_data) %>%
        bind_cols(tibble(comb08 = test_data$comb08))
      
      incProgress(1.0, detail = "Done!")
    })
    
    showNotification("Model updated!", type = "message", duration = 2)
    list(fit = fit, preds = preds, params = params)
  })
  
  output$training_status <- renderUI({
    if (training_running()) {
      tags$p("Training...", style = "color: orange; font-weight: bold;")
    } else {
      tags$p("Model is ready", style = "color: green; font-weight: bold;")
    }
  })
  
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
  
  output$actual_vs_pred <- renderPlot({
    req(model_results())
    model_results()$preds %>%
      ggplot(aes(x = comb08, y = .pred)) +
      geom_point(alpha = 0.3) +
      geom_abline(linetype = "dashed", color = "red") +
      labs(x = "Actual comb08", y = "Predicted comb08") +
      theme_minimal()
  })
  
  output$residual_plot <- renderPlot({
    req(model_results())
    model_results()$preds %>%
      mutate(residual = comb08 - .pred) %>%
      ggplot(aes(x = .pred, y = residual)) +
      geom_point(alpha = 0.3) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      labs(x = "Predicted comb08", y = "Residuals") +
      theme_minimal()
  })
  
  output$importance_plot <- renderPlot({
    req(model_results())
    tryCatch({
      model_results()$fit %>%
        extract_fit_parsnip() %>%
        vip::vip(num_features = 15) +
        theme_minimal()
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, "Feature importance not available for this model type.", cex = 1.2)
    })
  })
}

# ── RUN ───────────────────────────────────────────────────────────────
shinyApp(ui, server)