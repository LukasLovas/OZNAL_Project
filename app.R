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
library(ggcorrplot)
library(shinydashboard)
library(pls)

# ── COLUMN VALIDATION ─────────────────────────────────────────────────
required_cols <- c(
  "fuelType", "fuelType1", "displ", "cylinders", "trany",
  "drive", "VClass", "year", "make", "comb08"
)

validate_columns <- function(data) {
  missing <- setdiff(required_cols, names(data))
  extra   <- setdiff(names(data), names(vehicles_model))
  
  errors <- c()
  
  if (length(missing) > 0) {
    errors <- c(errors, paste("Missing required columns:", paste(missing, collapse = ", ")))
  }
  if (length(extra) > 0) {
    warning(paste("Extra columns (will be ignored):", paste(extra, collapse = ", ")))
  }
  
  if (length(errors) > 0) stop(paste(errors, collapse = "\n"))
  invisible(TRUE)
}


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
DATA_PATH <- file.path("data", "vehicles.csv")
MODELS_DIR <- "models"

vehicles_model <- prepare_data(DATA_PATH)

set.seed(123)
data_split <- initial_split(vehicles_model, prop = 0.8, strata = comb08)
train_data <- training(data_split)
test_data  <- testing(data_split)

# ── RECIPES ───────────────────────────────────────────────────────────

clean_recipe <- recipe(comb08 ~ ., data = train_data) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_impute_median(displ, cylinders) %>%
  step_mutate(
    phevBlended = as.integer(phevBlended),
    is_cvt = as.integer(is_cvt)
  ) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

processed_recipe <- prep(clean_recipe, training = train_data)
test_processed <- bake(processed_recipe, new_data = test_data)

# ── LOAD PRE-TRAINED MODELS ───────────────────────────────────────────
model_path <- function(filename) {
  file.path(MODELS_DIR, filename)
}

load_model_safe <- function(path) {
  if (file.exists(path)) readRDS(path) else NULL
}

model_entry <- function(filename, type = "workflow") {
  object <- load_model_safe(model_path(filename))
  if (is.null(object)) return(NULL)
  list(object = object, type = type)
}

expected_model_files <- c(
  "OLS + interaction"  = "ols_interact_fit.rds",
  "Decision tree"      = "dt_final.rds",
  "Random forest"      = "rf_final.rds",
  "KNN"                = "knn_final.rds",
  "Elastic Net"        = "en_fit.rds",
  "Forward stepwise"   = "forward_model.rds",
  "Random forest + PCA" = "pca_rf_final.rds",
  "PLS"                = "pls_final.rds"
)

missing_model_files <- expected_model_files[
  !file.exists(file.path(MODELS_DIR, expected_model_files))
]

pretrained_models <- list(
  "OLS + interaction"  = model_entry(expected_model_files[["OLS + interaction"]]),
  "Decision tree"      = model_entry(expected_model_files[["Decision tree"]]),
  "Random forest"      = model_entry(expected_model_files[["Random forest"]]),
  "KNN"                = model_entry(expected_model_files[["KNN"]]),
  "Elastic Net"        = model_entry(expected_model_files[["Elastic Net"]]),
  "Forward stepwise"   = model_entry(expected_model_files[["Forward stepwise"]], "processed_lm"),
  "Random forest + PCA" = model_entry(expected_model_files[["Random forest + PCA"]]),
  "PLS"                = model_entry(expected_model_files[["PLS"]], "pls")
)

pretrained_models <- Filter(Negate(is.null), pretrained_models)
cor_matrix <- load_model_safe(model_path("cor_matrix.rds"))

# ── UI ────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("Fuel efficiency prediction"),
  
  tabsetPanel(
    
    # ════════════════════════════════════════════════════
    # TAB 1: Data Preview
    # ════════════════════════════════════════════════════
    tabPanel("Data",
             br(),
             fluidRow(
               column(4,
                      h5("The default dataset is automatically loaded."),
                      hr(),
                      fileInput("file", "Upload CSV", accept = ".csv")
               )
             ),
             hr(),
             h3("Basic info about the default dataset"),
             hr(),
             
             # ── Basic info ──
             h4("Dataset overview before encoding"),
             fluidRow(
               column(3, valueBoxOutput("n_rows",    width = 12)),
               column(3, valueBoxOutput("n_cols",    width = 12)),
               column(3, valueBoxOutput("n_train",   width = 12)),
               column(3, valueBoxOutput("n_test",    width = 12))
             ),
             hr(),
             
             # ── Data preview ──
             h4("Data preview before encoding"),
             tableOutput("data_preview"),
             hr(),
             
             # ── comb08 distribution ──
             h4("Target variable: comb08"),
             fluidRow(
               column(6, plotOutput("eda_histogram")),
               column(6, plotOutput("eda_boxplot"))
             ),
             hr(),
             
             # ── Train vs Test ──
             h4("Train vs Test distribution"),
             plotOutput("eda_train_test"),
             hr(),
             
             # ── Correlation matrix ──
             h4("Correlation matrix before preprocessing"),
             plotOutput("eda_correlation", height = "500px")
    ),
    
    # ════════════════════════════════════════════════════
    # TAB 2: Pre-trained Models
    # ════════════════════════════════════════════════════
    tabPanel("Trained models",
             br(),
             sidebarLayout(
               sidebarPanel(
                 h5("These models were trained and tuned in the project R markdown file."),
                 hr(),
                 selectInput("pretrained_model", "Select model",
                             choices = names(pretrained_models)),
                 uiOutput("pretrained_model_status")
               ),
               mainPanel(
                 h4("Performance metrics"),
                 tableOutput("pretrained_metrics"),
                 hr(),
                 h4("Actual vs Predicted"),
                 plotOutput("pretrained_actual_vs_pred"),
                 hr(),
                 h4("Residual plot"),
                 plotOutput("pretrained_residuals"),
                 hr(),
                 h4("Feature importance"),
                 plotOutput("pretrained_importance")
               )
             )
    ),
    
    # ════════════════════════════════════════════════════
    # TAB 3: Custom Model
    # ════════════════════════════════════════════════════
    tabPanel("Custom model",
             br(),
             sidebarLayout(
               sidebarPanel(
                 
                 h5("Any change automatically refits the model."),
                 hr(),
                 
                 # Model type
                 selectInput("model_type", "Select model",
                             choices = c("Linear regression" = "lm",
                                         "Lasso"             = "lasso",
                                         "Ridge"             = "ridge",
                                         "Elastic Net"       = "elastic_net",
                                         "Random forest"     = "rf",
                                         "Decision tree"     = "dt",
                                         "KNN"               = "knn")),
                 
                 # Regularized linear model parameters
                 conditionalPanel(
                   condition = "input.model_type == 'lasso' || input.model_type == 'ridge' || input.model_type == 'elastic_net'",
                   sliderInput("penalty", "Penalty (lambda)",
                               min = 0.001, max = 1, value = 0.1, step = 0.001)
                 ),
                 conditionalPanel(
                   condition = "input.model_type == 'elastic_net'",
                   sliderInput("mixture", "Mixture (alpha)",
                               min = 0, max = 1, value = 0.5, step = 0.05)
                 ),
                 
                 # Random Forest parameters
                 conditionalPanel(
                   condition = "input.model_type == 'rf'",
                   sliderInput("trees", "trees",
                               min = 50, max = 500, value = 200, step = 50),
                   sliderInput("mtry", "mtry",
                               min = 2, max = 22, value = 10, step = 1),
                   sliderInput("min_n_rf", "min_n",
                               min = 2, max = 20, value = 2, step = 1)
                 ),
                 
                 # Decision Tree parameters
                 conditionalPanel(
                   condition = "input.model_type == 'dt'",
                   sliderInput("cost_complexity", "cost_complexity",
                               min = 0.0001, max = 0.1, value = 0.0001, step = 0.0001),
                   sliderInput("tree_depth", "tree_depth",
                               min = 3, max = 15, value = 15, step = 1),
                   sliderInput("min_n_dt", "min_n",
                               min = 2, max = 20, value = 8, step = 1)
                 ),
                 
                 # KNN parameters
                 conditionalPanel(
                   condition = "input.model_type == 'knn'",
                   sliderInput("neighbors", "neighbors",
                               min = 3, max = 25, value = 3, step = 1),
                   selectInput("weight_func", "weight_func",
                               choices = c("rectangular", "triangular", "gaussian"),
                               selected = "gaussian"),
                   radioButtons("dist_power", "dist_power",
                                choices = c("1" = 1, "2" = 2),
                                selected = 1, inline = TRUE)
                 ),
                 
                 hr(),
                 
                 # Feature space
                 selectInput("feature_space", "Feature space",
                             choices = c("Full features (22 predictors)" = "full",
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
                 h4("Performance metrics"),
                 tableOutput("metrics"),
                 hr(),
                 h4("Actual vs Predicted"),
                 plotOutput("actual_vs_pred"),
                 hr(),
                 h4("Residual plot"),
                 plotOutput("residual_plot"),
                 hr(),
                 h4("Feature importance"),
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
  data_r <- reactiveVal(head(vehicles_model, 10))  # default preview
  
  observeEvent(input$load_default, {
    data_r(vehicles_model)
    showNotification("Default dataset loaded.", type = "message", duration = 2)
  })
  
  observeEvent(input$file, {
    raw <- tryCatch(
      read_csv(input$file$datapath, show_col_types = FALSE),
      error = function(e) {
        showNotification("Could not read the file. Please upload a valid CSV.", 
                         type = "error", duration = 5)
        NULL
      }
    )
    
    req(raw)
    
    missing <- setdiff(required_cols, names(raw))
    
    if (length(missing) > 0) {
      showNotification(
        paste0("File cannot be used, becuase it is missing some of the required columns: ", 
               paste(missing, collapse = ", ")),
        type     = "error",
        duration = 8
      )
      return()
    }
    
    data_r(prepare_data(input$file$datapath))
    showNotification("File loaded successfully.", type = "message", duration = 2)
  })
  
  output$data_preview <- renderTable({
    req(data_r())
    head(data_r(), 15)
  })
  
  # ── Pre-trained models ───────────────────────────────────────────────
  output$pretrained_model_status <- renderUI({
    if (length(pretrained_models) == 0) {
      return(tags$p(
        "No pretrained models loaded. Run the model export chunk in project.Rmd.",
        style = "color: #B00020; font-weight: bold;"
      ))
    }
    
    if (length(missing_model_files) == 0) {
      return(tags$p(
        "All expected pretrained models are loaded.",
        style = "color: #1B7F37;"
      ))
    }
    
    tags$div(
      tags$p("Missing model files:", style = "color: #B26A00; font-weight: bold;"),
      tags$ul(lapply(unname(missing_model_files), tags$li))
    )
  })
  
  predict_pretrained <- function(entry) {
    if (entry$type == "processed_lm") {
      return(tibble(
        .pred = as.numeric(predict(entry$object, newdata = test_processed)),
        comb08 = test_processed$comb08
      ))
    }
    
    if (entry$type == "pls") {
      pls_object <- entry$object
      pls_model <- if (inherits(pls_object, "mvr")) pls_object else pls_object$model
      pls_ncomp <- if (inherits(pls_object, "mvr")) 3 else pls_object$ncomp
      
      pls_pred <- predict(
        pls_model,
        newdata = test_processed,
        ncomp = pls_ncomp
      )[, , 1]
      
      return(tibble(
        .pred = as.numeric(pls_pred),
        comb08 = test_processed$comb08
      ))
    }
    
    predict(entry$object, test_data) %>%
      bind_cols(tibble(comb08 = test_data$comb08))
  }
  
  pretrained_preds <- reactive({
    req(input$pretrained_model)
    entry <- pretrained_models[[input$pretrained_model]]
    req(entry)
    predict_pretrained(entry)
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
    entry <- pretrained_models[[input$pretrained_model]]
    req(entry)
    fit <- entry$object
    req(fit)
    tryCatch({
      fit %>%
        extract_fit_parsnip() %>%
        vip(num_features = 15) +
        labs(title = paste("Feature importance –", input$pretrained_model)) +
        theme_minimal()
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, "Feature importance not available for this model.", cex = 1.2)
    })
  })
  
  # ── Custom model ─────────────────────────
  model_inputs <- reactive({
    list(
      model_type    = input$model_type,
      feature_space = input$feature_space,
      penalty       = input$penalty,
      mixture       = input$mixture,
      trees         = input$trees,
      mtry          = input$mtry,
      n_comp        = input$n_comp,
      neighbors     = input$neighbors,
      tree_depth    = input$tree_depth,
      min_n_dt      = input$min_n_dt,
      min_n_rf       = input$min_n_rf,
      cost_complexity = input$cost_complexity,
      weight_func    = input$weight_func,
      dist_power     = input$dist_power
    )
  })
  
  # waits 1.5s after last change before fitting
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
                     elastic_net = linear_reg(penalty = params$penalty, mixture = params$mixture) %>% set_engine("glmnet"),
                     rf  = rand_forest(trees = params$trees, mtry = params$mtry,
                                       min_n = params$min_n_rf) %>%
                       set_mode("regression") %>%
                       set_engine("ranger", importance = "impurity"),
                     
                     dt  = decision_tree(cost_complexity = params$cost_complexity,
                                         tree_depth      = params$tree_depth,
                                         min_n           = params$min_n_dt) %>%
                       set_mode("regression") %>% set_engine("rpart"),
                     
                     knn = nearest_neighbor(neighbors   = params$neighbors,
                                            weight_func = params$weight_func,
                                            dist_power  = as.numeric(params$dist_power)) %>%
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
        vip(num_features = 15) +
        theme_minimal()
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, "Feature importance not available for this model type.",
           cex = 1.2, col = "gray50")
    })
  })
  
  # ── Value boxes ──────────────────────────────────────────────────────
  output$n_rows  <- renderValueBox({
    valueBox(nrow(vehicles_model), "Total rows")
  })
  output$n_cols  <- renderValueBox({
    valueBox(ncol(vehicles_model), "Total columns")
  })
  output$n_train <- renderValueBox({
    valueBox(nrow(train_data), "Train rows")
  })
  output$n_test  <- renderValueBox({
    valueBox(nrow(test_data), "Test rows")
  })
  
  # ── Histogram ────────────────────────────────────────────────────────
  output$eda_histogram <- renderPlot({
    ggplot(vehicles_model, aes(x = comb08)) +
      geom_histogram(bins = 40, fill = "steelblue", color = "white") +
      labs(x = "comb08 (MPG)", y = "Count", title = "Distribution of comb08") +
      theme_minimal()
  })
  
  # ── Boxplot ──────────────────────────────────────────────────────────
  output$eda_boxplot <- renderPlot({
    ggplot(vehicles_model, aes(y = comb08)) +
      geom_boxplot(fill = "steelblue", alpha = 0.7) +
      coord_flip() +
      labs(y = "comb08 (MPG)", title = "Boxplot of comb08") +
      theme_minimal()
  })

  # ── Train vs Test ────────────────────────────────────────────────────
  output$eda_train_test <- renderPlot({
    bind_rows(
      train_data %>% select(comb08) %>% mutate(split = "Train"),
      test_data  %>% select(comb08) %>% mutate(split = "Test")
    ) %>%
      ggplot(aes(x = comb08, fill = split)) +
      geom_density(alpha = 0.5) +
      scale_fill_manual(values = c("Train" = "steelblue", "Test" = "tomato")) +
      labs(x = "comb08 (MPG)", y = "Density",
           title = "Train vs Test distribution") +
      theme_minimal()
  })
  
  # ── Correlation matrix ───────────────────────────────────────────────
  output$eda_correlation <- renderPlot({
    if (is.null(cor_matrix)) {
      plot.new()
      text(0.5, 0.5, "Correlation matrix file not found. Run the export chunk in project.Rmd.", cex = 1.1)
      return()
    }
    
    ggcorrplot(cor_matrix)
  })
}

# ── RUN ───────────────────────────────────────────────────────────────
shinyApp(ui, server)
