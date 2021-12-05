---
title: "Machine Learning End to End"
authors: Gabriel Blanco, Sebastian Esponda 
date: 04 december 2021
output: html_document
Inputs: Dataset from UCI machine learning repository. 
Output: Charts, submission.

#---- Libraries ----
suppressPackageStartupMessages(
  {
    library(data.table)    # Fast Data processing.
    library(caret)        # Machine learning algorithms.
    library(tidytable)     # data.table with dplyr syntax.
    library(inspectdf)     # Automatic EDA.
    library(DataExplorer)  # Automatic EDA.
    library(dplyr)         # Data processing with pipes.
    library(ranger)        # Fast RandomForest.
    library(magrittr)      # Piping.
    library(ggplot2)       # The most beautiful charts in the ML world.
    library(forcats)       # Treat categorical variables
    library(missRanger)    # NA's imputation with ranger.
    library(tidytable)     # Usar data.table como si fuese dplyr.
    library(gclus)         # Panels in scatter plots matrices
  } 
)


# ---- INI DATA LOADING ----

data <- as.data.frame(fread("./bank/bank-full.csv"))

# ---- END DATA LOADING ----

# ---- INI EXPLORATORY DATA ANALYSIS ----

# Un summary de todas las columnas
summary(data)



# Horizontal bar plot for categorical column composition
x <- inspect_cat(data) 
show_plot(x,col_palette = 1)

# Se observa como la variable default tiene dos categorias ("no" y "yes")
# En default la categoria no supone practicamente la totalidad de la columna.

# Se observa como la variable y que es la target, el no es ampliamente superior al yes
# por lo que los datos estan desbalanceados


# Correlation betwee numeric columns + confidence intervals
x <- inspect_cor(data)
show_plot(x)

# No hay una correlación muy elevada entre las variables, la correlación más destacada seria
# (previous & pdays) de aproximadamente un 0.5

# Scatter plot Previous - Pdays

previous_pdays <- ggplot(data, aes(x=previous, y=pdays, colour =)) +
                  labs(
                    title = "Scatter Plot Previous - Pdays",
                    caption = "Data source: UCI ML Repository",
                    tag = waiver()
                  ) +
                  geom_point(color = "4E84C4")
previous_pdays

# Se realiza un grafico scatter plot de todas las variables númericas
var_num <- data %>% 
  select(where(is.numeric)) # seleccionamos variables númericas

# Atributos de la correlacion
corr <- abs(cor(var_num))

colors <- dmat.color(corr)

order <- order.single(corr)

#Utilizamos la funcion cpairs

cpairs(var_num, order, panel.colors = colors, gap = 0.5,
       main = "Variables ordenadas y coloreadas por correlación")

# Gráficos distribuciones de varibles categoricas con variable target 

variables <- list('job', 'marital', 'education', 'housing', 'month', 'loan')


for (i in variables){
  plot <-  ggplot(data, aes_string(x = i, fill = as.factor(data$y)))+ 
    geom_bar( position = "stack")+ scale_fill_discrete(name = "churn")
  print(plot)
}


# Bar plot of most frequent category for each categorical column in %
x <- inspect_imb(data)
show_plot(x)

# Se observa como se comprobo anteriormente que en las variables:
#  en la columna default la categoria "no" esta presente en el 98.2 % de las observaciones
# la variable target y esta desbalanceada pues la categoria no es del 88.3


# Occurence of NAs in each column ranked in descending order
x <- inspect_na(data)
show_plot(x)

# No hay Na´s

# Histograms for numeric columns
x <- inspect_num(data)
show_plot(x)

# Hay varios en outliers en practicamente todas las columnas menos : age, day
# En edad habrá algunas personas con edadad de mas de 75 años
# En balance, vemos como hay algunos que tienen balance negativo y la gran mayoria tiene menos de 10mil
# En campaign cercanos a 0 y algun outlier
# Duración hay algun outlier
# Pdays algun outlier y valores negativos
# Previous algun outlier y valores cercanos a 0 



# Barplot of column types
x <- inspect_types(data)
show_plot(x)

# Hay 7 variables númericas y 10 categóricas

# Grafico para ver scatter plots

var_num <- data %>% 
  select(where(is.numeric))

# Scatter plot eje y: age
plot_scatterplot(var_num, by = "age", sampled_rows = 1000L)

#Scatter plot eje y: day 

plot_scatterplot(var_num, by = "duration", sampled_rows = 1000L)

# Grafico para ver boxplots de los datos que tenemos

p <- ggplot(data, aes(x=education, y=age,fill=education)) + 
  geom_boxplot()+
  labs(title="Plot of Age per Education",x="Education", y = "Age")

p

# Vemos como en cuanto a educación la población mas joven tiene un nivel educativo mas alto.




p1 <- ggplot(data, aes(x=marital, y=age,fill=marital)) + 
  geom_boxplot()+
  labs(title="Plot of Age per Marital",x="Marital", y = "Age")

p1

# El estado civil de nuestro conjunto, los mas jóvenes solteros y los adultos divorciados y casados.



p2 <- ggplot(data, aes(x=job, y=age,fill=job)) + 
  geom_boxplot()+
  labs(title="Plot of Age per Job",x="Type of Job", y = "Age")

p2

p3 <- ggplot(data, aes(x=contact, y=age,fill=contact)) + 
  geom_boxplot()+
  labs(title="Plot of age per Contact",x="Contact", y = "Age")

p3

p4 <- ggplot(data, aes(x=month, y=campaign,fill=month)) + 
  geom_boxplot()+
  labs(title="Plot of Campaign per Month",x="Month", y = "Age")

p4


# 17 variables: 7 de ellas numericas y 10 tipo character

# ---- INI FEATURE ENGINEERING ----
# Pasamos la variable target a factor

data$y <-ifelse(data$y=="yes",1,0)

data$y <- as.factor(data$y)


# Eliminamos las columnas que no son significativas

data <- data %>% dplyr::select(-default)

# Tratamiento de outliers



# Datos desbalanceados

# HAY QUE HACER FEATURE ENGINEERING




# Modeling

#Pruebo Random Forest

TrainingIndex <- createDataPartition(data$y, p=0.8, list = FALSE)
TrainingSet <- data[TrainingIndex,] # Training Set
TestingSet <- data[-TrainingIndex,] # Test Set

#Random forest with Ranger

mymodel <- ranger( 
  y ~ . , 
  data = TrainingSet,
  importance = 'impurity',
  verbose = TRUE
)

# Accuracy 
acierto_val <- 1 - mymodel$prediction.error
acierto_val

# Importancia de las variables

var_impor <- as.data.frame(mymodel$variable.importance)
names(var_impor) <- c('importancia') 
var_impor$variables <- rownames(var_impor)
rownames(var_impor) <- NULL

var_impor %<>%
  arrange(desc(importancia))
var_impor


var_impor %>%
  ggplot(aes(x = fct_reorder(variables, importancia), y = importancia)) +
  geom_col(group = 1, fill = 'darkred') +
  labs(
    x = '', 
    y = 'Importancia Relativa',
    title = "IMPORTANCIA MODELO",
    subtitle = paste("algoritmo ranger - acierto: ", round(acierto_val,4), sep = ""), 
    caption = paste("num vars: ", nrow(var_impor), sep = "") 
  ) +
  coord_flip() + 
  theme_bw()
# Test predictions

mypred <- predict( mymodel, TestingSet)$predictions



confusionMatrix(mypred, TestingSet$y)



