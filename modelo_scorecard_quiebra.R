#############################################################################
# Scorecard de riesgo de quiebra corporativa — Taiwanese Bankruptcy Prediction
# Caso piloto de portfolio / prueba de estrés del equipo de agentes
# Roberto Castellanos Aguirre
#
# Fuente de datos: UCI Machine Learning Repository, dataset "Taiwanese
# Bankruptcy Prediction" (id 572), datos reales del Taiwan Economic Journal,
# empresas listadas en la Bolsa de Taiwán 1999-2009. 6.819 empresas, 95
# ratios financieros, target binario "Bankrupt?" (1 = quiebra, 0 = no
# quiebra). Clase MUY desbalanceada: 220 quiebras (3,23%) vs 6.599 sanas
# (96,77%).
# Descarga: https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip
#
# Tipo de problema: clasificación binaria supervisada con desbalance de
# clases severo, para estimación de probabilidad de default (PD) /
# riesgo crediticio corporativo — no es un problema de series de tiempo ni
# de proyección de flujo de caja.
#
# Metodología elegida: Weight of Evidence (WOE) + Information Value (IV)
# para binning/selección de variables, seguido de regresión logística sobre
# las variables transformadas a WOE, escalada a un scorecard de puntos
# (convención PDO — "Points to Double the Odds"). Esta es la metodología
# ESTÁNDAR de la industria de riesgo crediticio (usada por FICO, por bancos
# para scorecards de PD bajo Basilea, y documentada en libros de referencia
# de credit scoring como Siddiqi "Credit Risk Scorecards"). Se eligió sobre
# alternativas de caja negra (random forest, XGBoost) como modelo PRINCIPAL
# porque el entregable pedido es explícitamente un scorecard interpretable
# para un cliente no técnico, y la regresión logística sobre bins WOE es el
# único enfoque de los evaluados que produce una tabla de puntos por rango
# de cada ratio, auditable y explicable rubro por rubro (a diferencia de un
# modelo de árboles, que da importancia de variables pero no una tabla de
# puntos). Un Random Forest se usa en paralelo SOLO como validación cruzada
# de qué variables son predictivas (chequeo de robustez), no como el modelo
# final.
#
# Manejo del desbalance de clases (96,8% / 3,2%):
#   1. Selección de variables con Information Value (IV), que es una medida
#      de poder discriminante que NO se distorsiona por el desbalance de
#      clases (a diferencia de accuracy).
#   2. Ponderación de casos (case weights) en la regresión logística: los
#      casos de quiebra (minoría) se ponderan por el ratio
#      n_sanas/n_quiebras para que el estimador no colapse en predecir
#      siempre "no quiebra".
#   3. Evaluación con métricas robustas a la prevalencia baja: AUC-ROC,
#      AUC-PR (precision-recall), KS (Kolmogorov-Smirnov) y Gini — las
#      métricas estándar de la industria de credit scoring — en vez de
#      accuracy cruda. Se calcula explícitamente el "accuracy" del modelo
#      naive (predecir siempre "no quiebra") para mostrar por qué esa
#      métrica sola es engañosa en este problema.
#   4. Punto de corte (threshold) elegido por el estadístico de Youden en
#      el set de train, no el default de 0.5 (que en un problema con 3,2%
#      de prevalencia casi nunca se dispara).
#
# Paquetes usados (paquetes ya validados del ecosistema R, no
# reimplementación manual del método):
#   - scorecard (ShichenXie): binning WOE óptimo, cálculo de IV, filtro de
#     variables, construcción de la tabla de puntos del scorecard. Es EL
#     paquete de referencia en R para credit scoring estilo FICO/Basilea.
#   - pROC: curvas ROC, AUC, cálculo del punto de corte óptimo (Youden).
#   - randomForest: modelo de validación cruzada de variables predictivas.
#############################################################################

suppressPackageStartupMessages({
  library(here)
  library(scorecard)
  library(pROC)
  library(randomForest)
})

set.seed(20260915)
options(scipen = 100, digits = 4)

# here::here() resuelve la raíz del repo (detecta el .git al clonar), así
# que el script corre igual sin importar la máquina o el directorio desde
# el que se invoque.
proyecto_dir <- here::here()
dir.create(file.path(proyecto_dir, "results"), showWarnings = FALSE)
dir.create(file.path(proyecto_dir, "plots"), showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. Carga y limpieza mínima de datos
# ---------------------------------------------------------------------------

dt <- read.csv(file.path(proyecto_dir, "taiwan_bankruptcy_data.csv"),
                stringsAsFactors = FALSE, check.names = TRUE)
names(dt) <- trimws(names(dt))
names(dt)[names(dt) == "Bankrupt."] <- "Bankrupt"
dt$Bankrupt <- as.integer(dt$Bankrupt)

cat("Dimensiones del dataset:", nrow(dt), "empresas x", ncol(dt), "columnas\n")
tab_target <- table(dt$Bankrupt)
cat("Distribución de clases -> No quiebra (0):", tab_target["0"],
    "| Quiebra (1):", tab_target["1"],
    sprintf("(%.2f%% de quiebras)\n", 100 * tab_target["1"] / sum(tab_target)))

# ---------------------------------------------------------------------------
# 2. Split train/test estratificado (70/30)
# ---------------------------------------------------------------------------

idx_1 <- which(dt$Bankrupt == 1)
idx_0 <- which(dt$Bankrupt == 0)
train_1 <- sample(idx_1, size = round(0.7 * length(idx_1)))
train_0 <- sample(idx_0, size = round(0.7 * length(idx_0)))
train_idx <- c(train_1, train_0)

dt_train <- dt[train_idx, ]
dt_test  <- dt[-train_idx, ]

cat("\nTrain:", nrow(dt_train), "empresas (", sum(dt_train$Bankrupt), "quiebras ) | Test:",
    nrow(dt_test), "empresas (", sum(dt_test$Bankrupt), "quiebras )\n")

# ---------------------------------------------------------------------------
# 3. Filtro de variables por Information Value (IV) — primera pasada sobre
#    las 95 variables originales. iv_limit = 0.02 es el umbral estándar de
#    la industria de credit scoring para descartar variables sin poder
#    predictivo relevante (IV < 0.02 ~ "no predictivo").
# ---------------------------------------------------------------------------

dt_train_f <- var_filter(dt_train, y = "Bankrupt",
                          iv_limit = 0.02, missing_limit = 0.95,
                          identical_limit = 0.95, var_kp = NULL,
                          return_rm_reason = FALSE)

vars_preseleccionadas <- setdiff(names(dt_train_f), "Bankrupt")
cat("\nVariables preseleccionadas por IV >= 0.02 (de 95 originales):",
    length(vars_preseleccionadas), "\n")

# ---------------------------------------------------------------------------
# 4. Binning WOE óptimo sobre las variables preseleccionadas
# ---------------------------------------------------------------------------

bins <- woebin(dt_train_f, y = "Bankrupt", positive = "1",
               method = "tree", no_cores = 1)

iv_tabla <- do.call(rbind, lapply(names(bins), function(v) {
  data.frame(variable = v, IV = unique(bins[[v]]$total_iv))
}))
iv_tabla <- iv_tabla[order(-iv_tabla$IV), ]
row.names(iv_tabla) <- NULL

# ---------------------------------------------------------------------------
# 5. Reducir multicolinealidad: entre pares de variables con correlación
#    > 0.85 (en WOE), nos quedamos con la de mayor IV. Los ratios
#    financieros suelen venir en familias casi idénticas (ej. tres
#    versiones de "Net Value Per Share"), y meterlas todas juntas en la
#    regresión infla los coeficientes sin agregar poder predictivo real.
# ---------------------------------------------------------------------------

dt_train_woe <- as.data.frame(woebin_ply(dt_train_f, bins, to = "woe"))
dt_test_woe  <- as.data.frame(woebin_ply(dt_test[, c("Bankrupt", vars_preseleccionadas)], bins, to = "woe"))

woe_vars <- grep("_woe$", names(dt_train_woe), value = TRUE)
cor_mat <- cor(dt_train_woe[, woe_vars])
alta_corr <- which(abs(cor_mat) > 0.85 & abs(cor_mat) < 1, arr.ind = TRUE)

vars_descartadas_corr <- character(0)
if (nrow(alta_corr) > 0) {
  iv_lookup <- setNames(iv_tabla$IV, paste0(iv_tabla$variable, "_woe"))
  for (i in seq_len(nrow(alta_corr))) {
    v1 <- woe_vars[alta_corr[i, 1]]
    v2 <- woe_vars[alta_corr[i, 2]]
    if (v1 %in% vars_descartadas_corr || v2 %in% vars_descartadas_corr) next
    perder <- if (iv_lookup[v1] < iv_lookup[v2]) v1 else v2
    vars_descartadas_corr <- union(vars_descartadas_corr, perder)
  }
}
woe_vars_finales <- setdiff(woe_vars, vars_descartadas_corr)
cat("\nVariables descartadas por alta correlación (>0.85) con otra de mayor IV:",
    length(vars_descartadas_corr), "\n")
cat("Variables finales para el modelo:", length(woe_vars_finales), "\n")

# ---------------------------------------------------------------------------
# 6. Regresión logística ponderada por clase (manejo del desbalance),
#    con selección stepwise (AIC) sobre las variables finales en WOE.
# ---------------------------------------------------------------------------

peso_quiebra <- sum(dt_train_woe$Bankrupt == 0) / sum(dt_train_woe$Bankrupt == 1)
pesos_train <- ifelse(dt_train_woe$Bankrupt == 1, peso_quiebra, 1)
cat(sprintf("\nPeso aplicado a casos de quiebra en la regresión: %.1fx (para compensar el 96.8%%/3.2%% de desbalance)\n",
            peso_quiebra))

formula_completa <- as.formula(paste("Bankrupt ~", paste(woe_vars_finales, collapse = " + ")))
modelo_completo <- glm(formula_completa, data = dt_train_woe, family = binomial(),
                        weights = pesos_train)
modelo_final <- step(modelo_completo, direction = "both", trace = 0)

cat("\n--- Variables retenidas en el modelo final (regresión logística stepwise) ---\n")
print(summary(modelo_final)$coefficients)

vars_modelo_final <- sub("_woe$", "", names(coef(modelo_final))[-1])

# ---------------------------------------------------------------------------
# 7. Scorecard: escala de puntos (convención PDO estándar de la industria)
#    points0 = 600 a odds0 = 1/19 (~95% de probabilidad de "no quiebra"),
#    PDO = 50 (los puntos se duplican el odds cada 50 puntos). Es la misma
#    lógica de escala que usan los scorecards estilo FICO; el cliente puede
#    recalibrar points0/pdo a la escala de rating interna que prefiera sin
#    cambiar el modelo subyacente.
# ---------------------------------------------------------------------------

card <- scorecard(bins, modelo_final, points0 = 600, odds0 = 1 / 19, pdo = 50)

tabla_puntos <- do.call(rbind, card[setdiff(names(card), "basepoints")])
tabla_puntos <- tabla_puntos[, c("variable", "bin", "woe", "points")]
row.names(tabla_puntos) <- NULL

score_train <- scorecard_ply(dt_train, card, only_total_score = TRUE)
score_test  <- scorecard_ply(dt_test, card, only_total_score = TRUE)

dt_train$score <- score_train$score
dt_test$score  <- score_test$score

# ---------------------------------------------------------------------------
# 8. Evaluación del modelo: AUC, KS, Gini, AUC-PR, y comparación explícita
#    contra el "modelo naive" (predecir siempre no-quiebra) para mostrar por
#    qué accuracy sola es una métrica engañosa en este problema.
# ---------------------------------------------------------------------------

prob_train <- predict(modelo_final, newdata = dt_train_woe, type = "response")
prob_test  <- predict(modelo_final, newdata = dt_test_woe, type = "response")

roc_train <- roc(dt_train$Bankrupt, prob_train, quiet = TRUE)
roc_test  <- roc(dt_test$Bankrupt, prob_test, quiet = TRUE)

auc_train <- as.numeric(auc(roc_train))
auc_test  <- as.numeric(auc(roc_test))
gini_train <- 2 * auc_train - 1
gini_test  <- 2 * auc_test - 1

# KS: máxima distancia entre distribuciones acumuladas de score de buenos y malos
ks_stat <- function(score, y) {
  suppressWarnings(ks.test(score[y == 1], score[y == 0])$statistic)
}
ks_train <- ks_stat(dt_train$score, dt_train$Bankrupt)
ks_test  <- ks_stat(dt_test$score, dt_test$Bankrupt)

# AUC-PR (precision-recall), más informativa que ROC bajo desbalance extremo
pr_auc <- function(prob, y) {
  ord <- order(-prob)
  y_ord <- y[ord]
  tp <- cumsum(y_ord == 1)
  fp <- cumsum(y_ord == 0)
  precision <- tp / (tp + fp)
  recall <- tp / sum(y == 1)
  sum(diff(c(0, recall)) * precision, na.rm = TRUE)
}
prauc_train <- pr_auc(prob_train, dt_train$Bankrupt)
prauc_test  <- pr_auc(prob_test, dt_test$Bankrupt)

# Umbral óptimo por estadístico de Youden en TRAIN, aplicado a TEST
umbral_youden <- coords(roc_train, "best", best.method = "youden", ret = "threshold")
umbral_youden <- as.numeric(umbral_youden[1])
pred_test_clase <- ifelse(prob_test >= umbral_youden, 1, 0)

matriz_confusion <- table(Real = dt_test$Bankrupt, Predicho = pred_test_clase)
sensibilidad <- matriz_confusion["1", "1"] / sum(matriz_confusion["1", ])
especificidad <- matriz_confusion["0", "0"] / sum(matriz_confusion["0", ])
accuracy_modelo <- sum(diag(matriz_confusion)) / sum(matriz_confusion)
accuracy_naive <- sum(dt_test$Bankrupt == 0) / nrow(dt_test)

metricas <- data.frame(
  metrica = c("AUC-ROC", "Gini", "KS", "AUC-PR", "Umbral (Youden)",
              "Sensibilidad (Recall quiebras)", "Especificidad",
              "Accuracy del modelo", "Accuracy modelo naive (predecir siempre 'no quiebra')"),
  train = c(auc_train, gini_train, ks_train, prauc_train, NA, NA, NA, NA, NA),
  test  = c(auc_test, gini_test, ks_test, prauc_test, umbral_youden,
            sensibilidad, especificidad, accuracy_modelo, accuracy_naive)
)

cat("\n--- Métricas de desempeño ---\n")
print(metricas, row.names = FALSE)
cat("\nMatriz de confusión (test, umbral =", round(umbral_youden, 4), "):\n")
print(matriz_confusion)

# ---------------------------------------------------------------------------
# 9. Random Forest de validación cruzada de variables predictivas (no es el
#    modelo final, solo un chequeo de robustez independiente del enfoque
#    WOE/logístico). Balanceo vía sampsize estratificado (submuestreo de la
#    clase mayoritaria en cada árbol) para que el ranking de importancia no
#    quede dominado por la clase mayoritaria.
# ---------------------------------------------------------------------------

dt_rf <- dt_train_f
dt_rf$Bankrupt <- factor(dt_rf$Bankrupt, levels = c(0, 1), labels = c("No", "Si"))
n_min <- sum(dt_rf$Bankrupt == "Si")

rf_modelo <- randomForest(
  Bankrupt ~ ., data = dt_rf,
  ntree = 500,
  sampsize = c(No = n_min, Si = n_min),
  strata = dt_rf$Bankrupt,
  importance = TRUE
)

importancia_rf <- data.frame(
  variable = rownames(importance(rf_modelo)),
  MeanDecreaseGini = importance(rf_modelo)[, "MeanDecreaseGini"]
)
importancia_rf <- importancia_rf[order(-importancia_rf$MeanDecreaseGini), ]
row.names(importancia_rf) <- NULL

cat("\n--- Top 10 variables según Random Forest (chequeo de robustez) ---\n")
print(head(importancia_rf, 10), row.names = FALSE)

# ---------------------------------------------------------------------------
# 10. Grados de riesgo (rating interno) a partir del score, en quintiles del
#     score de test, con la tasa de quiebra observada en cada grado.
# ---------------------------------------------------------------------------

cortes <- quantile(dt_test$score, probs = seq(0, 1, 0.2))
dt_test$grado <- cut(dt_test$score, breaks = unique(cortes),
                      include.lowest = TRUE,
                      labels = paste0("Grado ", 5:1))  # score alto = mejor => Grado 1 el mejor
# Reordenar etiquetas: score más alto = mejor calidad crediticia
niveles_ordenados <- c("Grado 1", "Grado 2", "Grado 3", "Grado 4", "Grado 5")
dt_test$grado <- factor(dt_test$grado, levels = rev(niveles_ordenados))

tabla_grados <- aggregate(cbind(n_empresas = rep(1, nrow(dt_test)),
                                 n_quiebras = dt_test$Bankrupt,
                                 score_min = dt_test$score,
                                 score_max = dt_test$score,
                                 score_prom = dt_test$score) ~ grado, data = dt_test,
                           FUN = function(x) x)
# aggregate con FUN=x no da min/max por columna limpio; se recalcula manualmente:
tabla_grados <- do.call(rbind, lapply(split(dt_test, dt_test$grado), function(sub) {
  data.frame(
    grado = unique(sub$grado),
    n_empresas = nrow(sub),
    n_quiebras = sum(sub$Bankrupt),
    tasa_quiebra_pct = round(100 * mean(sub$Bankrupt), 2),
    score_min = round(min(sub$score), 0),
    score_max = round(max(sub$score), 0),
    score_promedio = round(mean(sub$score), 0)
  )
}))
tabla_grados <- tabla_grados[order(-tabla_grados$score_promedio), ]
row.names(tabla_grados) <- NULL

cat("\n--- Grados de riesgo (rating interno) y tasa de quiebra observada en test ---\n")
print(tabla_grados, row.names = FALSE)

# ---------------------------------------------------------------------------
# 11. Guardar resultados (CSV)
# ---------------------------------------------------------------------------

write.csv(iv_tabla, file.path(proyecto_dir, "results", "01_iv_ranking_variables.csv"), row.names = FALSE)
write.csv(tabla_puntos, file.path(proyecto_dir, "results", "02_scorecard_puntos.csv"), row.names = FALSE)
write.csv(metricas, file.path(proyecto_dir, "results", "03_metricas_desempeno.csv"), row.names = FALSE)
write.csv(tabla_grados, file.path(proyecto_dir, "results", "04_grados_de_riesgo.csv"), row.names = FALSE)
write.csv(importancia_rf, file.path(proyecto_dir, "results", "05_importancia_random_forest.csv"), row.names = FALSE)

resultados_empresa <- data.frame(
  id = 1:nrow(dt_test),
  score = dt_test$score,
  grado = dt_test$grado,
  probabilidad_quiebra_modelo = round(prob_test, 4),
  quiebra_real = dt_test$Bankrupt
)
write.csv(resultados_empresa, file.path(proyecto_dir, "results", "06_scores_empresas_test.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 12. Gráficos
# ---------------------------------------------------------------------------

# Traducción al español de las 15 variables que entran al gráfico de IV
# (los nombres crudos de columna del dataset original vienen en inglés)
trad_variables_iv <- c(
  "Interest.Expense.Ratio" = "Ratio de gastos por intereses",
  "Continuous.interest.rate..after.tax." = "Tasa de interés continua después de impuestos",
  "ROA.A..before.interest.and...after.tax" = "ROA (A) - antes de intereses y después de impuestos",
  "ROA.B..before.interest.and.depreciation.after.tax" = "ROA (B) - antes de intereses y depreciación, después de impuestos",
  "Persistent.EPS.in.the.Last.Four.Seasons" = "EPS persistente (últimos 4 trimestres)",
  "Per.Share.Net.profit.before.tax..Yuan..." = "Utilidad neta por acción antes de impuestos (Yuan)",
  "ROA.C..before.interest.and.depreciation.before.interest" = "ROA (C) - antes de intereses y depreciación, antes de intereses",
  "Net.profit.before.tax.Paid.in.capital" = "Utilidad neta antes de impuestos / Capital social",
  "Interest.Coverage.Ratio..Interest.expense.to.EBIT." = "Cobertura de intereses (gasto por intereses / EBIT)",
  "After.tax.net.Interest.Rate" = "Tasa de interés neta después de impuestos",
  "Degree.of.Financial.Leverage..DFL." = "Grado de apalancamiento financiero (DFL)",
  "Total.debt.Total.net.worth" = "Deuda total / Patrimonio neto",
  "Pre.tax.net.Interest.Rate" = "Tasa de interés neta antes de impuestos",
  "Debt.ratio.." = "Ratio de endeudamiento (deuda / activos)",
  "Net.worth.Assets" = "Patrimonio neto / Activos"
)

# Gráfico 1: curva ROC (train y test)
png(file.path(proyecto_dir, "plots", "grafico_curva_roc.png"), width = 900, height = 750, res = 120)
plot(roc_test, col = "darkred", lwd = 2, main = "Curva ROC — Modelo de quiebra corporativa",
     xlab = "Especificidad", ylab = "Sensibilidad")
lines(roc_train, col = "steelblue", lwd = 2, lty = 2)
legend("bottomright",
       legend = c(sprintf("Prueba (AUC = %.3f)", auc_test),
                  sprintf("Entrenamiento (AUC = %.3f)", auc_train)),
       col = c("darkred", "steelblue"), lwd = 2, lty = c(1, 2), cex = 0.85)
dev.off()

# Gráfico 2: Top 15 variables por Information Value
png(file.path(proyecto_dir, "plots", "grafico_top_variables_iv.png"), width = 1100, height = 800, res = 120)
top_iv <- head(iv_tabla, 15)
etiquetas_iv_es <- ifelse(top_iv$variable %in% names(trad_variables_iv),
                           trad_variables_iv[top_iv$variable], top_iv$variable)
par(mar = c(5, 24, 4, 2))
barplot(rev(top_iv$IV), horiz = TRUE, names.arg = rev(etiquetas_iv_es),
        las = 1, cex.names = 0.62, col = "steelblue",
        xlab = "Information Value (IV)",
        main = "Top 15 ratios financieros por poder predictivo (IV)")
abline(v = 0.1, col = "darkgreen", lty = 2)
abline(v = 0.3, col = "darkorange", lty = 2)
text(0.1, 1, "IV=0.1\n(fuerte)", col = "darkgreen", cex = 0.6, pos = 4)
text(0.3, 1, "IV=0.3\n(muy fuerte)", col = "darkorange", cex = 0.6, pos = 4)
dev.off()

# Gráfico 3: distribución del score por clase real (sanas vs quiebra)
png(file.path(proyecto_dir, "plots", "grafico_distribucion_score.png"), width = 1000, height = 700, res = 120)
plot(density(dt_test$score[dt_test$Bankrupt == 0]), col = "darkgreen", lwd = 2,
     main = "Distribución del score del scorecard por clase real (test)",
     xlab = "Score del scorecard (más alto = menor riesgo)",
     ylab = "Densidad",
     ylim = c(0, max(density(dt_test$score[dt_test$Bankrupt == 0])$y,
                      density(dt_test$score[dt_test$Bankrupt == 1])$y) * 1.1))
lines(density(dt_test$score[dt_test$Bankrupt == 1]), col = "firebrick", lwd = 2)
legend("topleft", legend = c("Empresas sanas (no quiebra)", "Empresas en quiebra"),
       col = c("darkgreen", "firebrick"), lwd = 2, cex = 0.85)
dev.off()

# Gráfico 4: tasa de quiebra observada por grado de riesgo
png(file.path(proyecto_dir, "plots", "grafico_tasa_quiebra_por_grado.png"), width = 950, height = 700, res = 120)
bp <- barplot(tabla_grados$tasa_quiebra_pct,
              names.arg = paste0(tabla_grados$grado, "\n(score ", tabla_grados$score_min,
                                  "-", tabla_grados$score_max, ")"),
              col = colorRampPalette(c("darkgreen", "goldenrod", "firebrick"))(nrow(tabla_grados)),
              ylab = "Tasa de quiebra observada (%)", cex.names = 0.75,
              main = "Tasa de quiebra observada por grado de riesgo (test)")
text(bp, tabla_grados$tasa_quiebra_pct, labels = paste0(tabla_grados$tasa_quiebra_pct, "%"),
     pos = 3, cex = 0.8)
dev.off()

cat("\n============================================================\n")
cat("Fin del script. Archivos guardados en:\n", proyecto_dir, "\n")
cat("============================================================\n")
