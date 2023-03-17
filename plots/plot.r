#!/usr/bin/env Rscript
library(ggplot2)
library(stringr)
library(dplyr)
library(tikzDevice)
library(scales)
library(ggh4x)
library(patchwork)
library(tidyr)

source("plots/layout.r")
options(tikzLatexPackages = c(getOption("tikzLatexPackages"),
                              "\\usepackage{amsmath}"))

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
    result_path = "experimental_analysis/logs/paper_results"
} else if (length(args)==1) {
    result_path = args[1]
} else {
    stop("Usage: Rscript plots/plot.r [<result_path>]")
}

create_save_locations()

max_episodes <- 19000
plot_limit <- 75

load_and_label_data <- function(file) {
    data <- read.csv(file, stringsAsFactors = FALSE)
    if(ncol(data) == 0 | nrow(data) == 0) {
        return(data)
    }

    if (grepl("critic", file, fixed = TRUE)) {
        data$mode <- "Q-Critic"
    } else if (grepl("agent", file, fixed = TRUE)) {
        data$mode <- "Q-Actor"
    } else if (grepl("quantum", file, fixed = TRUE)) {
        data$mode <- "Fully Quantum"
    } else {
        data$mode <- "Classical"
    }

    if (grepl("rels4", file, fixed = TRUE)) {
        data$num_rels <- 4
    } else if (grepl("rels5", file, fixed = TRUE)) {
        data$num_rels <- 5
    } else {
        data$num_rels <- 17
    }


    n_l <- as.numeric(gsub(".*num_layers([0-9]+)/.*$", "\\1", file))
    if (is.na(n_l)) {
        data$l <- 0
    } else {
        data$l <- n_l
    }

    num_episodes <- as.numeric(gsub(".*num_episodes([0-9]+)/.*$", "\\1", file))
    if (is.na(n_l)) {
        data$num_episodes <- 0
    } else {
        data$num_episodes <- num_episodes
    }

    if (grepl("/nodes128/", file, fixed = TRUE)) {
        data$nodes <- 128
    } else if (grepl("/nodes256/", file, fixed = TRUE)) {
        data$nodes <- 256
    } else if (grepl("/nodes384/", file, fixed = TRUE)) {
        data$nodes <- 384
    } else {
        data$nodes <- 0
    }

    if (grepl("/data_reupl/", file, fixed = TRUE)) {
        data$data_reuploading <- TRUE
    } else {
        data$data_reuploading <- FALSE
    }

    return(data)
}

load_and_label_noisy_data <- function(file) {
    data <- read.csv(file, stringsAsFactors = FALSE)
    if(ncol(data) == 0 | nrow(data) == 0) {
        return(data)
    }

    if (grepl("critic", file, fixed = TRUE)) {
        data$mode <- "Q-Critic"
    } else if (grepl("agent", file, fixed = TRUE)) {
        data$mode <- "Q-Actor"
    } else if (grepl("quantum", file, fixed = TRUE)) {
        data$mode <- "Fully Quantum"
    } else {
        data$mode <- "Classical"
    }

    if (grepl("rels4", file, fixed = TRUE)) {
        data$num_rels <- 4
    } else if (grepl("rels5", file, fixed = TRUE)) {
        data$num_rels <- 5
    } else {
        data$num_rels <- 17
    }


    n_l <- as.numeric(gsub(".*num_layers([0-9]+)/.*$", "\\1", file))
    if (is.na(n_l)) {
        data$l <- 0
    } else {
        data$l <- n_l
    }

    if (grepl("/data_reupl/", file, fixed = TRUE)) {
        data$data_reuploading <- TRUE
    } else {
        data$data_reuploading <- FALSE
    }

    if (grepl("depol_error_prob00/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0
    } else if (grepl("depol_error_prob001/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0.01
    } else if (grepl("depol_error_prob002/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0.02
    } else if (grepl("depol_error_prob003/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0.03
    } else if (grepl("depol_error_prob004/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0.04
    } else if (grepl("depol_error_prob005/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0.05
    } else if (grepl("depol_error_prob01/", file, fixed = TRUE)) {
        data$depol_error_prob <- 0.1
    }

    return(data)
}

load_training_data_from_dir <- function(root) {
    files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

    data <- load_and_label_data(files[7])

    for (i in seq_along(files)) {
        if(grepl("query_val_train_avg", files[i]) & i!=7) {
            data_new <- load_and_label_data(files[i])
            if(ncol(data_new) == 0 | nrow(data_new) == 0) {
                next
            }
            data <- rbind(data, data_new)
        }
    }
    return(data)
}

load_val_data_from_dir <- function(root) {
    files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

    data_val <- load_and_label_data(files[6])
    data_val <- remove_duplicate_last_row(data_val)

    for (i in seq_along(files)) {
        if(grepl("query_val.csv", files[i]) & i!=6) {
            data_new <- load_and_label_data(files[i])
            data_new <- remove_duplicate_last_row(data_new)
            if(ncol(data_new) == 0 | nrow(data_new) == 0) {
                next
            }
            data_val <- rbind(data_val, data_new)
        }
    }
    return(data_val)
}

load_noisy_val_data_from_dir <- function(root) {
    files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

    data_val <- load_and_label_noisy_data(files[1])
    data_val <- remove_duplicate_last_row(data_val)

    for (i in seq_along(files)) {
        if(grepl("query_val.csv", files[i]) & i!=1) {
            data_new <- load_and_label_noisy_data(files[i])
            data_new <- remove_duplicate_last_row(data_new)
            if(ncol(data_new) == 0 | nrow(data_new) == 0) {
                next
            }
            data_val <- rbind(data_val, data_new)
        }
    }
    return(data_val)
}

remove_duplicate_last_row <- function(d) {
   if (tail(d, n=1)$episode == head(tail(d, n=2), n=1)$episode) {
       d <- head(d, -1)
   }
    return(d)
}

select_best_run_from_dir <- function(root) {
    files <- list.files(path = root, full.names = TRUE, recursive = TRUE)

    min_data <- load_and_label_data(files[5])
    #min_data <- remove_duplicate_last_row(min_data)
    d <- load_and_label_data(files[7])
    last5 <- min_data %>% tail(n=3)
    min_avg <- mean(last5$mrc)

    for (i in seq_along(files)) {
        if(grepl("query_val_avg.csv", files[i]) & i!=5) {
            data_ref_new <- load_and_label_data(files[i])
            if(ncol(data_ref_new) == 0 | nrow(data_ref_new) == 0) {
                next
            }
            if (tail(data_ref_new, n=1)$episode < max_episodes) {
                next
            }
            #data_new <- remove_duplicate_last_row(data_new)
            last5 <- data_ref_new %>% tail(n=3)
            curr_avg <- mean(last5$mrc)
            if (ncol(min_data) == 0 | nrow(min_data) == 0) {
                min_avg <- curr_avg
                min_data <- data_ref_new
            }
            else if (tail(min_data, n=1)$episode < max_episodes | curr_avg < min_avg) {
                d <- load_and_label_data(files[i+2])
                min_avg <- curr_avg
                min_data <- data_ref_new
            }
        }
    }
    return(d)
}

do_some_statistics <- function(d, geqo=FALSE, group = c("episode")) {
  if(geqo) {
      d_geqo <- d %>%
        group_by(across(all_of(group))) %>%
        summarise(min = min(geqo_mrc),
                    quartile1 = quantile(geqo_mrc, 0.25),
                    median = median(geqo_mrc),
                    quartile3 = quantile(geqo_mrc, 0.75),
                    max = max(geqo_mrc))
      d_geqo$mode <- "GEQO"
      d_geqo$quartile1 <- d_geqo$quartile1[1]
      d_geqo$quartile3 <- d_geqo$quartile3[1]
      d_geqo$median <- d_geqo$median[1]
      d_geqo$min <- d_geqo$min[1]
      d_geqo$max <- d_geqo$max[1]
  }

  d <- d %>%
    group_by(across(all_of(group))) %>%
    summarise(min = min(mrc),
              quartile1 = quantile(mrc, 0.25),
              median = median(mrc),
              quartile3 = quantile(mrc, 0.75),
              max = max(mrc))
  if(geqo) {
    d$mode <- "\\emph{ReJoin}"
    d <- rbind(d, d_geqo)
  }

  d$max[d$max > plot_limit] <- plot_limit
  d$quartile3[d$quartile3 > plot_limit] <- plot_limit
  d$min[d$min < 1] <- 1
  d$quartile1[d$quartile1 < 1] <- 1
  d$median[d$median < 1] <- 1
  d$median[d$median > plot_limit] <- plot_limit

  return(d)
}

latex_percent <- function (x) {
    str_c(x, "\\%")
}

default_plot <- function(d) {
    return(ggplot(d, aes(color=mode, fill=mode)) +
#      geom_ribbon(mapping = aes(x = episode, ymin = min, ymax = max, alpha=0.1),
#                  alpha = 0.1, colour = NA) +
      geom_ribbon(mapping = aes(x = episode, ymin = quartile1, ymax = quartile3),
                  alpha = 0.4, colour = NA) +
      geom_line(aes(x = episode, y = median), linewidth = LINE.SIZE) +
      geom_hline(aes(yintercept = 1, colour="DP"), linetype="dashed", linewidth = LINE.SIZE) +
      labs(x="Episode", y="$C^{(\\text{CM})}/C_{\\text{DP}}^{(\\text{CM})}$ (log)") +
      scale_colour_manual(values=COLOURS.LIST, name="", breaks=c("DP", "Classical", "Fully Quantum", "Q-Actor", "Q-Critic","GEQO", "\\emph{ReJoin}")) +
      scale_fill_manual(values=COLOURS.LIST, name="", breaks=c("DP","Classical", "Fully Quantum", "Q-Actor", "Q-Critic", "GEQO", "\\emph{ReJoin}")) +
      scale_y_continuous(trans=log_trans(), breaks = c(1, 2, 5, 10, 20, 50)) +
      scale_x_continuous(labels = scales::label_number(suffix = "k", scale=1e-3)))
}

d_pg8 <- load_val_data_from_dir(str_c(result_path, "/xval/hyper/classical/PGCM8/basic_rejoin_shift.py/num_episodes20/mini_batchsize32/best_frequency20000/nodes128/lr_start9e-05/"))
d_pg8 <- do_some_statistics(d_pg8, TRUE)
d_pg8$mode[d_pg8$mode != "GEQO"] <- "Baseline"
d_pg8$cm <- "\\emph{PG8}"

d_baseline_val <- load_val_data_from_dir(str_c(result_path, "/xval/hyper/classical/PGCM16/basic_rejoin_shift.py"))
d_baseline_val <- do_some_statistics(d_baseline_val)
d_baseline_val$mode <- "Baseline"
d_mod_val <- load_val_data_from_dir(str_c(result_path, "/xval/hyper/classical/PGCM16/mod_reduced_rejoin_384.py"))
d_mod_val <- do_some_statistics(d_mod_val)
d_mod_val$mode <- "Modified \\& Reduced"
d_pg16 <- rbind(d_baseline_val, d_mod_val)
d_pg16$cm <- "\\emph{PG16}"

d_baseline_val <- load_val_data_from_dir(str_c(result_path, "/xval/hyper/classical/COUTCM/basic_rejoin_shift.py"))
d_baseline_val <- do_some_statistics(d_baseline_val)
d_baseline_val$mode <- "Baseline"
d_mod_val <- load_val_data_from_dir(str_c(result_path, "/xval/hyper/classical/COUTCM/mod_reduced_rejoin_384.py"))
d_mod_val <- do_some_statistics(d_mod_val)
d_mod_val$mode <- "Modified \\& Reduced"

d_out <- rbind(d_baseline_val, d_mod_val)
d_out$cm <- "\\emph{OUT}"

d <- rbind(d_pg8, d_pg16)
d <- rbind(d, d_out)

cm_labeller <- function(cm) {
    paste0("CM = ", cm)
}

cm_levels <- c("\\emph{PG8}", "\\emph{PG16}", "\\emph{OUT}")

g <- default_plot(d) + theme_paper_base_no_shrink() +
  facet_grid(. ~ factor(cm, levels=cm_levels), labeller=as_labeller(cm_labeller)) +
  guides(fill="none") +
  scale_colour_manual(values=COLOURS.LIST, name="", breaks=c("DP", "Baseline", "Modified \\& Reduced", "GEQO")) +
  scale_fill_manual(values=COLOURS.LIST, name="", breaks=c("DP", "Baseline", "Modified \\& Reduced", "GEQO"))

save_name <- "convergence_rejoin_base"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.5*COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.5*COLWIDTH)
print(g)
dev.off()

d_cl <- load_val_data_from_dir(str_c(result_path, "/xval/hyper_search/hyper_classical"))
d_cl <- do_some_statistics(d_cl, FALSE, c("episode", "mode"))
d_critic <- load_val_data_from_dir(str_c(result_path, "/xval/hyper_search/hyper_quantum_critic/rels4/depol_error_prob00/num_layers20/data_reupl"))
d_critic <- do_some_statistics(d_critic, FALSE, c("episode", "mode"))
d <- rbind(d_critic, d_cl)
d_agent <- load_val_data_from_dir(str_c(result_path, "/xval/hyper_search/hyper_quantum_agent/rels4/depol_error_prob00/num_layers20/data_reupl"))
d_agent <- do_some_statistics(d_agent, FALSE, c("episode", "mode"))
d <- rbind(d_agent, d)
d_base <- load_val_data_from_dir(str_c(result_path, "/xval/hyper_search/hyper_quantum_base/rels4/depol_error_prob00/num_layers20/data_reupl"))
d_base <- do_some_statistics(d_base, FALSE, c("episode", "mode"))
d <- rbind(d_base, d)

d$num_rels <- 4
files <- list.files(path = str_c(result_path, "/QML4JOO/NoReUploading"), full.names = TRUE, recursive = TRUE)
files <- files[grepl("slow", files, fixed=TRUE)]
files_reupl <- list.files(path = str_c(result_path, "/QML4JOO/ReUploading"), full.names = TRUE, recursive = TRUE)

c_names <- c("episode", "mean_reward", "median_reward", "mean_log_reward", "median_log_reward", "mean_lin_reward", "median_lin_reward",
             "num_best", "num_worst", "num_invalid", "num_x_joins", "num_queries", "rewards")
d_tobias <- read.csv(files[1], stringsAsFactors=FALSE, header=FALSE)
n_l <- as.numeric(gsub(".*/([0-9]+)layers.*$", "\\1", files[1]))
colnames(d_tobias) <- c_names
d_tobias$l <- n_l
d_tobias$split <- 1
for (i in seq_along(files)) {
  if (i==1) {
    next
  }
  d_new <- read.csv(files[i], stringsAsFactors=FALSE, header=FALSE)
  n_l <- as.numeric(gsub(".*/([0-9]+)layers.*$", "\\1", files[i]))
  colnames(d_new) <- c_names
  d_new$l <- n_l
  d_new$split <- i
  d_tobias <- rbind(d_tobias, d_new)
}
d_tobias$data_reuploading <- FALSE

d_tobias_reupl <- read.csv(files_reupl[1], stringsAsFactors=FALSE, header=FALSE)
colnames(d_tobias_reupl) <- c_names
n_l <- as.numeric(gsub(".*/ReupRed_([0-9]+)layers.*$", "\\1", files_reupl[1]))
d_tobias_reupl$l <- n_l
d_tobias_reupl$split <- 1
for (i in seq_along(files_reupl)) {
  if (i==1) {
    next
  }
  d_new <- read.csv(files_reupl[i], stringsAsFactors=FALSE, header=FALSE)
  n_l <- as.numeric(gsub(".*/ReupRed_([0-9]+)layers.*$", "\\1", files_reupl[i]))
  colnames(d_new) <- c_names
  d_new$l <- n_l
  d_new$split <- i
  d_tobias_reupl <- rbind(d_tobias_reupl, d_new)
}
d_tobias_reupl$data_reuploading <- TRUE
d_tobias <- rbind(d_tobias, d_tobias_reupl)

d_tobias <- d_tobias %>% separate_rows(rewards, sep=";")
d_tobias$mrc <- 1 / as.double(d_tobias$rewards)
d_tobias$mode <- "Single Step QML"

d_tobias <- d_tobias %>% group_by(episode, l, mode, data_reuploading) %>%
    summarise(min = min(mrc),
              quartile1 = quantile(mrc, 0.25),
              median = median(mrc),
              quartile3 = quantile(mrc, 0.75),
              max = max(mrc)) %>%
    ungroup() %>%
    filter(episode %% 500 == 0 | episode == 19900)
d_tobias20 <- d_tobias %>% filter(l == 20 & data_reuploading)
d <- bind_rows(d, d_tobias20)

mode_levels <- c("DP", "Classical", "Q-Critic", "Q-Actor", "Fully Quantum", "Single Step QML")

g <- ggplot(d, aes(color=mode, fill=mode)) +
  #geom_ribbon(mapping = aes(x = episode, ymin = quartile1, ymax = quartile3),
  #            alpha = 0.2, colour = NA) +
  geom_line(aes(x = episode, y = median), linewidth = LINE.SIZE) +
  geom_hline(aes(yintercept = 1, colour="DP"), linetype="dashed", linewidth = LINE.SIZE) +
  labs(x="Episode", y="$C/C_{\\text{DP}}$") +
  scale_colour_manual(values=COLOURS.LIST, name="", breaks=c("DP", "Classical", "Fully Quantum", "Q-Actor", "Q-Critic", "Single Step QML")) +
  scale_fill_manual(values=COLOURS.LIST, name="", breaks=c("DP","Classical", "Fully Quantum", "Q-Actor", "Q-Critic", "Single Step QML")) +
  scale_y_continuous(trans=log_trans(), breaks = c(1, 2, 3)) +
  scale_x_continuous(labels = scales::label_number(suffix = "k", scale=1e-3)) +
  # facet_grid(. ~ factor(mode, levels = mode_levels)) +
  theme_paper_legend_right() +
  theme(legend.key.size = unit(3, 'mm'),
        legend.margin=margin(0, -1, 0, -1, "mm")) +
  guides(fill="none") +
  labs(x="Episode", y="$C/C_{\\text{DP}}$ (log)")

save_name <- "convergence_rejoin_q_4rels"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.45 * COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.45 * COLWIDTH)
print(g)
dev.off()

d <- load_val_data_from_dir(str_c(result_path, "/xval/hyper_search/"))
d <- do_some_statistics(d, FALSE, c("episode", "mode", "l", "data_reuploading"))
d <- d %>% filter(mode != "Classical")
d <- rbind(d, d_tobias)

layer_labeller <- function(layer) {
    paste0("\\# Layers = ", layer)
}

dr_labeller <- function(layer) {
  c("Without DRU", "With DRU")
}

g <- ggplot(d, aes(colour=factor(l), fill=mode)) +
      geom_line(aes(x = episode, y = median), linewidth = LINE.SIZE) +
      geom_hline(aes(yintercept = 1, colour="DP"), linetype="dashed", linewidth = LINE.SIZE) +
      scale_colour_manual(values=COLOURS.LIST, name="\\# Layers", breaks=c("DP", "4", "8", "12", "16", "20")) +
      scale_y_continuous(trans=log_trans(), breaks = c(1, 2, 3)) +
      scale_x_continuous(labels = scales::label_number(suffix = "k", scale=1e-3), breaks = c(5000, 15000)) +
      theme_paper_base() +
      guides(fill="none", colour=guide_legend(nrow=1)) +
      labs(x="Episode", y="$C/C_{\\text{DP}}$ (log)") +
      facet_grid(data_reuploading ~ factor(mode, levels=mode_levels), labeller=labeller(
                 data_reuploading = as_labeller(dr_labeller)))

save_name <- "convergence_rejoin_q_layers"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.65 * COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.65 * COLWIDTH)
print(g)
dev.off()

d <- read.csv("scalability/num_params.csv", stringsAsFactors = FALSE)

d <- d %>%
    group_by(num_relations, mode, num_repetitions, num_params, circuit_depth) %>%
    summarise(min = min(t_adam),
              quartile1 = quantile(t_adam, 0.25),
              median = median(t_adam),
              quartile3 = quantile(t_adam, 0.75),
              max = max(t_adam)) %>%
    ungroup()

d_time_filtered <- d %>%
    filter(num_repetitions == 0 | num_repetitions == 1)

g_opt <- ggplot(d_time_filtered, mapping=aes(colour=mode, fill=mode)) +
      geom_ribbon(mapping = aes(x = num_relations, ymin = quartile1, ymax = quartile3),
                  alpha = 0.4, colour = NA) +
      geom_line(aes(x = num_relations, y = median), linewidth = LINE.SIZE) +
      scale_colour_manual(values=COLOURS.LIST, name="") +
      scale_fill_manual(values=COLOURS.LIST, name="") +
      scale_y_continuous() +
      labs(x="", y="$t_\\text{opt}$ [ms]") +
      theme_paper_no_legend_small() +
      theme(axis.title.x = element_blank(),
            plot.margin = unit(c(0,0,0,0), 'cm'),
            panel.background = element_rect(fill = "transparent"))

d_q <- d %>% filter(num_repetitions %in% c(0, 1, 5, 10))
d_c <- d %>% filter(mode == "Classical")

g <- ggplot(d_q, mapping=aes(colour=mode, linetype=factor(num_repetitions))) +
    geom_line(aes(x = num_relations, y = num_params), linewidth = LINE.SIZE) +
    scale_colour_manual(values=COLOURS.LIST, name="") +
    scale_linetype_discrete(breaks=c("1", "5", "10"), name="\\# DRU Repetitions") +
    labs(x="\\# Relations", y="\\# Trainable Parameters") +
    theme_paper_legend_right() +
    scale_y_continuous(labels = scales::label_number(suffix = "k", scale=1e-3)) +
    guides(linetype=guide_legend(title.position="top", ncol=1, nrow=3, byrow=TRUE)) +
    guides(colour=guide_legend(ncol=1,nrow=4,byrow=TRUE)) +
    theme(plot.margin = unit(c(0,0,0,0), 'cm'),
          legend.key.size = unit(3, 'mm'),
          legend.margin=margin(0, -3, 0, -1, "mm")) +
    inset_element(g_opt, 0.02, 0.5, 0.6, 0.95)

save_name <- "num_params"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.55*COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.55*COLWIDTH)
print(g)
dev.off()


d_cd <- read.csv("scalability/circuit_depth.csv", stringsAsFactors=FALSE)
d_cd$jo_variant <- "Left Deep"
d_cd$jo_variant[d_cd$method %in% c("BiDEDE", "QDSM", "QRL", "QML")] <- "Bushy"

g_cd <- ggplot(d_cd, mapping=aes(colour=method, linetype=jo_variant)) +
    geom_line(aes(x = num_relations, y = circuit_depth), linewidth = LINE.SIZE) +
    labs(x="", y="Circuit Depth (log)") +
    scale_colour_manual(values=COLOURS.LIST, labels=c("SIGMOD'23~\\cite{schoenberger23}", "BiDEDE'23~\\cite{nayak23}", "QDSM'23~\\cite{schoenberger23:qdsm}", "VLDB'24~\\cite{schoenberger24}", "Single-Step QML~\\cite{winker23}", "Multi-Step QRL"), breaks=c("SIGMOD", "BiDEDE", "QDSM", "VLDB", "QML", "QRL"), name="Method") +
    scale_linetype_manual(values=c("solid", "dotted"), name="Join Search Space") +
    scale_y_continuous(trans="log10", breaks = c(10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9), labels = trans_format('log10', math_format(10^.x)), limits=c(3, 1000)) +
    theme_paper_no_legend_small() +
    theme(axis.title.x = element_blank(),
          plot.margin = unit(c(0,0,0,0), 'cm'),
          panel.background = element_rect(fill = "transparent")
    )

d_n_qubits <- read.csv("scalability/num_qubits.csv", stringsAsFactors=FALSE)
d_n_qubits$jo_variant <- "Left Deep"
d_n_qubits$jo_variant[d_n_qubits$method %in% c("BiDEDE'23", "QDSM'23", "QRL", "SingleStep")] <- "Bushy"
g <- ggplot(d_n_qubits, mapping=aes(colour=method, linetype=jo_variant)) +
    geom_line(aes(x = num_relations, y = num_qubits), linewidth = LINE.SIZE) +
    labs(x="\\# Relations", y="\\# Qubits (log)") +
    theme_paper_legend_right() +
    scale_colour_manual(values=COLOURS.LIST, labels=c("SIGMOD'23~\\cite{schoenberger23}", "BiDEDE'23~\\cite{nayak23}", "QDSM'23~\\cite{schoenberger23:qdsm}", "VLDB'24~\\cite{schoenberger24}", "Single-Step QML~\\cite{winker23}", "Multi-Step QRL"), breaks=c("SIGMOD'23", "BiDEDE'23", "QDSM'23", "VLDB'24", "SingleStep", "QRL"), name="Method") +
    scale_linetype_manual(values=c("solid", "dotted"), name="Join Search Space") +
    scale_y_continuous(trans="log10", breaks = c(10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9), labels = trans_format('log10', math_format(10^.x))) +
    guides(colour=guide_legend(ncol=1,nrow=6,byrow=TRUE)) +
    theme(plot.margin = unit(c(0,0,0,0), 'cm'),
          legend.key.size = unit(3, 'mm'),
          legend.margin=margin(0, -1, 0, -1, "mm")) +
    inset_element(g_cd, 0.02, 0.46, 0.6, 0.95)

save_name <- "num_qubits"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.55*COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.55*COLWIDTH)
print(g)
dev.off()

d_noisy <- load_noisy_val_data_from_dir(str_c(result_path, "/validation"))
d_noisy <- d_noisy %>% filter(depol_error_prob < 0.1)
d_noisy$depol_error_prob <- d_noisy$depol_error_prob * 100
d_noisy_stat <- do_some_statistics(d_noisy, FALSE, c("mode", "l", "data_reuploading", "depol_error_prob"))

g <- ggplot(d_noisy_stat, mapping=aes(x=depol_error_prob, y=median, colour=factor(l))) +
    theme_paper_legend_right() +
    geom_line(linewidth = LINE.SIZE) +
    geom_hline(aes(yintercept = 1, colour="DP"), linetype="dashed", linewidth = LINE.SIZE) +
    scale_colour_manual(values=COLOURS.LIST, name="\\# Layers", breaks=c("DP", "4", "8", "12", "16", "20")) +
    scale_y_continuous(trans="log10", breaks = c(1, 2, 3, 5)) +
    scale_x_continuous(labels=latex_percent) +
    labs(x="Depolarising Error Probability", y="$C/C_{\\text{DP}}$ (log)") +
    facet_grid(data_reuploading ~ factor(mode, levels = mode_levels),
               labeller=labeller(data_reuploading = as_labeller(dr_labeller))) +
    theme(plot.margin = unit(c(0,0,0,0), 'cm'),
        legend.key.size = unit(3, 'mm'),
        legend.margin=margin(0, -1, 0, -1, "mm"))


save_name <- "noisy_validation_median"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.5*COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.5*COLWIDTH)
print(g)
dev.off()

d_noisy$depol_error_prob <- as.factor(d_noisy$depol_error_prob)

g <- ggplot(d_noisy, mapping=aes(x=depol_error_prob, y=mrc, colour=factor(l))) +
    theme_paper_base() +
    geom_boxplot(outlier.size=0.2) +
    scale_colour_manual(values=COLOURS.LIST, name="\\# Layers", breaks=c("DP", "4", "8", "12", "16", "20")) +
    scale_y_continuous(trans="log10", breaks = c(1, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9), labels = trans_format('log10', math_format(10^.x))) +
    scale_x_discrete(labels=latex_percent) +
    labs(x="Depolarising Error Probability", y="$C/C_{\\text{DP}}$ (log)") +
    facet_grid(data_reuploading ~ factor(mode, levels = mode_levels),
               labeller=labeller(data_reuploading = as_labeller(dr_labeller)))


save_name <- "noisy_validation"
pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = COLWIDTH, height = 0.7*COLWIDTH)
print(g)
dev.off()
tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = COLWIDTH, height = 0.7*COLWIDTH)
print(g)
dev.off()

