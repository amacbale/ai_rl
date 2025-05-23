rm(list=ls())

library(tidyverse)
library(patchwork)

{
  seed <- sample(1:10000, 1)
  set.seed(seed)
  
  # 1) Problem spec
  k      <- 5
  true_p <- runif(k, 0.1, 0.9)
  T      <- 5000
  alpha  <- 0.01
  eps    <- 0.1
  
  # 2) AIF knobs
  w_epi  <- 10.0
  lambda <- 5.0
  gamma  <- 15.0
  
  # 3) Storage & init
  Q          <- rep(0.5, k)
  a_counts   <- rep(1, k)
  b_counts   <- rep(1, k)
  reward_rl  <- numeric(T)
  reward_ai  <- numeric(T)
  rl_Q_hist  <- matrix(0, nrow=T, ncol=k)
  ai_p_hist  <- matrix(0, nrow=T, ncol=k)
  
  # unlock tracking
  secret_seq      <- c(2,4,3)
  L               <- length(secret_seq)
  unlock_arm      <- secret_seq[1]
  last_rl         <- integer(0); unlocked_rl <- FALSE
  last_ai         <- integer(0); unlocked_ai <- FALSE
  unlocked_rl_vec <- logical(T)
  unlocked_ai_vec <- logical(T)
  
  # KL helper
  kl_beta <- function(a1,b1,a0,b0) {
    t1 <- lgamma(a0+b0)-lgamma(a0)-lgamma(b0)
    t2 <- lgamma(a1)+lgamma(b1)-lgamma(a1+b1)
    t3 <- (a1-a0)*(digamma(a1)-digamma(a1+b1))
    t4 <- (b1-b0)*(digamma(b1)-digamma(a1+b1))
    t5 <- (a0+b0-a1-b1)*(digamma(a0+b0)-digamma(a1+b1))
    t1+t2+t3+t4+t5
  }
  
  # 4) Simulation
  for(t in seq_len(T)){
    #–– RL step ––
    rl_Q_hist[t,] <- Q
    if(runif(1)<eps) a_rl <- sample.int(k,1) else a_rl <- which.max(Q)
    r_rl <- if(unlocked_rl && a_rl==unlock_arm) 1 else rbinom(1,1,true_p[a_rl])
    Q[a_rl]       <- Q[a_rl] + alpha * (r_rl - Q[a_rl])
    reward_rl[t]  <- r_rl
    
    # update RL history & unlock flag
    last_rl <- c(last_rl, a_rl)
    if(length(last_rl)>L) last_rl <- tail(last_rl, L)
    if(!unlocked_rl && length(last_rl)==L && all(last_rl==secret_seq)) {
      unlocked_rl <- TRUE
    }
    unlocked_rl_vec[t] <- unlocked_rl
    
    #–– AIF step ––
    seqs      <- expand.grid(a1=1:k, a2=1:k, a3=1:k)
    best_score<- Inf
    best_a    <- NA
    p0        <- a_counts / (a_counts + b_counts)
    
    # helper: expected KL for pulling arm a once
    efe_one <- function(a, p_prior, alpha, beta) {
      # P(o=1) = p_prior,  P(o=0) = 1-p_prior
      kl1 <- kl_beta(alpha[a]+1, beta[a], alpha[a], beta[a])
      kl0 <- kl_beta(alpha[a],   beta[a]+1, alpha[a], beta[a])
      epi <- p_prior[a]*kl1 + (1-p_prior[a])*kl0
      ext <- p_prior[a]        # expected reward
      list(extr=ext, epi=epi)
    }
    
    # simulate each sequence
    for(i in seq_len(nrow(seqs))) {
      plan <- as.integer(seqs[i, ])
      score <- 0
      # local copies
      p_sim     <- p0
      alpha_sim<- a_counts
      beta_sim <- b_counts
      
      # roll out 3 steps
      for(h in 1:3) {
        a_h <- plan[h]
        # exact expected EFE for that pull
        tmp <- efe_one(a_h, p_sim, alpha_sim, beta_sim)
        score <- score - (lambda * tmp$extr)       # extrinsic
        score <- score - (w_epi * tmp$epi)         # epistemic
        
        # now *update* your simulated counts
        # assume the *mean* outcome: alpha+=p, beta+=(1-p)
        alpha_sim[a_h] <- alpha_sim[a_h] + p_sim[a_h]
        beta_sim[a_h]  <- beta_sim[a_h]  + (1 - p_sim[a_h])
        # recompute p_sim afterwards
        p_sim[a_h]     <- alpha_sim[a_h] / (alpha_sim[a_h] + beta_sim[a_h])
      }
      
      if (score < best_score) {
        best_score <- score
        best_a     <- plan[1]
      }
    }
    
    #  now execute best_a
    a_ai <- best_a
    r_ai <- if (unlocked_ai && a_ai==unlock_arm) 1 else rbinom(1,1,true_p[a_ai])
    reward_ai[t] <- r_ai
    
    # 5) Update AIF’s memory & unlock flag (unchanged)
    last_ai <- c(last_ai, a_ai)
    if(length(last_ai)>L) last_ai <- tail(last_ai, L)
    if(!unlocked_ai && length(last_ai)==L && all(last_ai==secret_seq)){
      unlocked_ai <- TRUE
    }
    unlocked_ai_vec[t] <- unlocked_ai
    
    # 6) Real Bayesian update of counts
    a_counts[a_ai] <- a_counts[a_ai] + r_ai
    b_counts[a_ai] <- b_counts[a_ai] + (1 - r_ai)
  }
  
  arm_labels <- paste0("Arm ", seq_len(k), " (", sprintf("%.3f", true_p), ")")
  
  # 5) Plotting
  
  # 5.1) Unlock times
  unlock_time_rl <- which(unlocked_rl_vec)[1]
  unlock_time_ai <- which(unlocked_ai_vec)[1]
  
  # 5.2) Cumulative reward
  df_r <- tibble(
    trial = 1:T,
    RL    = cumsum(reward_rl),
    AIF   = cumsum(reward_ai)
  ) %>% pivot_longer(-trial, names_to="agent", values_to="cum_reward")
  
  vline_df <- tibble(
    agent       = c("RL","AIF"),
    unlock_time = c(unlock_time_rl, unlock_time_ai)
  )
  
  p1 <- ggplot(df_r, aes(trial, cum_reward, color=agent)) +
    geom_line() +
    geom_vline(data=vline_df, aes(xintercept=unlock_time, color=agent),
               linetype="dashed") +
    theme_minimal() +
    labs(title="Cumulative Reward with Unlock Points",
         x="Trial", y="Cum. Reward")
  
  # 5.3) Belief trajectories
  df_q <- as_tibble(rl_Q_hist) %>%
    set_names(arm_labels) %>%
    mutate(trial=1:T) %>%
    pivot_longer(-trial, names_to="arm", values_to="value") %>%
    mutate(agent="RL")
  
  df_p <- as_tibble(ai_p_hist) %>%
    set_names(arm_labels) %>%
    mutate(trial=1:T) %>%
    pivot_longer(-trial, names_to="arm", values_to="value") %>%
    mutate(agent="AIF")
  
  p2 <- bind_rows(df_q, df_p) %>%
    ggplot(aes(trial, value, color=arm)) +
    geom_line(alpha=0.7) +
    facet_wrap(~agent, ncol=1) +
    theme_minimal() +
    labs(title="Estimated Arm Means", x="Trial", y="Estimate",
         color="Arm (true p)")
  
  # 5.4) Unlock flag over time
  df_u <- tibble(
    trial = 1:T,
    RL    = unlocked_rl_vec,
    AIF   = unlocked_ai_vec
  ) %>% pivot_longer(-trial, names_to="agent", values_to="unlocked")
  
  p3 <- ggplot(df_u, aes(trial, unlocked, color=agent)) +
    geom_step() +
    theme_minimal() +
    labs(title="Unlock Flag Over Time", x="Trial", y="Unlocked?")
  
  # 5.5) Combine
  print(p1 / (p2 | p3))
}
