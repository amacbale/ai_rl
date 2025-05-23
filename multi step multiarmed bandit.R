rm(list=ls())

library(tidyverse)
library(patchwork)

{
  seed <- sample(1:10000, 1)
  #set.seed(seed)
  set.seed(509)
  # 1) Problem spec
  k      <- 5
  true_p <- runif(k, 0.1, 0.9)
  T      <- 5000
  alpha  <- 0.01
  eps    <- 0.3
  
  # 2) AIF knobs
  w_epi  <- 5.0
  lambda <- 5.0
  gamma  <- 5.0
  
  # 3) Storage & init
  Q          <- rep(0.5, k)
  a_counts   <- rep(1, k)
  b_counts   <- rep(1, k)
  reward_rl  <- numeric(T)
  reward_ai  <- numeric(T)
  rl_Q_hist  <- matrix(0, T, k)
  ai_p_hist  <- matrix(0, T, k)
  
  # unlock tracking
  secret_seq      <- c(2,4,3)
  L               <- length(secret_seq)
  unlock_arm      <- secret_seq[1]
  last_rl         <- integer(0); unlocked_rl <- FALSE
  last_ai         <- integer(0); unlocked_ai <- FALSE
  unlocked_rl_vec <- logical(T)
  unlocked_ai_vec <- logical(T)
  
  # plan buffer for AIF
  plan_buf <- integer(0)
  
  # KL helper
  kl_beta <- function(a1,b1,a0,b0) {
    t1 <- lgamma(a0+b0)-lgamma(a0)-lgamma(b0)
    t2 <- lgamma(a1)+lgamma(b1)-lgamma(a1+b1)
    t3 <- (a1-a0)*(digamma(a1)-digamma(a1+b1))
    t4 <- (b1-b0)*(digamma(b1)-digamma(a1+b1))
    t5 <- (a0+b0-a1-b1)*(digamma(a0+b0)-digamma(a1+b1))
    t1+t2+t3+t4+t5
  }
  
  # precompute all 3-step sequences only once
  seqs <- expand.grid(a1=1:k, a2=1:k, a3=1:k)
  
  # 4) Simulation
  for(t in seq_len(T)){
    #–– RL step ––
    rl_Q_hist[t,] <- Q
    if(runif(1)<eps) a_rl <- sample.int(k,1) else a_rl <- which.max(Q)
    r_rl <- if(unlocked_rl && a_rl==unlock_arm) 1 else rbinom(1,1,true_p[a_rl])
    Q[a_rl]      <- Q[a_rl] + alpha * (r_rl - Q[a_rl])
    reward_rl[t] <- r_rl
    
    # update RL history & unlock flag
    last_rl <- c(last_rl, a_rl)
    if(length(last_rl)>L) last_rl <- tail(last_rl, L)
    if(!unlocked_rl && length(last_rl)==L && all(last_rl==secret_seq)){
      unlocked_rl <- TRUE
    }
    unlocked_rl_vec[t] <- unlocked_rl
    
    #–– AIF step with plan buffer ––
    if(length(plan_buf)==0){
      # need to re-plan a fresh 3-step sequence
      best_score <- Inf
      best_seq   <- NULL
      p0 <- a_counts/(a_counts + b_counts)
      for(i in seq_len(nrow(seqs))){
        plan <- as.integer(seqs[i,])
        score <- 0
        p_sim <- p0
        alpha_sim <- a_counts
        beta_sim  <- b_counts
        
        for(h in 1:3){
          a_h <- plan[h]
          # extrinsic
          score <- score - lambda * p_sim[a_h]
          # epistemic
          kl1 <- kl_beta(alpha_sim[a_h]+1, beta_sim[a_h], alpha_sim[a_h], beta_sim[a_h])
          kl0 <- kl_beta(alpha_sim[a_h],   beta_sim[a_h]+1, alpha_sim[a_h], beta_sim[a_h])
          score <- score - w_epi * (p_sim[a_h]*kl1 + (1-p_sim[a_h])*kl0)
          # simulate Bayesian update (fractional)
          alpha_sim[a_h] <- alpha_sim[a_h] + p_sim[a_h]
          beta_sim[a_h]  <- beta_sim[a_h]  + (1-p_sim[a_h])
          p_sim[a_h]     <- alpha_sim[a_h]/(alpha_sim[a_h]+beta_sim[a_h])
        }
        if(score < best_score){
          best_score <- score
          best_seq   <- plan
        }
      }
      plan_buf <- best_seq
    }
    
    # execute next action from the buffer
    a_ai <- plan_buf[1]
    plan_buf <- plan_buf[-1]
    
    # sample AIF reward
    r_ai <- if(unlocked_ai && a_ai==unlock_arm) 1 else rbinom(1,1,true_p[a_ai])
    reward_ai[t] <- r_ai
    
    # update AIF history & unlock flag
    last_ai <- c(last_ai, a_ai)
    if(length(last_ai)>L) last_ai <- tail(last_ai, L)
    if(!unlocked_ai && length(last_ai)==L && all(last_ai==secret_seq)){
      unlocked_ai <- TRUE
    }
    unlocked_ai_vec[t] <- unlocked_ai
    
    # real Bayesian update of counts
    a_counts[a_ai] <- a_counts[a_ai] + r_ai
    b_counts[a_ai] <- b_counts[a_ai] + (1 - r_ai)
    # record posterior means
    ai_p_hist[t,]  <- a_counts/(a_counts + b_counts)
  }
  
  arm_labels <- paste0("Arm ",1:k," (",sprintf("%.3f",true_p),")")
  
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
