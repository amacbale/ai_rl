rm(list=ls())
library(tidyverse)
library(patchwork)

# -------------------------------------------------------------------
# k-Armed Bernoulli Bandit: RL vs AIF with Beta-Bayes planning
# -------------------------------------------------------------------
{
  seed <- sample(1:10000, 1)
  #set.seed(4696)
  #set.seed(667)
  #set.seed(1424)
  #set.seed(9726)
  #set.seed(4728)
  
  set.seed(seed)
  
  # 1) Problem spec
  k      <- 5
  true_p <- runif(k, 0.1, 0.9)   # ground truth (unknown to agents)
  T      <- 4000
  alpha  <- 0.1                  # RL learning rate
  eps    <- 0.1                  # RL epsilon-greedy
  
  # 2) AIF parameters
  w_epi  <- 5.0                  # weight on info-gain
  lambda <- 5.0                  # weight on reward
  gamma  <- 5.0                  # softmax inverse temperature
  
  # 3) Storage & init
  # RL
  Q        <- rep(0.5, k)
  # AIF: Beta(alpha,beta) counts for each arm
  a_counts <- rep(1, k)
  b_counts <- rep(1, k)
  
  reward_rl <- numeric(T)
  reward_ai <- numeric(T)
  rl_Q_hist <- matrix(0, nrow=T, ncol=k)
  ai_p_hist <- matrix(0, nrow=T, ncol=k)
  
  # helper: KL divergence Beta(a1,b1) || Beta(a0,b0)
  kl_beta <- function(a1,b1,a0,b0) {
    term1 <- lgamma(a0+b0) - lgamma(a0) - lgamma(b0)
    term2 <- lgamma(a1) + lgamma(b1) - lgamma(a1+b1)
    term3 <- (a1-a0)*(digamma(a1) - digamma(a1+b1))
    term4 <- (b1-b0)*(digamma(b1) - digamma(a1+b1))
    term5 <- (a0+b0 - a1 - b1)*(digamma(a0+b0) - digamma(a1+b1))
    term1 + term2 + term3 + term4 + term5
  }
  
  # 4) Simulation
  for(t in 1:T){
    # RL: epsilon-greedy
    rl_Q_hist[t,] <- Q
    if(runif(1)<eps) a_rl <- sample.int(k,1)
    else             a_rl <- which.max(Q)
    r_rl <- rbinom(1,1,true_p[a_rl])
    Q[a_rl] <- Q[a_rl] + alpha*(r_rl-Q[a_rl])
    reward_rl[t] <- r_rl
    
    # AIF
    p_hat <- a_counts/(a_counts+b_counts)  
    ai_p_hist[t,] <- p_hat
    G <- numeric(k)
    for(a in 1:k){
      ext <- -lambda * p_hat[a]
      kl1 <- kl_beta(a_counts[a]+1, b_counts[a], a_counts[a], b_counts[a])
      kl0 <- kl_beta(a_counts[a],   b_counts[a]+1, a_counts[a], b_counts[a])
      epi <- -w_epi * ( p_hat[a]*kl1 + (1-p_hat[a])*kl0 )
      G[a] <- ext + epi
    }
    pi   <- exp(-gamma*G); pi <- pi/sum(pi)
    a_ai <- sample.int(k,1,prob=pi)
    r_ai <- rbinom(1,1,true_p[a_ai])
    reward_ai[t] <- r_ai
    
    # update counts
    a_counts[a_ai] <- a_counts[a_ai] + r_ai
    b_counts[a_ai] <- b_counts[a_ai] + (1-r_ai)
  }
  arm_labels <- paste0("Arm ", seq_len(k), " (", sprintf("%.3f", true_p), ")")
  
  # 5) Plots
  library(tidyverse)
  library(patchwork)
  
  # 5.1) Cumulative rewards
  df_r <- tibble(
    trial = 1:T,
    RL    = cumsum(reward_rl),
    AIF   = cumsum(reward_ai)
  ) |> pivot_longer(-trial, names_to="agent", values_to="cum_reward")
  
  p1 <- ggplot(df_r, aes(trial, cum_reward, color=agent)) +
    geom_line() +
    theme_minimal() +
    labs(title="Cumulative Reward", x="Trial", y="Cum. Reward")
  
  # 5.2) Learned‐mean trajectories
  # RL Q‐values
  df_q <- as_tibble(rl_Q_hist) |>
    set_names(arm_labels) |>        
    mutate(trial=1:T) |>
    pivot_longer(-trial, names_to="arm", values_to="value") |>
    mutate(agent="RL")
  
  # AIF posterior means
  df_p <- as_tibble(ai_p_hist) |>
    set_names(arm_labels) |>       
    mutate(trial=1:T) |>
    pivot_longer(-trial, names_to="arm", values_to="value") |>
    mutate(agent="AIF")
  
  df_both <- bind_rows(df_q, df_p)
  
  p2 <- ggplot(df_both, aes(trial, value, color=arm)) +
    geom_line(alpha=0.7) +
    facet_wrap(~agent, ncol=1) +
    theme_minimal() +
    labs(title="Estimated Arm Means",
         x="Trial", y="Estimate",
         color="Arm (true p)")  +     
    theme(legend.position="bottom")
  
  print(p1 / p2)
}


