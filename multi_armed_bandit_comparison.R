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
  
  set.seed(7382)
  
  # 1) Problem spec
  k      <- 5
  true_p <- runif(k, 0.1, 0.9)   # ground truth (unknown to agents)
  T      <- 5000
  alpha  <- 0.01                  # RL learning rate
  eps    <- 0.1                 # RL epsilon-greedy
  
  # 2) AIF parameters
  w_epi  <- 5.0                  # weight on info-gain
  lambda <- 25.0                  # weight on reward
  gamma  <- 15.0                  # softmax inverse temperature
  
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


# 1) Fixed bandit setup
k      <- 5
T      <- 4000
true_p <- runif(k, .1, .9)
p_star <- max(true_p)  
seed   <- 123

# shared simulation code (returns a list with rl and aif cum rewards)
run_bandit <- function(alpha, eps, w_epi, lambda, gamma, seed=123) {
  set.seed(seed)
  Q        <- rep(0.5, k)
  a_counts <- rep(1, k); b_counts <- rep(1, k)
  cum_rl   <- 0; cum_ai <- 0
  
  kl_beta <- function(a1,b1,a0,b0) {
    t1 <- lgamma(a0+b0)-lgamma(a0)-lgamma(b0)
    t2 <- lgamma(a1)+lgamma(b1)-lgamma(a1+b1)
    t3 <- (a1-a0)*(digamma(a1)-digamma(a1+b1))
    t4 <- (b1-b0)*(digamma(b1)-digamma(a1+b1))
    t5 <- (a0+b0-a1-b1)*(digamma(a0+b0)-digamma(a1+b1))
    t1+t2+t3+t4+t5
  }
  
  for(t in 1:T){
    # RL step
    if(runif(1)<eps) a_rl <- sample.int(k,1) else a_rl <- which.max(Q)
    r_rl <- rbinom(1,1,true_p[a_rl])
    Q[a_rl] <- Q[a_rl] + alpha*(r_rl-Q[a_rl])
    cum_rl <- cum_rl + r_rl
    
    # AIF step
    p_hat <- a_counts/(a_counts+b_counts)
    G     <- numeric(k)
    for(a in 1:k){
      ext  <- -lambda * p_hat[a]
      kl1  <- kl_beta(a_counts[a]+1, b_counts[a], a_counts[a], b_counts[a])
      kl0  <- kl_beta(a_counts[a],   b_counts[a]+1, a_counts[a], b_counts[a])
      epi  <- -w_epi * (p_hat[a]*kl1 + (1-p_hat[a])*kl0)
      G[a] <- ext + epi
    }
    pi    <- exp(-gamma*G); pi <- pi/sum(pi)
    a_ai  <- sample.int(k,1,prob=pi)
    r_ai  <- rbinom(1,1,true_p[a_ai])
    cum_ai <- cum_ai + r_ai
    
    a_counts[a_ai] <- a_counts[a_ai] + r_ai
    b_counts[a_ai] <- b_counts[a_ai] + (1-r_ai)
  }
  
  regret_rl <- p_star * T - cum_rl
  regret_ai <- p_star * T - cum_ai
  
  list(cum_rl=cum_rl, cum_ai=cum_ai,
       regret_rl=regret_rl, regret_ai=regret_ai)
}

# 2) RL grid: sweep alpha × eps (AIF held at defaults)
grid_rl <- expand.grid(
  alpha = c(0.01, 0.05, 0.1, 0.2, 0.22, 0.25),
  eps   = c(0, 0.05, 0.1, 0.2)
)
results_rl <- grid_rl %>%
  pmap_dfr(function(alpha, eps) {
    out <- run_bandit(alpha, eps,
                      w_epi=5, lambda=5, gamma=5,
                      seed=seed)
    tibble(alpha, eps,
           cum_rl   = out$cum_rl,
           regret_rl= out$regret_rl)
  })

# plot RL heatmap
ggplot(results_rl, aes(alpha, eps, fill=regret_rl)) +
  geom_tile() +
  scale_fill_viridis_c(option="magma", direction=-1) +
  labs(title="RL Regret vs (α,ε)",
       x=expression(alpha), y=expression(epsilon),
       fill="Regret") +
  theme_minimal()

# 3) AIF grid: sweep w_epi × lambda, faceted by gamma (RL held fixed)
grid_aif <- expand.grid(
  w_epi  = c(1, 2, 5, 10, 15, 20, 25, 30),
  lambda = c(1, 2, 5, 10, 15, 20, 25, 30),
  gamma  = c(1, 5, 10, 15)
)
results_aif <- grid_aif %>%
  pmap_dfr(function(w_epi, lambda, gamma) {
    out <- run_bandit(alpha=0.1, eps=0.1,
                      w_epi=w_epi, lambda=lambda, gamma=gamma,
                      seed=seed)
    tibble(w_epi, lambda, gamma,
           cum_ai    = out$cum_ai,
           regret_ai = out$regret_ai)
  })

# plot AIF heatmaps faceted by gamma
ggplot(results_aif, aes(w_epi, lambda, fill=regret_ai)) +
  geom_tile() +
  scale_fill_viridis_c(option="magma", direction=-1) +
  facet_wrap(~gamma, ncol=3) +
  labs(title="AIF Regret vs (w_epi,λ) for different γ",
       x=expression(w[epi]), y=expression(lambda),
       fill="Regret") +
  theme_minimal()

results_aif[which.max(results_aif$cum_ai),]
results_rl[which.max(results_rl$cum_rl),]


results_aif[which.min(results_aif$regret_ai),]
results_rl[which.min(results_rl$regret_rl),]
