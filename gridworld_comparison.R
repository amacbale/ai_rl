# ─────────────────────────────────────────────────────────────────────
# Multi‐Episode Maze: RL vs Naïve AIF Planner
# ─────────────────────────────────────────────────────────────────────
library(tidyverse)

set.seed(42)

# 1) FIXED MAP (5×5)
N     <- 5
cells <- expand.grid(x=1:N, y=1:N)
goal  <- data.frame(x=5,y=5)
clue  <- data.frame(x=1,y=5)
itemA <- data.frame(x=3,y=2)
itemB <- data.frame(x=4,y=4)
start <- data.frame(x=1,y=1)

dirs <- tibble(
  action=c("up","down","left","right"),
  dx    =c( 0,  0, -1,  1),
  dy    =c( 1, -1,  0,  0)
)

clamp <- function(v) pmin(pmax(v,1),N)


# 2) EPISODES & HORIZON
nEps    <- 200
horizon <- 10

# RL hyperparams
alpha   <- 0.1
eps     <- 0.1
gammaRL <- 0.9

# AIF rewards & costs
step_cost      <- -0.1
reward_item    <- +1
reward_exit_ok <- +5
reward_exit_bad<- -5

# 3) STORAGE FOR RETURNS
returns <- tibble(
  episode = 1:nEps,
  RL      = numeric(nEps),
  AIF     = numeric(nEps)
)

# 4) INIT Q-TABLE FOR RL:
# state dims: x,y, haveA(2), haveB(2), clueFound(2), action(4)
Qtab <- array(0, dim=c(N,N,2,2,2,4))


# 5) EPISODE LOOP
for(ep in seq_len(nEps)){
  # reset RL agent’s state
  stateRL <- list(x= start$x, y= start$y,
                  haveA=FALSE, haveB=FALSE, clue=FALSE)
  # reset AIF agent’s state & beliefs
  stateAIF <- list(x= start$x, y= start$y,
                   haveA=FALSE, haveB=FALSE, clue=FALSE)
  # uniform marginals over clue and items
  bClue <- matrix(1/(N^2), N, N)
  bA    <- matrix(1/(N^2), N, N)
  bB    <- matrix(1/(N^2), N, N)
  
  retRL  <- 0
  retAIF <- 0
  
  # one‐step AIF planner needs an entropy helper
  entropy <- function(m){
    v <- as.numeric(m); v <- v[v>0]
    -sum(v * log(v))
  }
  
  for(t in seq_len(horizon)){
    # ----- RL STEP -----
    sid <- c(stateRL$x, stateRL$y,
             stateRL$haveA+1, stateRL$haveB+1, stateRL$clue+1)
    if(runif(1)<eps) {
      aRL <- sample.int(4,1)
    } else {
      aRL <- which.max(Qtab[sid[1],sid[2],sid[3],sid[4],sid[5],])
    }
    mv <- dirs[aRL,]
    nx <- clamp(stateRL$x + mv$dx)
    ny <- clamp(stateRL$y + mv$dy)
    stateRL$x <- nx; stateRL$y <- ny
    
    # observe + reward RL
    r   <- step_cost
    if(!stateRL$clue && nx==clue$x && ny==clue$y){
      stateRL$clue <- TRUE
    }
    if(!stateRL$haveA && nx==itemA$x && ny==itemA$y){
      stateRL$haveA <- TRUE; r <- r + reward_item
    }
    if(!stateRL$haveB && nx==itemB$x && ny==itemB$y){
      stateRL$haveB <- TRUE; r <- r + reward_item
    }
    if(nx==goal$x && ny==goal$y){
      r <- r + ifelse(stateRL$haveA && stateRL$haveB && stateRL$clue,
                      reward_exit_ok, reward_exit_bad)
    }
    retRL <- retRL + r
    
    # Q‐learning update
    sid2 <- c(nx, ny,
              stateRL$haveA+1, stateRL$haveB+1, stateRL$clue+1)
    oldQ <- Qtab[sid[1],sid[2],sid[3],sid[4],sid[5],aRL]
    Qtab[sid[1],sid[2],sid[3],sid[4],sid[5],aRL] <-
      oldQ + alpha*(r + gammaRL * max(Qtab[sid2[1],sid2[2],
                                           sid2[3],sid2[4],sid2[5],]) - oldQ)
    
    # ----- AIF STEP -----
    # 1) One‐step EFE for each direction
    H0   <- entropy(bClue) + entropy(bA) + entropy(bB)
    Gs   <- numeric(4)
    prefs <- c(
      empty = step_cost,
      clue  = 0,
      A     = reward_item,
      B     = reward_item,
      exit_ok  = reward_exit_ok,
      exit_bad = reward_exit_bad
    )
    for(a in 1:4){
      mv2 <- dirs[a,]
      cx  <- clamp(stateAIF$x + mv2$dx)
      cy  <- clamp(stateAIF$y + mv2$dy)
      
      # predictive P(o|a): over {empty,clue,A,B,exit_ok,exit_bad}
      p_clue <- if(!stateAIF$clue) bClue[cx,cy] else 0
      p_A    <- if(!stateAIF$haveA) bA[cx,cy]      else 0
      p_B    <- if(!stateAIF$haveB) bB[cx,cy]      else 0
      p_exit <- if(cx==goal$x && cy==goal$y){
        if(stateAIF$clue && stateAIF$haveA && stateAIF$haveB) 1 else 1
      } else 0
      # if exit on bad conditions => bad exit
      p_exit_ok  <- if(p_exit==1 && stateAIF$clue && stateAIF$haveA && stateAIF$haveB) 1 else 0
      p_exit_bad <- if(p_exit==1 && !(stateAIF$clue && stateAIF$haveA && stateAIF$haveB)) 1 else 0
      # empty covers the rest
      p_empty <- 1 - (p_clue + p_A + p_B + p_exit_ok + p_exit_bad)
      
      Po <- c(
        empty     = p_empty,
        clue      = p_clue,
        A         = p_A,
        B         = p_B,
        exit_ok   = p_exit_ok,
        exit_bad  = p_exit_bad
      )
      
      # extrinsic = –E[reward]
      ext <- - sum(Po * prefs)
      
      # epistemic = expected entropy reduction
      # we simulate posterior beliefs for each possible obs:
      H_post <- 0
      # for each o, compute posterior belief & its entropy
      for(o in names(Po)){
        if(Po[o]==0) next
        # copy beliefs
        bCl2 <- bClue; bA2 <- bA; bB2 <- bB
        
        if(o=="empty"){
          bCl2[cx,cy] <- 0; bCl2 <- bCl2/sum(bCl2)
          bA2[cx,cy]  <- 0; bA2  <- bA2/sum(bA2)
          bB2[cx,cy]  <- 0; bB2  <- bB2/sum(bB2)
        }
        if(o=="clue"){
          stateAIF$clue <- TRUE  # temporary flag
          # collapse clue (no longer needed)
          bCl2[,] <- 0; 
          # assume clue reveals items exactly: we’ll treat that
          bA2[,]  <- 0; bA2[itemA$x,itemA$y] <- 1
          bB2[,]  <- 0; bB2[itemB$x,itemB$y] <- 1
        }
        if(o=="A"){
          bA2[,] <- 0; bA2[cx,cy] <- 1
        }
        if(o=="B"){
          bB2[,] <- 0; bB2[cx,cy] <- 1
        }
        # exit_obs gives no info about items/clue
        H_post <- H_post + Po[o] * (entropy(bCl2) + entropy(bA2) + entropy(bB2))
      }
      epi <- H0 - H_post
      
      Gs[a] <- ext - epi
    }
    
    # 2) Softmax policy
    pi   <- exp(-gamma*Gs); pi <- pi/sum(pi)
    aAIF <- sample.int(4,1,prob=pi)
    
    mv  <- dirs[aAIF,]
    cx  <- clamp(stateAIF$x + mv$dx)
    cy  <- clamp(stateAIF$y + mv$dy)
    stateAIF$x <- cx; stateAIF$y <- cy
    
    # observe + reward AIF
    ra <- step_cost
    if(!stateAIF$clue && cx==clue$x && cy==clue$y){
      stateAIF$clue <- TRUE
      bClue[,] <- 0
    }
    if(!stateAIF$haveA && cx==itemA$x && cy==itemA$y){
      stateAIF$haveA <- TRUE; ra <- ra + reward_item
      bA[,] <- 0; bA[itemA$x,itemA$y] <- 1
    }
    if(!stateAIF$haveB && cx==itemB$x && cy==itemB$y){
      stateAIF$haveB <- TRUE; ra <- ra + reward_item
      bB[,] <- 0; bB[itemB$x,itemB$y] <- 1
    }
    if(cx==goal$x && cy==goal$y){
      ra <- ra + ifelse(stateAIF$clue && stateAIF$haveA && stateAIF$haveB,
                        reward_exit_ok, reward_exit_bad)
    }
    retAIF <- retAIF + ra
  }
  
  returns$RL[ep]  <- retRL
  returns$AIF[ep] <- retAIF
}

# 6) Plot learning curves
df <- returns %>% pivot_longer(-episode, names_to="agent", values_to="return")

ggplot(df, aes(episode, return, color=agent)) +
  stat_summary(fun=mean, geom="line", size=1) +
  theme_minimal() +
  labs(title="Mean Episode Return: RL vs Naive AIF",
       x="Episode", y="Mean Return") +
  geom_hline(yintercept = reward_item*2 + reward_exit_ok + step_cost*horizon,
             linetype="dashed", color="black") +
  annotate("text", x=nEps, 
           y=reward_item*2 + reward_exit_ok + step_cost*horizon,
           label="Optimal", hjust=1.1, vjust=-0.5)
