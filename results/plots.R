
library(ggplot2)

data <- read.csv("res_tab.csv")

t1 <- data[data$epsilon == 0, ]
t1 <- t1[t1$method == "MPC+DP" & t1$partitions == 1, ]
t1 <- t1[t1$data_size < 100000, ]
t1["dif_AUC"] <- abs(t1$scikit_auc - t1$AUC)


ggplot(t1, aes(fill=as.factor(data_size), x=as.factor(n_steps), y=dif_AUC)) +
  geom_bar(position = 'dodge', stat = 'identity') +
  coord_cartesian(ylim=c(0, 0.2)) +
  theme_minimal() + 
  labs(x='# of thresholds', y='absolut diff AUC', title='all sizes with different nr of thresholds', fill='data size')

ggplot(t1, aes(x=as.factor(n_steps), y=dif_AUC, group=as.factor(data_size), color=as.factor(data_size))) +
  geom_line() +
  coord_cartesian(ylim=c(0, 0.2)) +
  theme_minimal() + 
  labs(x='# of thresholds', y='absolut diff AUC', title='all sizes with different nr of thresholds', fill='data size')

t1_2 <- t1[t1$data_size < 10000, ]

ggplot(t1_2, aes(x=as.factor(n_steps), y=dif_AUC, group=as.factor(data_size), color=as.factor(data_size))) +
  geom_line() +
  coord_cartesian(ylim=c(0, 0.1)) +
  theme_minimal() + 
  labs(x='# of thresholds', y='absolut diff AUC', title='all sizes with different nr of thresholds', labs='data size')
  
# ----------------------------------

t2 <- data[data$data_size == 100000 & data$method == "MPC+DP", ]
t2["dif_AUC"] <- abs(t2$scikit_auc - t2$AUC)

ggplot(t2, aes(fill=as.factor(partitions), x=as.factor(epsilon), y=dif_AUC)) +
  geom_bar(position = 'dodge', stat = 'identity') +
  coord_cartesian(ylim=c(0, 0.2)) +
  theme_minimal() + 
  labs(x='epsilon', y='absolut diff AUC', title='100k with different partitionings', fill='# of partitions')

# ---------------------------------


