# import
library(sparklyr)
library(tidyverse)

# get directory
directory <- dirname(rstudioapi::getActiveDocumentContext()$path)
filepath <- file.path(directory, 'song')
setwd(filepath)

# create empty dataframe
df <- data.frame(
  song_id=character(),
  playcount=numeric(),
  stringsAsFactors=FALSE
)

# iterate csv part files -> read & append to dataframe
for (f in list.files(pattern='.csv'))
{
  tmp <- read.csv(f, header=FALSE, stringsAsFactors=FALSE) %>%
    rename(song_id=V1) %>%
    rename(playcount=V2)
  df <- rbind(df, tmp)
}

summary(df)

ggplot(data=df) +
  geom_density(mapping=aes(playcount), fill='#56595C', color='#56595C', alpha=0.8) +
  labs(title='THE DISTRIBUTION OF SONG POPULARITY', 
       subtitle='', 
       x='PLAYCOUNT', y='DENSITY') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        axis.title=element_text(size=9, face='bold', color='#56595C'),
        plot.title=element_text(size=12, face='bold', color='#56595C'),
        plot.subtitle=element_text(size=9, face='bold', color='#56595C'),
        legend.text=element_text(size=10, face='bold', color='#56595C'))


# get directory
directory <- dirname(rstudioapi::getActiveDocumentContext()$path)
filepath <- file.path(directory, 'user')
setwd(filepath)

# create empty dataframe
df <- data.frame(
  user_id=character(),
  songcount=numeric(),
  stringsAsFactors=FALSE
)

# iterate csv part files -> read & append to dataframe
for (f in list.files(pattern='.csv'))
{
  tmp <- read.csv(f, header=FALSE, stringsAsFactors=FALSE) %>%
    rename(user_id=V1) %>%
    rename(songcount=V2)
  df <- rbind(df, tmp)
}

summary(df)

ggplot(data=df) +
  geom_density(mapping=aes(songcount), fill='#56595C', color='#56595C', alpha=0.8) +
  labs(title='THE DISTRIBUTION OF USER ACTIVITY', 
       subtitle='', 
       x='SONGCOUNT', y='DENSITY') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        axis.title=element_text(size=9, face='bold', color='#56595C'),
        plot.title=element_text(size=12, face='bold', color='#56595C'),
        plot.subtitle=element_text(size=9, face='bold', color='#56595C'),
        legend.text=element_text(size=10, face='bold', color='#56595C'))
