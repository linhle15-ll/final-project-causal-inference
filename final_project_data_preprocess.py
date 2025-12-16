#!/usr/bin/env python
# coding: utf-8

# (final_template)=
# 
# # Final Project: Data Preprocessing
# 
# :::{epigraph}
# 
# -- Student name: Ngoc Linh Le
# :::

# The purpose of data process is to prepare a clean and focused data frame for the final project: Estimating the Causal Effect of High Short Video Usage on Academic Performance Among Elementary Students. 

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import ipywidgets as widgets


# In[46]:


# import dataset
vid_df = pd.read_excel('~/comsc341cd.github.io/projects/final_project/final-data.xlsx')
vid_df


# In particular, there are the following variables in the dataset:

# In[47]:


# Column names 
vid_df.columns


# Based on the study design, I selected the variables most relevant to estimating the causal effect of high short-video usage on academic performance. The following code (1) renames the raw survey columns into clear, analysis-ready variable names, and (2) filters the dataset to include only the key covariates and outcome measures used in the causal analysis: student demographics (gender, school type, grade), parent's education level, daily short-video usage duration, and total exam score.

# In[49]:


# rename columns
vid_df = vid_df.rename(columns={
    '2.gender':'gender', # student's gender
    '3.school':'school', # student's school type 
    '4.grade':'grade', # student's grade
    '7、What is your academic qualification?': 'parent_edu', # parent's education level
    '15、How long do you spend on average to use short videos every day?':'ave_short_video_usage/day', # daily short-video usage duration
    'Academic：Total':'exam_score' # total exam score 
})

# filter the important columns to focus on
vid_df = vid_df.loc[:, ['gender', 'school', 'grade', 'parent_edu', 'ave_short_video_usage/day', 'exam_score']]


# In[50]:


# check NaN (nice!)
vid_df.isna().sum()


# To facilitate data analysis, I will transform the binary variable into 0/1 instead of 1/2 as currently.

# In[51]:


vid_df['gender'] = (vid_df['gender'] == 1).astype(int) # male = 1, others (female) = 0
vid_df['school'] = (vid_df['school'] == 1).astype(int) # urban = 1, others (suburban) = 0
vid_df.head(10)


# In[52]:


# Save processed data to a file and import it
# index can change when resort/ filter row/ reset index -> set index=False to prevent inconsistencies
vid_df.to_csv("preprocessed_data.csv", index=False) 
vid_df

