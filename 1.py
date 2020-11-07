import streamlit as st
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from PIL import Image
import time 
import pickle
default=1
def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

rad=st.sidebar.radio("Navigation:",['Home','Model'])

if(rad=='Home'):
    st.title('Overall analysis')
    image = Image.open('id.jpg')
    st.image(image, caption='',use_column_width=True)
    df=pd.read_csv('data1.csv')
    st.write('Line Graphs:')
    st.area_chart(df)
    st.write('Data of the students:')
    st.write(df)
    st.write('Maximum analysis:')
    st.dataframe(df.style.highlight_max(axis=0))
    st.write('bargraph:')
    st.bar_chart(df)
    
    col=st.sidebar.multiselect("select Game on which you want to see analysis:",df.columns)
    fig, ax = plt.subplots()
   
    plt.plot(df['Improve'],df[col])
    
    st.pyplot(fig)
    
    cnt=df['Improve'].value_counts()
    cnt=str(cnt)
    
    st.success("Improved intellectual-disability students:"+cnt[5:7])
    st.error("Not Improved intellectual-disability students:"+cnt[10:-27])
    column1=df['G1']
    column2=df['G2']
    column3=df['G3']
    column4=df['G4']
    c1=column1.max()
    c2=column2.max()
    c3=column3.max()
    c4=column4.max()
    st.success("Maximum value for the G1 game:"+str(c1))
    st.success("Maximum value for the G2 game:"+str(c2))
    st.success("Maximum value for the G3 game:"+str(c3))
    st.success("Maximum value for the G4 game:"+str(c4))
if (rad=='Model'):
    progress=st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)
    st.balloons()    
    st.header("Model for prediction")
    df=pd.read_csv('data1.csv')
    train,test=data_split(df,0.2)

    X_train=train[['G1','G2','G3','G4']].to_numpy()
    X_test=train[['G1','G2','G3','G4']].to_numpy()
    Y_train=train[{'Improve'}].to_numpy().reshape(40)
    Y_test=test[{'Improve'}].to_numpy().reshape(9)

    clf=LogisticRegression()
    clf.fit(X_train,Y_train);

    G1=st.number_input("Enter the 1stGame")
    G1=int(G1)
    G2=st.number_input("Enter the 2stGame")
    G2=int(G2)
    G3=st.number_input("Enter the 3rd Game")
    G3=int(G3)
    G4=st.number_input("Enter the 4th Game")
    G4=int(G4)
    
    
    Improve=clf.predict([[G1,G2,G3,G4]])
    if(Improve==0):
        st.success("Student is Improved")
       
        
    else:
        st.success("student is Not Improved")
        
