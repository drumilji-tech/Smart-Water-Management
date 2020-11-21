import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
def main():
    st.title("Smart Water Management Using Data Science and Internet of Things(IOT)")
    st.sidebar.title("Machine Learning and its specifications")
    
    
    st.markdown("The problem statement we are trying to address here is a regression problem. We have a dataset that has the water consumption history of a area , and its corresponding charges using this we need to predict what will be the charges corresponding to the consumption.")
    
    
    st.markdown("So, Let's evaluate our model with different Evaluation metrices as the metrices provide us how effective our model is.")
    st.sidebar.markdown("Let\'s do it")
    data = pd.read_csv('https://raw.githubusercontent.com/drumilji-tech/Smart-Water-Management/main/Water_Consumption_And_Cost__2013_-_2020_.csv')
    
    @st.cache(persist=True)
    def split(data):
         
         x= data[['Consumption']]
         y= data[['Total Charges']]
         
         
         x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
         return x_train,x_test,y_train,y_test
     
   
             
     
    x_train,x_test,y_train,y_test = split(data)

    
    st.sidebar.subheader('Choose Model')
    Model = st.sidebar.selectbox("Model",('Linear Regression','Decision Tree','Random Forest','Support Vector Machine'))
    
    if Model == "Linear Regression":
        st.sidebar.subheader("Model Hyperparameters")
        max_iter_log = st.sidebar.slider("Maximum Number of Iterations",100,500,key='max_iter')
        metrics_log = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'))
        
        if st.sidebar.button("Regress",key='class'):
            st.subheader("Linear Regression Results")
            Model = LinearRegression()
            Model.fit(x_train,y_train)
            y_pred = Model.predict(x_test)
            cv=ShuffleSplit(n_splits=7,test_size=0.35,random_state=100)
            st.write("Accuracy Score:",cross_val_score(model,x,y,cv=cv).mean().round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))

        
           
            
    if Model == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000,step=10,key='n_est')
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",('True','False'),key='bootstrap')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'),key='1')
        
        if st.sidebar.button("Regress",key='class'):
            st.subheader("Random Forest Result")
            Model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            Model.fit(x_train,y_train)
        
            y_pred = Model.predict(x_test)
            cv=ShuffleSplit(n_splits=7,test_size=0.35,random_state=100)
            st.write("Accuracy Score:",cross_val_score(model,x,y,cv=cv).mean().round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))
            
    
    if Model == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameters")
        criterion= st.sidebar.radio('Criterion(measures the quality of split)', ('Gini', 'Entropy'), key='criterion')
        splitter = st.sidebar.radio('Splitter (How to split at each node?)', ('Best','Random'), key='splitter')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'),key='1')
        
        if st.sidebar.button("Regress",key='class'):
            st.subheader('Decision Tree Results')
            model = DecisionTreeRegressor(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            cv=ShuffleSplit(n_splits=7,test_size=0.35,random_state=100)
            st.write("Accuracy Score:",cross_val_score(model,x,y,cv=cv).mean().round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))
       
    if Model == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        kernel= st.sidebar.radio('Type of Kernel to be selected', ('Linear', 'RBF','Ploynomial'), key='kernel')
        C_value = st.sidebar.slider("Select C Value",1,20,key='C_value')
        metrics_svm = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'))
        
        if st.sidebar.button("Regress",key='class'):
            st.subheader('Decision Tree Results')
            model = SVR(kernel=kernel,C=C_value)
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_test)
            cv=ShuffleSplit(n_splits=7,test_size=0.35,random_state=100)
            st.write("Accuracy Score:",cross_val_score(model,x,y,cv=cv).mean().round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))               
        
    
     
    
    if st.sidebar.checkbox("Show Raw Data",False):
        st.subheader("Consumption Data")
        st.write(data)




if __name__ == '__main__':
    main()
