import streamlit as st
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

st.title('Penguin Classifier: A Machine Learning App')
st.write("This app uses 6 inputs to predict the species of penguin using"
         "a model built on the Palmer's Penguins's dataset."
         "Use the form below to  get started")

penguin_df = pd.read_csv('penguins.csv')
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle= open('output_penguin.pickle','rb')
rfc= pickle.load(rf_pickle)
unique_penguin_mapping=pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

with st.form('user input'):
    island = st.selectbox('Penguin Island', options=[
        'Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female' ,'Male'])
    bill_length=st.number_input('Bill Length (mm)', min_value=0)
    bill_depth=st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length=st.number_input('Flipper Length (mm)', min_value=0)
    body_mass=st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()

island_biscoe,island_dream,island_torgerson = 0,0,0

if island=='Biscoe':
    island_biscoe=1
elif island=='Dream':
    island_dream=1
elif island=='Torgerson':
    island_torgerson=1

sex_female, sex_male=0,0

if sex=='Female':
    sex_female=1
elif sex=='Male':
    sex_male=1

new_prediction = rfc.predict([[bill_length,bill_depth,flipper_length,body_mass,
                               island_biscoe,island_dream,island_torgerson,
                               sex_female,sex_male]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.subheader("Predicting your penguin's species:")

st.write('We predict your penguins is of the {} species'.format(prediction_species))
st.write('We used a machine learning (Random Forest) model to predict the species, the features used int his prediction '
         'are ranked by relative importance below.')

#st.image('feature_importance.png')

st.write('Below are the histograms for each continuous variable separated by penguin species. '
         'The vertical line represents your inputted value.')

fig, ax = plt.subplots()
ax=sns.displot(x=penguin_df['bill_length_mm'],
               hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill length by Species')
st.pyplot(ax)

fig,ax = plt.subplots()

ax= sns.displot(x=penguin_df['flipper_length_mm'],
                hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper length by Species')
st.pyplot(ax)


