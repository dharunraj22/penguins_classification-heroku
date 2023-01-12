import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

html=pd.read_html("https://github.com/dataprofessor/data/blob/master/penguins_cleaned.csv")
df=pd.DataFrame(html[0])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.to_csv("penguins.csv", index=False)

target=['species']
encode=['sex','island']

for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy], axis=1)
    del df[col]
        
target_mapper={'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encoder(val):
    return target_mapper[val]

df['species']=df['species'].apply(target_encoder)

X=df.drop(["species"],axis=1)
Y=df["species"]

model=RandomForestClassifier()
model.fit(X,Y)

pickle.dump(model,open('penguins_model.pkl','wb'))