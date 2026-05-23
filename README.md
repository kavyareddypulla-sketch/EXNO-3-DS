## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```

import pandas as pd
import numpy as np
from scipy import stats

```
```

df = pd.read_csv("C:/Users/acer/Downloads/data.csv")
print("Dataset Loaded Successfully")
df.head()

```

<img width="539" height="318" alt="image" src="https://github.com/user-attachments/assets/8d8271f9-8f5e-475c-9441-fce1b0189b14" />

```

from sklearn.preprocessing import OrdinalEncoder
climate = ['Cold', 'Warm', 'Hot', 'Very Hot']
oe = OrdinalEncoder(categories=[climate])
df['Ord_1_Encoded'] = oe.fit_transform(df[['Ord_1']])
df.head()

```

<img width="680" height="235" alt="image" src="https://github.com/user-attachments/assets/9022cca3-9164-400a-9f50-d51690d1b873" />

```

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Ord_2_Encoded'] = le.fit_transform(df['Ord_2'])
df.head()

```

<img width="761" height="278" alt="image" src="https://github.com/user-attachments/assets/8d25319d-0df6-4951-822c-269950658f21" />

```

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df[['City']])
encoded_df = pd.DataFrame(
encoded,
columns=ohe.get_feature_names_out(['City'])
)
df_ohe = pd.concat([df, encoded_df], axis=1)
df_ohe.head()

```

<img width="845" height="197" alt="image" src="https://github.com/user-attachments/assets/2db2e866-3cd4-47d7-b3cc-a9624ec3aa2c" />

```

pd.get_dummies(df, columns=['City'])

```

<img width="678" height="253" alt="image" src="https://github.com/user-attachments/assets/fbf7be83-d991-4538-b8f4-7a4ce51be1bb" />

```

from category_encoders import BinaryEncoder
be = BinaryEncoder()
binary_data = be.fit_transform(df['Ord_2'])
df_binary = pd.concat([df, binary_data], axis=1)
df_binary.head()
pip install category_encoders

```

<img width="769" height="204" alt="image" src="https://github.com/user-attachments/assets/54428e68-aae3-4a20-a646-47943f44f34f" />

```

from category_encoders import TargetEncoder
te = TargetEncoder()
target_encoded = te.fit_transform(
X=df[['City']],
y=df['Target']
)
df_target = pd.concat([df, target_encoded], axis=1)
df_target.head()

```

<img width="718" height="166" alt="image" src="https://github.com/user-attachments/assets/b09036ab-8bcc-4f77-9379-d4d737b467d5" />

```
df = pd.read_csv("C:/Users/acer/Downloads/Data_to_Transform.csv")
df.head()

```

<img width="636" height="190" alt="image" src="https://github.com/user-attachments/assets/bda85b04-ee16-4f65-abe8-aa9afd22fcdc" />

```

df.skew()

```

<img width="280" height="99" alt="image" src="https://github.com/user-attachments/assets/14020186-ab94-4a30-90bf-c2f51be8cd7e" />

```

np.log(df["Highly Positive Skew"])

```

<img width="429" height="217" alt="image" src="https://github.com/user-attachments/assets/6612422c-ba76-4e12-bd17-10b96f2be4b3" />

```

np.reciprocal(df["Moderate Positive Skew"])

```

<img width="487" height="209" alt="image" src="https://github.com/user-attachments/assets/a1cbda36-800c-41cc-9157-3092ecea263e" />

```

df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(
    df["Highly Positive Skew"]
)
 
df

```

<img width="792" height="360" alt="image" src="https://github.com/user-attachments/assets/70c4d0fa-515a-43bf-be7c-9ce295ab7098" />

```

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
 
df["Moderate Negative Skew_1"] = qt.fit_transform(
    df[["Moderate Negative Skew"]]
)
 
df

```

<img width="784" height="323" alt="image" src="https://github.com/user-attachments/assets/2ee73c19-d480-49cd-be0c-ff342cbfa18e" />

```

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(
df['Moderate Negative Skew'],
line='45'
)
plt.show()

```

<img width="573" height="378" alt="image" src="https://github.com/user-attachments/assets/fcbcc070-5973-417e-b7d8-6d7123f886c7" />

```
````
```

sm.qqplot(
df['Moderate Negative Skew_1'],
line='45'
)
plt.show()

```

<img width="567" height="372" alt="image" src="https://github.com/user-attachments/assets/edd3c141-e077-46bb-b4f4-09f5008f9293" />

 ```

sm.qqplot(
df['Highly Negative Skew'],
line='45'
)
plt.show()

```

<img width="598" height="400" alt="image" src="https://github.com/user-attachments/assets/1db49916-5445-4030-8cc3-157558b08eef" />

```

sm.qqplot(
np.reciprocal(df["Moderate Negative Skew"]),
line='45'
)
plt.show()

```

<img width="588" height="354" alt="image" src="https://github.com/user-attachments/assets/329b3cf3-08f2-48b5-b0f7-1aa9cc19bc61" />

# RESULT:
        
Thus,the program is executed successfully.  
