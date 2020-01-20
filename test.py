import pandas as pd

df1 = pd.DataFrame(data={'col1':['A','A','B','B'], 'col2': ['g','d','h','p'], 'Val1': [4,3,5,7], 'Val2': [6,8,10,14]}, columns=['col1','col2','Val1','Val2'])

df2 = pd.DataFrame(data={'col1':['A','B'], 'Val1': [2,1], 'Val2': [3,4]}, columns=['col1','Val1','Val2'])

print (df1)
print (df2)

## join df1, df2

merged_df = pd.merge(left=df1, right=df2, how='inner', on='col1')

print (merged_df)
merged_df['Val1'] = merged_df['Val1_x']/merged_df['Val1_y']
merged_df['Val2'] = merged_df['Val2_x']/merged_df['Val2_y']

merged_df.drop(columns=['Val1_x', 'Val1_y', 'Val2_x', 'Val2_y'], inplace=True)

print (merged_df)

