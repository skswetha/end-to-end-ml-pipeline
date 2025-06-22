"""
# Outputs:

df_example:
      Name  Age
0    Alice   22
1      Bob   42
2  Charles   18
3    Daria   28
df_example_preproc:
[[-0.60461468  1.          0.          0.          0.        ]
 [ 1.59398416  0.          1.          0.          0.        ]
 [-1.04433445  0.          0.          1.          0.        ]
 [ 0.05496497  0.          0.          0.          1.        ]]
df_example2:
    Name  Age
0  Alice   22
1    Bob   42
2  Louis   30
df_example2_preproc:
[[-0.60461468  1.          0.          0.          0.        ]
 [ 1.59398416  0.          1.          0.          0.        ]
 [ 0.27482485  0.          0.          0.          0.        ]]
"""

def main():
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    df_example = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charles", "Daria"],
        "Age": [22, 42, 18, 28],
    })
    numerical_features = ["Age"]
    categorical_features = ["Name"]

    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standardize features
    ])

    # Create a pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    # Calculate means/stds and categorical categories from `df_example`
    preprocessor.fit(df_example)

    # Visualize preprocessing done on `df_example` and a new df `df_example2`
    print(f"df_example:")
    print(df_example)
    df_example_preproc = preprocessor.transform(df_example)
    print(f"df_example_preproc:")
    print(df_example_preproc)

    # new dataframe
    df_example2 = pd.DataFrame({
        "Name": ["Alice", "Bob", "Louis"],
        "Age": [22, 42, 30],
    })

    print(f"df_example2:")
    print(df_example2)
    df_example2_preproc = preprocessor.transform(df_example2)
    print(f"df_example2_preproc:")
    print(df_example2_preproc)


if __name__ == '__main__':
    main()
