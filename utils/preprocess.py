import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['label'] = df.apply(lambda x: 1 if x['src_ip'].startswith('192.168') else 0, axis=1)
    return df
