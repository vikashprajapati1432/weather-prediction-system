import pandas as pd
import numpy as np

def generate_sample_data(num_samples=1000):
    """
    Generates a synthetic weather dataset similar to weatherAUS.csv 
    for initial testing of the pipeline.
    """
    np.random.seed(42)
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=num_samples),
        'Location': np.random.choice(['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'], num_samples),
        'MinTemp': np.random.uniform(5, 25, num_samples),
        'MaxTemp': np.random.uniform(15, 40, num_samples),
        'Rainfall': np.random.exponential(scale=2, size=num_samples),
        'Evaporation': np.random.uniform(0, 15, num_samples),
        'Sunshine': np.random.uniform(0, 14, num_samples),
        'WindGustDir': np.random.choice(['W', 'E', 'N', 'S', 'NW', 'NE', 'SW', 'SE'], num_samples),
        'WindGustSpeed': np.random.uniform(10, 100, num_samples),
        'WindDir9am': np.random.choice(['W', 'E', 'N', 'S', 'NW', 'NE', 'SW', 'SE'], num_samples),
        'WindDir3pm': np.random.choice(['W', 'E', 'N', 'S', 'NW', 'NE', 'SW', 'SE'], num_samples),
        'WindSpeed9am': np.random.uniform(0, 40, num_samples),
        'WindSpeed3pm': np.random.uniform(0, 40, num_samples),
        'Humidity9am': np.random.uniform(20, 100, num_samples),
        'Humidity3pm': np.random.uniform(10, 90, num_samples),
        'Pressure9am': np.random.uniform(990, 1040, num_samples),
        'Pressure3pm': np.random.uniform(990, 1040, num_samples),
        'Cloud9am': np.random.randint(0, 9, num_samples),
        'Cloud3pm': np.random.randint(0, 9, num_samples),
        'Temp9am': np.random.uniform(10, 30, num_samples),
        'Temp3pm': np.random.uniform(15, 35, num_samples),
        'RainToday': np.random.choice(['No', 'Yes'], num_samples),
        'RainTomorrow': np.random.choice(['No', 'Yes'], num_samples)
    }
    
    df = pd.DataFrame(data)
    # Add some correlation for humidity and rain
    df.loc[df['Humidity3pm'] > 70, 'RainTomorrow'] = 'Yes'
    df.loc[df['Humidity3pm'] < 40, 'RainTomorrow'] = 'No'
    
    df.to_csv('weatherAUS.csv', index=False)
    print("Synthetic weatherAUS.csv generated successfully!")

if __name__ == "__main__":
    generate_sample_data()
