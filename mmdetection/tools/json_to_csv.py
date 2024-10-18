import pandas as pd
import json

def create_prediction_dataframe(path):
    with open(path+'/test.bbox.json') as f:
        json_data = json.load(f)
        
    df = pd.DataFrame(json_data)

    df['image_id'] = df['image_id'].apply(lambda x: f"test/{x:04d}.jpeg")

    grouped = df.groupby('image_id')

    prediction_strings = []
    image_ids = []

    for image_id, group in grouped:
        predictions = [
            f"{row['category_id']} {row['score']} " + ' '.join(map(str, row['bbox']))
            for _, row in group.iterrows()
        ]
        prediction_strings.append(' '.join(predictions))
        image_ids.append(image_id)

    result = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': image_ids
    })

    result[['PredictionString','image_id']].to_csv(path+'/output.csv',index=False)