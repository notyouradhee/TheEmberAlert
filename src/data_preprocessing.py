import pandas as pd
import os

PROCESSED_DIR = 'data/processed'

def preprocess_and_combine(country_keyword, output_name):
    combined_df = []

    for filename in sorted(os.listdir(PROCESSED_DIR)):
        if filename.endswith('.csv') and country_keyword in filename and not filename.endswith(f"{output_name}_combined.csv"):
            print(f"✅ Processing: {filename}")
            file_path = os.path.join(PROCESSED_DIR, filename)
            df = pd.read_csv(file_path)

            # Convert acq_date and acq_time to datetime if needed
            if 'acq_date' in df.columns and 'acq_time' in df.columns:
                df['datetime'] = pd.to_datetime(
                    df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
                    format='%Y-%m-%d %H%M'
                )
                df.drop(['acq_date', 'acq_time'], axis=1, inplace=True)

            df.dropna(inplace=True)
            combined_df.append(df)

    if combined_df:
        final_df = pd.concat(combined_df, ignore_index=True)
        save_path = os.path.join(PROCESSED_DIR, f"{output_name}_combined.csv")
        final_df.to_csv(save_path, index=False)
        print(f"🎉 Combined file saved: {save_path}")
    else:
        print(f"⚠️ No matching files found for '{country_keyword}' in {PROCESSED_DIR}")

if __name__ == "__main__":
    preprocess_and_combine('Nepal', 'nepal')
    preprocess_and_combine('Republic_of_Korea', 'korea')
