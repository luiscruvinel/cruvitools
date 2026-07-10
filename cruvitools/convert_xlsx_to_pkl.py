import pandas as pd
from pathlib import Path

raw_data_dir = Path('who-data-raw')

def convert_xlsx_to_pkl(xlsx_path, pkl_path):
    """
    Convert an xlsx file to a pickle file.
    """
    df = pd.read_excel(xlsx_path)
    df.to_pickle(pkl_path)

for source in raw_data_dir.rglob('*2007*.xlsx'):
    print(source)
    source_name = source.name
    pkl_target_dir = Path().parent / 'who-data-processed'
    pkl_file_name = source.stem + '.pkl'
    pkl_final_path = pkl_target_dir / pkl_file_name

    convert_xlsx_to_pkl(source, pkl_final_path)

# https://www.who.int/tools/child-growth-standards
# https://www.who.int/tools/growth-reference-data-for-5to19-years/indicators

# 1 month = 30.4375 days (instructions)