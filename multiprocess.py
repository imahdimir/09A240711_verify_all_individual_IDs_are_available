import pandas as pd
from pathlib import Path

from multiprocessing import Pool

def read_df_lift() :
    fn = '/disk/homedirs/nber/mahdimir/chr_pos_rsid_map.p'

    df = pd.read_parquet(fn)

    df = df.drop_duplicates(subset = ['chr' , 'pos'])

    return df

LIFT = read_df_lift()

def read_bim_merge_with_lift(bim_fn , df_lift) :
    df = pd.read_csv(bim_fn , sep = '\t' , header = None)

    df = df[[0 , 3]]

    df.columns = ['chr' , 'pos']

    df = df.astype('string')

    df_cols = pd.merge(df , df_lift , on = ['chr' , 'pos'] , how = 'left')

    return df_cols

def save_gt(bim_fn , df_cols) :
    bed_fn = '/homes/nber/mahdimir/bed_files_converted_2_csv/' + Path(bim_fn).with_suffix(
            '.csv').name

    df_gt = pd.read_csv(bed_fn)

    df_gt.columns = df_cols['rsid']

    df_gt = df_gt.loc[: , df_gt.columns.notna()]

    df_gt = df_gt.loc[: , ~ df_gt.columns.duplicated()]

    od = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual/'

    fo = od + Path(bim_fn).with_suffix('.p').name
    print(fo)

    df_gt.to_parquet(fo , index = False)

    return df_gt

def job(bim_fn) :
    df_cols = read_bim_merge_with_lift(bim_fn , LIFT)
    df_gt = save_gt(bim_fn , df_cols)

def main() :
    # get_all_bim_files in the folder
    bims = Path('/homes/nber/mahdimir/plink_out/').rglob('*.bim')

    b = list(bims)

    print(len(b))

    with Pool(20) as p :
        print(20)
        p.map(job , b)

if __name__ == '__main__' :
    main()
