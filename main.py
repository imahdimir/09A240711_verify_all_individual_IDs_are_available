"""

    """

import pandas as pd
from pathlib import Path

class Var :
    id1 = 'ID1'
    id2 = 'ID2'
    inf_type = 'InfType'
    fs = 'FS'
    po = 'PO'
    iid = 'IID'
    check = 'Check'
    suf = 'suffix'
    fol = 'folder'
    path = 'path'
    s1_check = 'S1Check'
    s2_check = 'S2Check'
    fn = 'fn'
    g1 = 'g1'
    g2 = 'g2'
    g1_plus_g2 = 'g1_plus_g2'
    g1_minus_g2 = 'g1_minus_g2'
    g1_hat = 'g1_hat'
    g2_hat = 'g2_hat'
    g1_plus_g2_hat = 'g1_plus_g2_hat'
    g1_minus_g2_hat = 'g1_minus_g2_hat'


V = Var()

# prepare all individual IDs, FS and PO

fn = '/disk/genetics/ukb/alextisyoung/haplotypes/relatives/bedfiles/hap.kin0'

dfr = pd.read_csv(fn , sep = '\t')

##
msk = dfr[V.inf_type].eq(V.fs)

dfs = dfr[msk]

##
fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_full_sibs_only.parquet'
dfs.to_parquet(fn , index = False)

##
msk = dfr[V.inf_type].eq(V.po)

dfp = dfr[msk]

##
fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_parent_offspring_only.parquet'

##
dfp.to_parquet(fn , index = False)

##


def main() :
    pass

    ##


    ##
    dfp = pd.read_parquet(fn)

    ##
    dfsp = pd.concat([dfs , dfp])

    ##
    df1 = dfsp[[V.id1]]
    df2 = dfsp[[V.id2]]

    ##
    df1.columns = [V.iid]
    df2.columns = [V.iid]

    ##
    df_iid = pd.concat([df1 , df2])

    ##

    # prepare all individual IDs from VCF files names

    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_get_all_WGS_vcf_filenames_on_UKB_RAP/out/all_WGS_filenames.txt'

    df = pd.read_csv(fn , header = None)

    ##
    df[V.iid] = df[0].str.split('_').str[0]

    ##
    df[V.iid] = df[V.iid].str.strip()
    df_iid[V.iid] = df_iid[V.iid].astype('string').str.strip()

    ##
    msk = df[V.iid].isin(df_iid[V.iid])

    df = df[msk]

    ##
    df[V.suf] = df[0].str.split('.').str[-2 :]

    df[V.suf] = df[V.suf].apply(lambda x : ".".join(x))

    ##
    msk = df[V.suf].eq('vcf.gz')

    df = df[msk]

    ##
    df[V.fol] = df[0].str[:2]

    ##

    # because the len of df and df_iid are not equal, we don't have some individual IDs in the VCF files names
    # maybe that's because of the ls doesn't work properly let's have two files first files with all individual IDs and the second one with the individual IDs from VCF files names

    ##
    path_prefix = 'project-Gjz9YXjJzVpj2P4v2vFpg4GP:/Bulk/DRAGEN WGS/Whole genome variant call files (VCFs) (DRAGEN) [500k release]'

    ##
    df[V.path] = path_prefix + '/' + df[V.fol] + '/' + df[0]

    ##
    df1 = df[[V.path]]

    # df1 = df1.iloc[:92]

    ##
    df2 = df1.copy()
    df2[V.path] = df2[V.path].str.replace('vcf.gz' , 'vcf.gz.tbi')

    ##
    df3 = pd.concat([df1 , df2])

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/all_for_dx_download.txt'
    df3.to_csv(fn , header = False , index = False)

    ##
    df1 = df[0]

    # df1 = df1.iloc[:92]

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/all_4_tabix.txt'
    df1.to_csv(fn , header = False , index = False)

    ##

    # keeping 1K siblings only

    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_full_sibs_only.parquet'
    dfs = pd.read_parquet(fn)

    ##
    df1 = dfs[[V.id1 , V.id2]]

    df1 = df1.astype('string')

    ##

    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/inp/merged.txt'

    df = pd.read_csv(fn , header = None)

    ##
    df[V.suf] = df[0].str.split('.').str[-2 :]

    df[V.suf] = df[V.suf].apply(lambda x : ".".join(x))

    ##
    msk = df[V.suf].eq('vcf.gz')

    df = df[msk]

    ##
    df[V.iid] = df[0].str.split('_').str[0]

    ##
    df1[V.s1_check] = df1[V.id1].isin(df[V.iid])
    df1[V.s2_check] = df1[V.id2].isin(df[V.iid])

    ##
    msk = df1[V.s1_check] & df1[V.s2_check]

    df1 = df1[msk]

    ##
    msk = df[V.iid].isin(df1[V.id1])
    msk |= df[V.iid].isin(df1[V.id2])

    df = df[msk]

    ##
    df[V.fol] = df[0].str[:2]

    ##

    # because the len of df and df_iid are not equal, we don't have some individual IDs in the VCF files names
    # maybe that's because of the ls doesn't work properly let's have two files first files with all individual IDs and the second one with the individual IDs from VCF files names

    ##
    path_prefix = 'project-Gjz9YXjJzVpj2P4v2vFpg4GP:/Bulk/DRAGEN WGS/Whole genome variant call files (VCFs) (DRAGEN) [500k release]'

    ##
    df[V.path] = path_prefix + '/' + df[V.fol] + '/' + df[0]

    ##
    df2 = df1.iloc[:1050]

    ##
    msk = df[V.iid].isin(df2[V.id1])
    msk |= df[V.iid].isin(df2[V.id2])

    df3 = df[msk]

    ##

    # keeping 1K siblings only

    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_full_sibs_only.parquet'
    dfs = pd.read_parquet(fn)

    ##
    df1 = dfs[[V.id1 , V.id2]]

    df1 = df1.astype('string')

    ##
    df2 = df1.iloc[:1100]

    ##
    df2[V.iid] = df2[V.id1]

    df3 = df2.copy()
    df3[V.iid] = df3[V.id2]

    ##
    df2 = df2[[V.iid]]
    df3 = df3[[V.iid]]

    ##
    df4 = pd.concat([df2 , df3])

    ##
    post_fix = '_24053_0_0.dragen.hard-filtered.vcf.gz'

    ##
    df4[V.fn] = df4[V.iid] + post_fix

    ##
    df4[V.fol] = df4[V.iid].str[:2]

    ##
    path_prefix = 'project-Gjz9YXjJzVpj2P4v2vFpg4GP:/Bulk/DRAGEN WGS/Whole genome variant call files (VCFs) (DRAGEN) [500k release]'

    ##
    df4[V.path] = path_prefix + '/' + df4[V.fol] + '/' + df4[V.fn]

    ##
    df5 = df4[[V.path]]

    ##
    df6 = df5.copy()
    df6[V.path] = df6[V.path] + '.tbi'

    ##
    df7 = pd.concat([df5 , df6])

    ##
    df7 = df7.drop_duplicates()

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/1K_sibs_for_dx_download.txt'
    df7.to_csv(fn , header = False , index = False)

    ##

    ##

    # keeping only 72 Individuals for test before 1K

    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_full_sibs_only.parquet'
    dfs = pd.read_parquet(fn)

    ##
    df1 = dfs[[V.id1 , V.id2]]

    df1 = df1.astype('string')

    ##
    df2 = df1.iloc[:72]

    ##
    df2[V.iid] = df2[V.id1]

    df3 = df2.copy()
    df3[V.iid] = df3[V.id2]

    ##
    df2 = df2[[V.iid]]
    df3 = df3[[V.iid]]

    ##
    df4 = pd.concat([df2 , df3])

    ##
    post_fix = '_24053_0_0.dragen.hard-filtered.vcf.gz'

    ##
    df4[V.fn] = df4[V.iid] + post_fix

    ##
    df4[V.fol] = df4[V.iid].str[:2]

    ##
    path_prefix = 'project-Gjz9YXjJzVpj2P4v2vFpg4GP:/Bulk/DRAGEN WGS/Whole genome variant call files (VCFs) (DRAGEN) [500k release]'

    ##
    df4[V.path] = path_prefix + '/' + df4[V.fol] + '/' + df4[V.fn]

    ##
    df5 = df4[[V.path]]

    ##
    df6 = df5.copy()
    df6[V.path] = df6[V.path] + '.tbi'

    ##
    df7 = pd.concat([df5 , df6])

    ##
    df7 = df7.drop_duplicates()

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/pairs_72_for_dx_download.txt'
    df7.to_csv(fn , header = False , index = False)

    ##
    df8 = df4[[V.fn]]

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/pairs_72_4_tabix.txt'
    df8.to_csv(fn , header = False , index = False)

    ##

    ##

    # keeping 1K Individuals

    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_full_sibs_only.parquet'
    dfs = pd.read_parquet(fn)

    ##
    df1 = dfs[[V.id1 , V.id2]]

    df1 = df1.astype('string')

    ##
    df2 = df1.iloc[:1100]

    ##
    df2[V.iid] = df2[V.id1]

    df3 = df2.copy()
    df3[V.iid] = df3[V.id2]

    ##
    df2 = df2[[V.iid]]
    df3 = df3[[V.iid]]

    ##
    df4 = pd.concat([df2 , df3])

    ##
    post_fix = '_24053_0_0.dragen.hard-filtered.vcf.gz'

    ##
    df4[V.fn] = df4[V.iid] + post_fix

    ##
    df4[V.fol] = df4[V.iid].str[:2]

    ##
    path_prefix = 'project-Gjz9YXjJzVpj2P4v2vFpg4GP:/Bulk/DRAGEN WGS/Whole genome variant call files (VCFs) (DRAGEN) [500k release]'

    ##
    df4[V.path] = path_prefix + '/' + df4[V.fol] + '/' + df4[V.fn]

    ##
    df5 = df4[[V.path]]

    ##
    df6 = df5.copy()
    df6[V.path] = df6[V.path] + '.tbi'

    ##
    df7 = pd.concat([df5 , df6])

    ##
    df7 = df7.drop_duplicates()

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/1k_4_download.txt'
    df7.to_csv(fn , header = False , index = False)

    ##
    df8 = df4[[V.fn]]

    ##
    fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/1k_4_tabix.txt'
    df8.to_csv(fn , header = False , index = False)

    ##

    ##

    ##
    import plinkio
    from plinkio import plinkfile

    ##
    fn = '/homes/nber/mahdimir/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bed'

    df = pd.read_csv(fn , sep = '\t' , header = None)

    ##
    plink_file = plinkfile.open(fn)

    # Access sample and locus data
    samples = plink_file.get_samples()
    loci = plink_file.get_loci()

    # Print some basic information
    print(f"Number of samples: {len(samples)}")
    print(f"Number of loci: {len(loci)}")

    ##

    ##
    import pandas as pd
    import numpy as np
    from plinkio import plinkfile

    fn = '/homes/nber/mahdimir/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bed'

    # Open the .bed file along with its corresponding .bim and .fam files
    plink_file = plinkfile.open(fn)

    # Access sample and locus data
    samples = plink_file.get_samples()
    loci = plink_file.get_loci()

    # Create lists to hold sample and locus IDs
    sample_ids = [f"{sample.fid}_{sample.iid}" for sample in samples]
    locus_ids = [locus.name for locus in loci]

    # Initialize an empty list to hold genotype data
    genotype_data = []

    # Read genotype data for each sample
    for sample in plink_file :
        genotype_data.append(sample)

    # Convert the genotype data to a numpy array
    genotype_array = np.array(genotype_data)

    # Create a pandas DataFrame from the numpy array
    df = pd.DataFrame(genotype_array , index = sample_ids , columns = locus_ids)

    # Print the DataFrame
    print(df)

    # Close the plink file
    plink_file.close()

    ##

    ##
    fn = '/homes/nber/mahdimir/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bim'

    df = pd.read_csv(fn , sep = '\t' , header = None)

    ##
    fn = '/disk/homedirs/nber/mahdimir/snps.lifted'

    df_lifted = pd.read_csv(fn , sep = '\t' , header = None)

    ##
    df1 = df[[0 , 3]]

    ##
    df1.columns = ['chr' , 'pos']

    ##
    df1 = df1.astype('string')

    ##
    df2 = df_lifted[[0 , 2 , 3]]

    ##
    df2.columns = ['chr' , 'pos' , 'rsid']

    ##
    df2 = df2.astype('string')

    ##
    df2['chr'] = df2['chr'].str.split('chr').str[-1]

    ##
    df3 = pd.merge(df1 , df2 , on = ['chr' , 'pos'] , how = 'left')

    ##
    df3['rsid'].isna().sum()

    ##
    from pysnptools.snpreader import Bed

    # Specify the path to your .bed file (without the .bed extension)
    bed_file_path = '/homes/nber/mahdimir/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bed'

    # Read the .bed file
    bed = Bed(bed_file_path)

    # Load the data into memory
    genotype_data = bed.read()

    # Extract SNPs and individual data
    snp_data = genotype_data.val  # SNP data as a numpy array
    individuals = genotype_data.iid  # Individual IDs
    snps = genotype_data.sid  # SNP IDs

    # Display some of the data
    print(snp_data.shape)  # (number of individuals, number of SNPs)
    print(individuals[:5])  # First 5 individual IDs
    print(snps[:5])  # First 5 SNP IDs

    ##

    ##

    # prepare df_lifted

    ##
    fn = '/disk/homedirs/nber/mahdimir/snps.lifted'

    df_lifted = pd.read_csv(fn , sep = '\t' , header = None)

    ##
    df2 = df_lifted[[0 , 2 , 3]]

    ##
    df2.columns = ['chr' , 'pos' , 'rsid']

    ##
    df2 = df2.astype('string')

    ##
    df2['chr'] = df2['chr'].str.split('chr').str[-1]

    ##
    df2['chr'] = df2['chr'].str.strip()

    ##
    fn = '/disk/homedirs/nber/mahdimir/chr_pos_rsid_map.p'

    df2.to_parquet(fn , index = False)

    ##

    ##

    # read a bim file prepare it for merging with the lifted file

    ##
        fn = '/homes/nber/mahdimir/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.bim'
        df_lift = df2.copy()

    ##
    df_lift = df_lift.drop_duplicates(subset = ['chr' , 'pos'])

    ##
    df = pd.read_csv(fn , sep = '\t' , header = None)

    ##
    df = df[[0 , 3]]

    ##
    df.columns = ['chr' , 'pos']

    ##
    df = df.astype('string')

    ##
    df1 = pd.merge(df, df_lift , on = ['chr' , 'pos'] , how = 'left')

    ##


    ##


    ##

    # read converted bed to csv file then rename the columns

    ##
        fn = '/homes/nber/mahdimir/bed_files_converted_2_csv/5968696_24053_0_0.dragen.hard-filtered.vcf.filtered.csv'

        df_cols = df1.copy()

    ##
    df_gt = pd.read_csv(fn)

    ##
    df_gt.columns = df_cols['rsid']

    ##
    df_gt = df_gt.loc[:, df_gt.columns.notna()]

    ##
    df_gt = df_gt.loc[: , ~ df_gt.columns.duplicated()]

    ##
    dyr = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual/'
    fo = dyr + Path(fn).with_suffix('.p').name
    print(fo)

    ##
    df_gt.to_parquet(fo , index = False)

    ##


    ##


    ##

    # get_all_bim_files in the folder
    bims = Path('/homes/nber/mahdimir/plink_out/').rglob('*.bim')

    b = list(bims)

    ##
    def read_df_lift():
        fn = '/disk/homedirs/nber/mahdimir/chr_pos_rsid_map.p'

        df = pd.read_parquet(fn)

        df = df.drop_duplicates(subset = ['chr' , 'pos'])

        return df

    ##
    df_lift = read_df_lift()

    ##

    ##
    def read_bim_merge_with_lift(bim_fn , df_lift) :
        df = pd.read_csv(bim_fn , sep = '\t' , header = None)

        df = df[[0 , 3]]

        df.columns = ['chr' , 'pos']

        df = df.astype('string')

        df_cols = pd.merge(df , df_lift , on = ['chr' , 'pos'] , how = 'left')

        return df_cols


    ##
    def save_gt(bim_fn , df_cols) :
        bed_fn = '/homes/nber/mahdimir/bed_files_converted_2_csv/' + Path(bim_fn).with_suffix('.csv').name

        df_gt = pd.read_csv(bed_fn)

        df_gt.columns = df_cols['rsid']

        df_gt = df_gt.loc[:, df_gt.columns.notna()]

        df_gt = df_gt.loc[: , ~ df_gt.columns.duplicated()]

        od = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual/'

        fo = od + Path(fn).with_suffix('.p').name
        print(fo)

        df_gt.to_parquet(fo , index = False)

        return df_gt


    ##
    df_lift = read_df_lift()

    # get_all_bim_files in the folder
    bims = Path('/homes/nber/mahdimir/plink_out/').rglob('*.bim')

    b = list(bims)

    ##
    def job(bim_fn):
        df_cols = read_bim_merge_with_lift(bim_fn , df_lift)
        df_gt = save_gt(bim_fn , df_cols)


    ##
    from multiprocess import Pool

    ##
    with Pool(20) as p:
        print(20)
        p.map(job , b[:10])

    ##


    ##

    # make a combined genotype dataset with IID and RSID

    ps = Path('/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual').rglob('*.parquet')

    p = list(ps)

    p


    ##
    f1 = p[0]
    f2 = p[1]

    ##
    # rs201106462

    ##
    f1.name.split('_')[0]


    ##
    df1 = pd.read_parquet(f1)
    df1['IID'] = f1.name.split('_')[0]

    df1 = df1[['IID', 'rs201106462']]

    ##

    df2 = pd.read_parquet(f2)
    df2['IID'] = f2.name.split('_')[0]


    ##
    df3 = pd.concat([df1 , df2] , axis = 0)

    ##
    df4 = df3[['IID']]

    ##
    ps = Path(
    '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual').rglob(
    '*.parquet')

    df = pd.DataFrame()
    for i, fn in enumerate(ps):
        df_i = pd.read_parquet(fn)
        df_i['IID'] = fn.name.split('_')[0]


        df_i = df_i[['IID' , 'rs201106462']]

        df = pd.concat([df , df_i], axis = 0)

        print(i)


    ##

    ##
    import dask.dataframe as dd

    # Read all Parquet files
    df = dd.read_parquet("/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual/'*.p")


    ##

    # Write combined Parquet file
    df.to_parquet("/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/WGS_GT_combined.p")

##
"/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med


find gt_by_individual/*.parquet | python3 -m joinem out.parquet --progress



rename .p .parquet *.p





##
from pyspark.sql import SparkSession

##
# Initialize Spark session
spark = SparkSession.builder \
    .appName("Merge Parquet Files") \
    .getOrCreate()

##
# Read multiple Parquet files into a DataFrame
parquet_files_pat = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual/*.parquet'
df = spark.read.parquet(parquet_files_pat)


##


##
fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/merged.parquet'
# Write the DataFrame back to a single Parquet file
df.write.mode("overwrite").parquet(fn)

##
# Stop the Spark session
spark.stop()

##



##

from pyarrow import dataset as ds

dyr = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual/'

data = ds.dataset(dyr, format="parquet")


##
do = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/merged.parquet'
ds.write_dataset(
    data,
    do,
    format="parquet",
)

##
import pandas as pd

do = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/merged.parquet'

df = pd.read_parquet(do)


##
import polars as pl

do = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/merged.parquet'


df = pl.scan_parquet(do).select(['IID' , 'rs201106462']).collect()

##


##
import dask.dataframe as dd
import pandas as pd
from pathlib import Path

ps = Path(
    '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual').rglob(
    '*.parquet')

p = list(ps)

len(p)

##
# List of Parquet files
parquet_files = p

# Function to read a single column from a Parquet file
def read_column(file_path):
    # Read the Parquet file into a Dask DataFrame
    pf = ParquetFile(file_path)
    column_names = pf.columns
    col = 'rs201106462'
    if col in column_names:
        df = dd.read_parquet(file_path, columns=[col])
        df['IID'] = file_path.name.split('_')[0]
        return df

##
# Read the specified column from each file in parallel
columns = [read_column(file) for file in parquet_files]

# Concatenate all the columns into a single DataFrame
combined_df = dd.concat(columns, axis=0)

# Convert the Dask DataFrame to a Pandas DataFrame (if needed)
combined_df = combined_df.compute()

print(combined_df)


##
from fastparquet import ParquetFile


# Read the Parquet file metadata
pf = ParquetFile(p[1])

# Get the column names
column_names = pf.columns

column_names


##
'IID' in column_names

##
df = pd.read_parquet(p[0])

##
p[0].name.split('_')[0]

##




##

# read parquet files in parallel keep only one column and save
do = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/one_snp_only/'

def read_and_keep_one_snp(file_path):
    # Read the Parquet file into a Dask DataFrame

    df = pd.read_parquet(file_path)
    col = 'rs201106462'
    if not col in df.columns:
        return
    df = df[[col]]
    df['IID'] = file_path.name.split('_')[0]
    fo = do + file_path.name
    df.to_parquet(fo, index = False)
    print(fo)

##
df = read_and_keep_one_snp(p[0])

##
from multiprocessing import Pool

ps = Path(
    '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/gt_by_individual').rglob(
    '*.parquet')


with Pool(20) as p:
    print(20)
    p.map(read_and_keep_one_snp , ps)

##



##
# make a merged parquet file from the saved parquet files
# then make g1 + g2 and g1 - g2 and save them
# then prepare the one for imputed data
# then just do the OLS and send the result to Alex
# that's enough for today

import dask.dataframe as dd

ps = Path(
    '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/one_snp_only').rglob(
    '*.parquet')




##
# Read the specified column from each file in parallel
columns = [dd.read_parquet(file) for file in ps]

##
# Concatenate all the columns into a single DataFrame
combined_df = dd.concat(columns, axis=0)


# Convert the Dask DataFrame to a Pandas DataFrame (if needed)
combined_df = combined_df.compute()

print(combined_df)

##
df = combined_df

##
df = df[['IID' , 'rs201106462']]

##
fo = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/one_snp_merged.parquet'

df.to_parquet(fo , index = False)

##
fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/out/hap_kin0_full_sibs_only.parquet'

df = pd.read_parquet(fn)
dfgt = pd.read_parquet(fo)

##
df = df[[V.id1, V.id2]]

##
df = df.astype('string')

##
dfgt = dfgt.set_index('IID')

##
df[V.g1] = df[V.id1].map(dfgt['rs201106462'])
df[V.g2] = df[V.id2].map(dfgt['rs201106462'])

##
df[V.g1_plus_g2] = df[V.g1] + df[V.g2]
df[V.g1_minus_g2] = df[V.g1] - df[V.g2]

##
df = df.dropna()

##
df.to_parquet('/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/sibs_wgs.parquet' , index = False)

##
fn = '/var/genetics/ws/mahdimir/local/prj_data/1144_plot_all_70_bins/med/hc/c1.prq'

df1 = pd.read_parquet(fn)


##
df2 = pd.read_csv('/homes/nber/mahdimir/snps.lifted', sep = '\t', header = None)

##
snp = 'rs201106462'
df1 = df1[[V.iid, snp]]

##
df1 = df1.set_index(V.iid)

df[V.g1_hat] = df[V.id1].map(df1[snp])
df[V.g2_hat] = df[V.id2].map(df1[snp])

##
df[V.g1_plus_g2_hat] = df[V.g1_hat] + df[V.g2_hat]

##
df[V.g1_minus_g2_hat] = df[V.g1_hat] - df[V.g2_hat]

##
fn = '/var/genetics/ws/mahdimir/local/prj_data/1174_verify_all_individual_IDs_are_available/med/sibs_model_data.parquet'


df.to_parquet(fn , index = False)

##





##

##




##








##








    ##


    ##

    ##



    ##


    ##






    ##


    ##
