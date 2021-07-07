# Adapted from https://github.com/Trusted-AI/AIX360/blob/master/aix360/datasets/meps_dataset.py
# Utilization target is kept as a continuous target.
import os

import pandas as pd


def default_preprocessing(df):
    """
    1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
      and 'non-White' otherwise
    2. Restrict to Panel 19
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION.
    """
    def race(row):
        if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
            return 'White'
        return 'Non-White'

    df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns = {'RACEV2X' : 'RACE'})

    df = df[df['PANEL'] == 19]

    # RENAME COLUMNS
    df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE'] >= 0] # remove values -1

    df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

    df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

    df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

    def utilization(row):
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

    df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)

    df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})

    df = df[['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42','PCS42',
                                 'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION','PERWT15F']]

    return df


class MEPSDataset():
    """
    The Medical Expenditure Panel Survey (MEPS) [#]_ data consists of large scale surveys of families and individuals,
    medical providers, and employers, and collects data on health services used, costs & frequency of services,
    demographics, health status and conditions, etc., of the respondents.
    This specific dataset contains MEPS survey data for calendar year 2015 obtained in rounds 3, 4, and 5 of Panel 19,
    and rounds 1, 2, and 3 of Panel 20.
    See :file:`uq360/data/meps_data/README.md` for more details on the dataset and instructions on downloading/processing the data.
    References:
        .. [#] `Medical Expenditure Panel Survey data <https://meps.ahrq.gov/mepsweb/>`_
    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', 'meps_data')

        self._filepath = os.path.join(self._dirpath, 'h181.csv')
        try:
            df = pd.read_csv(self._filepath, sep=',', na_values=[])
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the h181.csv:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '../data', 'meps_data'))))
            print("See :file:`uq360/data/meps_data/README.md` for more details on the dataset and instructions "
                  "on downloading/processing the data.")
            import sys
            sys.exit(1)

        if custom_preprocessing:
            self._data = custom_preprocessing(df)

    def data(self):
        return self._data