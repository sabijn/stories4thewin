import numpy as np
import folia.main as folia #https://foliapy.readthedocs.io/en/latest/folia.html#loading-a-document
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm.notebook import tqdm

def add_column(existing, new, function, df):
     """
     Function to add new columns based on an old one with a lambda function on rows.
     existing (str): column to apply the function at
     new (str): column to store output
     function (func): function to apply
     df (Pandas.dataframe): dataframe to modify

     Output - df (Pandas.dataframe)
     """
     df[new] = df.apply(lambda row: function(row[existing]), axis = 1) 
     return df

def create_BS_lexicon(BSdir, nlp, outputdir):
     BSlexicon = defaultdict(int)
     for file in tqdm(BSdir.glob('*.xml')):
          doc = folia.Document(file=str(file))
          enriched_doc = nlp(doc.text())
          for token in enriched_doc:
               BSlexicon[(token.lemma_, token.pos_)] += 1

     df_lexicon = pd.DataFrame([(k[0], k[1], v) for k, v in BSlexicon.items()],
     columns=['lemma', 'pos', 'freq'])

     df_lexicon.to_csv(outputdir)



