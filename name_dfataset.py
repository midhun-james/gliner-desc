# #function to take 5 rows of a excel as dictionary result
import sys
import pandas as pd
sys.path.append("..")
from path import GLINER_PATH, SPACY_PATH,FILE_PATH
def execel_to_dict(FILE_PATH):
    import pandas as pd
    df = pd.read_excel(FILE_PATH, nrows=5)

    return df.to_dict(orient='records')
data = execel_to_dict(FILE_PATH)
data=pd.DataFrame(data)
print(data)