import pandas as pd

def build_Xy_from_raw(df_raw):

    '''
    # obtenção do par variáveis e gabarito (X, y)
    '''

    y = df_raw.iloc[:, 0]
    X = df_raw.iloc[:, 1:].astype('uint8')

    return X, y

def build_mapping_from_raw(raw_map):

    '''
    # construção do mapeamento do target
    '''

    # Parse do mapeamento
    mapping = {}
    for line in raw_map.strip().splitlines():
        idx, ascii_code = line.strip().split()
        mapping[int(idx)] = chr(int(ascii_code))
    # Lista de classes ordenada por índice
    classes = [mapping[i] for i in sorted(mapping.keys())]
    
    return classes