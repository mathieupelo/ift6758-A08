import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


##########################################################################################
# Milestone 1
##########################################################################################

def shots_goals (data: pd.DataFrame, saison: int, log: bool):
    df = data[(data['gameId']//1000000)== saison]
    result = df.groupby(['shotCategory','goalFlag']).size().unstack(fill_value=0)
    result['Total']=result[False]+result[True]
    result['Efficacité']=np.round(result[True]/result['Total']*100,2)
    largeur_barre = 0.45
    indice = np.arange(len(result.index))
    plt.figure(figsize=(10, 6))
    plt.bar(indice, result['Total'], largeur_barre, label='TOTAL SHOTS')
    plt.bar(indice, result[True], largeur_barre, label='GOALS')
    plt.xlabel('Types de tirs')
    if log:
        plt.yscale('log')
    plt.title(f'Buts par types de tirs {saison}/{saison+1}')
    plt.xticks(indice, result.index)
    for i in range(len(result.index)):
        plt.text(indice[i], (result['Total'][i]), f'{result["Efficacité"][i]:.2f}%', ha='center', va='center', color='black')
    plt.legend()

    plt.tight_layout()
    plt.show()


def Distance_goals (data : pd.DataFrame, saison: int):
    df = data[(data['goalFlag'] == True) & (data['gameId']//1000000 == saison)].copy()
    df['distance'] = np.sqrt((90-np.absolute(df['coord_x'])).pow(2) + df['coord_y'].pow(2))
    probability=[]
    x=[]
    for i in range(0,110,5):
        probability.append(100*len(df[(df['distance'] > i) & (df['distance'] < i+5)])/len(df))
        x.append(i+2.5)
        
    plt.figure(figsize=(6, 4))

    plt.plot(x, probability, label=f'Saison {saison}/{saison+1}', color='b', marker='o')

    plt.xlabel('Distance(pd)')
    plt.ylabel('Chances que le tir soit un but (%)')
    plt.title('Distance vs Goal chances')
    plt.legend()

    plt.grid()
    plt.tight_layout()
    plt.show()


def Distance_goals_shots (data: pd.DataFrame, saison: int):
    df = data[(data['goalFlag'] == True) & (data['gameId']//1000000 == saison)].copy()
    df['distance'] = np.sqrt((90-np.absolute(df['coord_x'])).pow(2) + df['coord_y'].pow(2))
    
    for tir in data['shotCategory'].dropna().unique():
        probability_tir=[]
        x=[]
        for i in range(0,110,5):
            probability_tir.append(len(df[(df['distance'] > i) & (df['distance'] < i+5)& (df['shotCategory'] == tir)])/len(df[df['shotCategory']== tir]))
            x.append(i+2.5)
        plt.plot(x, probability_tir, label=f'{tir} {saison}/{saison+1}', marker='o')

    plt.xlabel('Distance(pd)')
    plt.ylabel('Probabilité')
    plt.title('Distance vs Goal chances')
    plt.legend()

    plt.grid()
    plt.tight_layout()
    plt.show()


def Offensive_coords(data_1:pd.DataFrame):
    dt=pd.DataFrame()
    for gameid in data_1['gameId'].unique():
      df= pd.DataFrame()
      dire = data_1[data_1['gameId']== gameid].reset_index().loc[0,'homeRinkSide']
      for i in range (1,5):
          df1=data_1[(data_1['gameId']== gameid) & (data_1['prd']== i)].copy()
          if dire == 'left':
                if i % 2 != 0:
                
                   df1['new_x']= df1.apply(lambda row: 90 - row['coord_x'] if row['hostTeam']==row['team'] else 90 + row['coord_x'], axis=1)
                   df1['new_y']= df1.apply(lambda row: row['coord_y'] if row['hostTeam']==row['team'] else (-1)*row['coord_y'], axis=1)
                else:
                   df1['new_x']= df1.apply(lambda row: 90 - row['coord_x'] if row['hostTeam']!=row['team'] else 90 + row['coord_x'], axis=1) 
                   df1['new_y']= df1.apply(lambda row: row['coord_y'] if row['hostTeam']!=row['team'] else (-1)*row['coord_y'], axis=1)        
          else:
                if i % 2 != 0:
                
                   df1['new_x']= df1.apply(lambda row: 90 - row['coord_x'] if row['hostTeam']!=row['team'] else 90 + row['coord_x'], axis=1)
                   df1['new_y']= df1.apply(lambda row: row['coord_y'] if row['hostTeam']!=row['team'] else (-1)*row['coord_y'], axis=1)
                else:
                   df1['new_x']= df1.apply(lambda row: 90 - row['coord_x'] if row['hostTeam']==row['team'] else 90 + row['coord_x'], axis=1)
                   df1['new_y']= df1.apply(lambda row: row['coord_y'] if row['hostTeam']==row['team'] else (-1)*row['coord_y'], axis=1)
          df=pd.concat([df,df1], axis=0, ignore_index=True)
      dt=pd.concat([dt, df], axis=0, ignore_index=True)
    return dt


def Taux_ligue(dt_f: pd.DataFrame, saison: int):
    taux_tir=[]
    df=dt_f[dt_f['gameId']//1000000 == saison]
    for i in range(0,90,5):
        row=[]
        for j in range(-45,40,5):
            row.append(len(df[ (df['new_x']< i+5) & (df['new_x']> i) & (df['new_y']< j+5) & (df['new_y']> j)])/len(df['gameId'].unique()))
        taux_tir.append(row)
    return taux_tir


def Taux_team(dt_f: pd.DataFrame, team: str, saison: int):
    taux_tir_team=[]
    df = dt_f[(dt_f['team'] == team) & (dt_f['gameId']//1000000 == saison)]
    for i in range(0,90,5):
        row=[]
        for j in range(-45,40,5):
            row.append(len(df[ (df['new_x']< i+5) & (df['new_x']> i) & (df['new_y']< j+5) & (df['new_y']> j)])/len(df['gameId'].unique()))
        taux_tir_team.append(row)
    return taux_tir_team


##########################################################################################
# Milestone 2
##########################################################################################

def rename_feature(feature: str):
    """
    Retourne le nom de la caractéristique à partir du nom de la variable

    Parameters
    ----------
    feature : str
        Nom de la variable
    
    Returns
    -------
    feature_name: str
        Nom de la caractéristique
    label_name: str
        Nom à utiliser pour les figures
    """
    if feature == 'distance_goal':
        label_name = 'Distance du filet (pieds)'
        feature_name = 'distance'
    elif feature == 'angle_goal':
        label_name = 'Angle du tir (degrés)'
        feature_name = "angle"
    else:
        label_name = feature
        feature_name = feature

    return feature_name, label_name

def hist_shots_goals_feature(data: pd.DataFrame, feature:str, transform: str, save: bool):
    """
    Historgramme du nombre de tirs séparés par but ou non but selon une `feature` donnée

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe contenant les données des tirs
    feature : str
        Nom de la colonne à utiliser en x
    transform : str
        Transformation à appliquer à la colonne `feature`. Peut être None, 'log', 
        'logit', etc. Voir la référence pour la liste des transformations disponibles
    save : bool
        Si True, sauvegarde la figure dans le folder `figures`

    References
    ----------
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html
    """
    df = data.copy()

    # Regrouper les distance en bins. Par défaut, 10 bins de même largeur
    hist, bins = np.histogram(df[feature])

    # Définir style color-blind friendly
    # plt.style.use('seaborn-v0_8-colorblind')

    # Plot les histogrammes selon si un tir a mené à un but ou non
    plt.figure(figsize=(6, 4))
    plt.hist(df[df['is_goal']==0][feature], bins=bins, label='Non but')
    plt.hist(df[df['is_goal']==1][feature], bins=bins, label='But')

    # Étiqueté l'axe x
    feature_name, xlabel = rename_feature(feature)
    plt.xlabel(xlabel)

    # Étiqueté l'axe y
    if transform is not None:
        plt.yscale(transform)
        plt.ylabel(f'Nombre de tirs ({transform})')
        filename = f'shots_goals_{feature_name}_log.png'
    else:
        plt.ylabel('Nombre de tirs')
        filename = f'shots_goals_{feature_name}.svg'

    plt.legend()
    plt.title(f'Nombre de tirs et buts selon {feature_name} du filet')
    
    # Sauvegarder la figure
    if save:
        plt.savefig(f'../figures/{filename}')


def hist_2d_shots(data: pd.DataFrame, x: str, y: str, hue: str, save: bool):
    """
    Histogramme 2D des tirs selon la variable x et y

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe contenant les données des tirs
    x : str
        Nom de la colonne à utiliser en x
    y : str
        Nom de la colonne à utiliser en y
    hue : str
        Nom de la colonne à utiliser pour séparer les points
    save : bool
        Si True, sauvegarde la figure dans le folder `figures`
    """
    # Définir le nom des axes
    x_name, xlabel = rename_feature(x)
    y_name, ylabel = rename_feature(y)
    filename = f'hist_2d_{x_name}_{y_name}.svg'

    plt.figure(figsize=(6, 4))
    sns.jointplot(data=data, x='distance_goal', y='angle_goal')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
    if save:
        plt.savefig(f'../figures/{filename}')