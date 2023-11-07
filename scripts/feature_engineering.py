import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def distance_goal(x: float, y: float):
    """
    Calculer la distance entre le tir et le filet

    Parameters
    ----------
    x: float
        Coordonnée en x du tir
    y: float
        Coordonnée en y du tir
    
    Returns
    -------
    distance: array
        Distance entre le tir et le filet pour tous les tirs
    """
    # Les coordonnées du filet ont été définies comme le centre de la ligne des buts.
    # Sachant que la ligne des buts est située à 11 pieds de la bande et que la patinoire
    # a une largeur de 200 pieds, nous pouvons calculer la coordonnées en x en faisant
    # 200/2 - 11 = 89. La coordonnée en y est simplement 0.
    x_goal, y_goal = 89, 0

    # Calculer la distance entre le tir et le filet
    distance = np.sqrt((x_goal - np.abs(x))**2 + (y_goal - np.abs(y))**2)

    return distance


def angle_goal(x: float, y: float):
    """
    Calculer l'angle entre le tir et le filet

    Parameters
    ----------
    x: float
        Coordonnée en x du tir
    y: float
        Coordonnée en y du tir

    Returns
    -------
    angle: array
        Angle entre le tir et le filet pour tous les tirs
    """
    # Les coordonnées du filet ont été définies comme le centre de la ligne des buts.
    # Sachant que la ligne des buts est située à 11 pieds de la bande et que la patinoire
    # a une largeur de 200 pieds, nous pouvons calculer la coordonnées en x en faisant
    # 200/2 - 11 = 89. La coordonnée en y est simplement 0.
    x_goal, y_goal = 89, 0

    # Calculer l'angle entre le tir et le filet
    angle = np.arctan((y_goal - y)/(x_goal - np.abs(x)))
    # Convertir l'angle en degrés
    angle = np.rad2deg(angle)

    return angle

def is_goal(data: pd.DataFrame):
    """
    Encoder la variable booléenne `goalFlag` en variable binaire

    Parameters
    ----------
    data: DataFrame
        Données des tirs
    
    Returns
    -------
    is_goal: array
        Variable binaire indiquant si le tir est un but ou non
    """
    is_goal = LabelEncoder().fit_transform(data['goalFlag'])
    
    return is_goal

def empty_goal(data: pd.DataFrame):
    """
    Encoder la variable booléenne `noGoalie` en variable binaire

    Parameters
    ----------
    data: DataFrame
        Données des tirs

    Returns
    -------
    empty_goal: array
        Variable binaire indiquant si le filet était désert ou non
    """
    empty_goal = LabelEncoder().fit_transform(data['noGoalie'])

    return empty_goal

def create_features1(data: pd.DataFrame, pattern: str, outname: str):
    """
    Fonction pour créer les nouvelles caractéristiques à partir des données spécifiées
    tel que demandé dans la section Ingénierie des caractéristiques I du Milestone 2 et 
    les sauvegarder dans un fichier csv.

    Parameters
    ----------
    data: DataFrame
        Données nettoyées provenant du Milestone 1
    pattern: str
        Regex pour sélectionner certaines données. Si None, toutes les données dans data
        seront utilisées.
    outname: str
        Nom du fichier csv dans lequel les nouvelles caractéristiques seront sauvegardées
    """
    # Instancier un nouveau DataFrame
    new_data = pd.DataFrame()

    # Isoler les données des saisons régulières de 2016-2017 à 2019-2020
    pattern = re.compile(pattern)
    if pattern is not None:
        data = data[data['gameId'].astype(str).str.match(pattern)]
        data.reset_index(inplace=True)

    # Créer la variable distance_goal
    new_data['distance_goal'] = distance_goal(data['coord_x'], data['coord_y'])
    # Créer la variable angle_goal
    new_data['angle_goal'] = angle_goal(data['coord_x'], data['coord_y'])
    # Créer la variable is_goal
    new_data['is_goal'] = is_goal(data)
    # Créer la variable empty_goal
    new_data['empty_goal'] = empty_goal(data)

    # Enlever les lignes avec des valeurs manquantes
    new_data.dropna(inplace=True)

    new_data.to_csv(f'../data/derivatives/{outname}', index=False)
    
    

def create_features2(data: pd.DataFrame, pattern: str):
    """
    Fonction pour ajouter de nouvelles caractéristiques aux données existantes et
    sauvegarder le résultat dans le même fichier CSV.

    Parameters
    ----------
    data: DataFrame
        Données nettoyées provenant du Milestone 1.
    pattern: str
        Regex pour sélectionner certaines données. Si None, toutes les données dans data
        seront utilisées.
    """

    data = data.copy()

    # Ajout des nouvelles caractéristiques au DataFrame
    data['game_seconds'] = data['prdTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    data['coord_x'].fillna(0, inplace=True)
    data['coord_y'].fillna(0, inplace=True)
    data['shot_distance'] = distance_goal(data['coord_x'], data['coord_y'])
    data['shot_angle'] = angle_goal(data['coord_x'], data['coord_y'])
    data['rebond'] = (data['last_event_type'] == 'SHOT')
    previous_shot_angle = angle_goal(data['last_event_x'], data['last_event_y'])
    data['changement_angle_tir'] = np.where(data['rebond'], previous_shot_angle + data['shot_angle'], 0)
    data['vitesse'] = data['distance_from_last_event'] / data['time_since_last_event']
    
    # Gestion des cas où time_since_last_event est zéro pour éviter une division par zéro
    data['vitesse'].replace(np.inf, 0, inplace=True)
    data['vitesse'].fillna(0, inplace=True)

    data.dropna(inplace=True)
    return data