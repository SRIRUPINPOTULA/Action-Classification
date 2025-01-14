import matplotlib.pyplot as plt
import numpy as np

# import SQL library
import sqlite3


def get_video_features(feature, video_name):

    connection = sqlite3.connect('../database/Phase_2.db')
    c = connection.cursor()

    get_query = f"SELECT {feature} FROM data WHERE Video_Name = '{video_name}';"

    c.execute(get_query)
    rows = c.fetchall()

    connection.close()

    cleaned_str = rows[0][0].strip("[]")
    number_list = list(map(int, cleaned_str.split()))

    data = np.array(number_list).reshape(12, 40)
    return data

def Visualize_HoG_HoF(feature, video_name):

    data = get_video_features(feature, video_name)

    tau_values = [2, 4]
    sigma_values = [4, 8, 16, 32, 64, 128]

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))

    axes = axes.flatten()

    i = 0
    for t in tau_values:
        for s in sigma_values:
            axes[i].hist(data[i], bins=np.arange(41) - 0.5, edgecolor='black')
            axes[i].set_title(f'Histogram for tau={t} sigma={s}')
            i += 1

    plt.tight_layout()
    
    plt.show()
