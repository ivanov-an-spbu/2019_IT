import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import gaussian_kde

#density estimation
def calculate_density(sensor_values, points, Y, density_out_buf):
    for i,label in enumerate(np.unique(Y)): # for each class
        kde = gaussian_kde(sensor_values[Y==label])
        density_out_buf[:, i] = kde.evaluate(points)
    return

#finds density curves intersections
def calculate_intersections(density, points):
    dif_dens = density[:, 0] - density[:, 1]
    sign = dif_dens[1:]*dif_dens[:-1]
    intersections = points[1:][sign<0] # find places of sign changing
                                       # to do: check also == 0 exactly
    return intersections


def main():
    df = pd.read_csv("data_density.csv")
    df = df.drop("sample index", axis=1)

    N=2 # only 2 columns (for example)
    #plots histograms and density distribution
    #for i in range(N):
    #    sensor = f"sensor{i}"
    #    sns.FacetGrid(df[[sensor, "class_label"]], hue="class_label").map(sns.distplot, sensor, bins=50)


    fig, axes = plt.subplots(1, N, figsize=(50,10))
    axes = axes.ravel()

    n_count=2000
    points = np.linspace(0,1,n_count) #values of sensors where density is estimated
    density = np.empty((n_count, 2))  #density for each class

    boundaries = [] # intersections of density curves for each sensor
    X,Y = df.values[:,1:N+1], df.values[:,0]
    for i,x in enumerate(X.T):
        calculate_density(x, points, Y, density)

        axes[i].plot(points, density)
        intersection_points = calculate_intersections(density, points)
        boundaries.append(np.append(intersection_points, 1))
        for x_bound in intersection_points:
            axes[i].axvline(x=x_bound, linestyle='--', color='k')

    for b in boundaries: # found out boundaries for each class
        print(b)

    plt.show()

if __name__ == "__main__":
    main()