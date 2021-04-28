import matplotlib.pyplot as plt
import numpy as np


def compute_slope_intercept(x, y):
    """ computes the slope and y intersept of the passed in x and y variables
    Args:
        x: (list) x variables
        y: (list) y variables
    Returns:
        m: (float) slope
        b: (float) y intersept
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y) 
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
        / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => y - mx
    b = mean_y - m * mean_x
    return m, b 

def bar_chart(x, y, ylabels, xlabels, titles):
    """ Plots a larger box plot given variables
    Args:
        x: (list) x values
        y: (list) y values
        ylabels: (String) ylabel
        xlabels: (String) xlabel
        titles: (String) title
    """
    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111)
    ax.bar(x, y)
    x_tick_labels = x
    plt.xticks(x, x_tick_labels, rotation=75, horizontalalignment="right")
    plt.ylabel(ylabels)
    plt.xlabel(xlabels)
    plt.title(titles)
    plt.show()
    
def bar_chart2(x, y, ylabels, xlabels, titles):
    """ Plots a smaller box plot given variables
    Args:
        x: (list) x values
        y: (list) y values
        ylabels: (String) ylabel
        xlabels: (String) xlabel
        titles: (String) title
    """
    plt.figure()
    plt.bar(x, y)
    x_tick_labels = x
    plt.xticks(x, x_tick_labels, rotation=75, horizontalalignment="right")
    plt.ylabel(ylabels)
    plt.xlabel(xlabels)
    plt.title(titles)
    plt.show()
    
def pie_chart(x, y, titles):
    
    """ Plots a pie chart
    Args:
        x: (list) x values
        y: (list) y values
    """
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title(titles)
    plt.show()
    
def histogram(data, ylabels, xlabels, titles, bins1):
    """ Plots a histogram
    Args:
        data: (list) data for generating histogram
        y: (list) y values
        ylabels: (String) ylabel
        xlabels: (String) xlabel
        titles: (String) title
        bins1: (int) #of bins wanted
    """
    plt.figure()
    plt.hist(data, align='mid')
    plt.ylabel(ylabels)
    plt.xlabel(xlabels)
    plt.title(titles)
    plt.show()
    
def scatter_plot(x,y,xlabels,ylabels,titles):
    
    """ Plots a scatter plot, line of best fit, correlation coeeficient and covariance
    Args:
        x: (list) x values
        y: (list) y values
        ylabels: (String) ylabel
        xlabels: (String) xlabel
        titles: (String) title
    """
    
    #creating scatter plot
    plt.figure()
    plt.plot(x, y, "b.")
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(titles)
    
    #computing and plotting line of best fit
    m, b = compute_slope_intercept(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
    
    #computing and pltting correlation coefficient and covarience
    r = np.corrcoef(x, y)
    cov = np.cov(x,y)
    cov_str = "Cov: " + str(round(cov[0][1],3))
    r_str = "Corr: " + str(round(r[0][1],3))
    plt.annotate(r_str, xy=(0.9, 0.9), xycoords="axes fraction", 
        horizontalalignment="center", color="blue")
    plt.annotate(cov_str, xy=(0.9, 0.8), xycoords="axes fraction", 
        horizontalalignment="center", color="blue")
    plt.show()

def movie_scatter_plot(x,y,xlabels,ylabels,titles):
    
    """ Plots a scatter plot and line of best fit for the movie data
    Args:
        x: (list) x values
        y: (list) y values
        ylabels: (String) ylabel
        xlabels: (String) xlabel
        titles: (String) title
    """
    
    #creating scatter plot
    plt.figure()
    plt.plot(x, y, "b.")
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(titles)
    
    #computing and plotting line of best fit
    m, b = compute_slope_intercept(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
  
    #computing and pltting correlation coefficient and covarience
    r = np.corrcoef(x, y)
    cov = np.cov(x,y)
    cov_str = "Cov: " + str(round(cov[0][1],3))
    r_str = "Corr: " + str(round(r[0][1],3))
    plt.annotate(r_str, xy=(0.9, 0.2), xycoords="axes fraction", 
        horizontalalignment="center", color="blue")
    plt.annotate(cov_str, xy=(0.9, 0.1), xycoords="axes fraction", 
        horizontalalignment="center", color="blue")
    plt.show()   

def box_plot(distributions, x_labels, x_label, y_label, title):
    '''
    Creates a box plot diagram (can include multiply boxes on a single plot).
    Args: distributions (1D or nested list of data), x_labels (1D list of strings to label 
    each box), x_label (string: x-axis label), y_label (y_axis label), title (string)
    Returns: n/a (displays an in-line chart)
    '''
    plt.figure(figsize=(7, 5))
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(x_labels) + 1)), x_labels, rotation=90)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()