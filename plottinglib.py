
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix,  precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import (OneHotEncoder, label_binarize, LabelEncoder )



def Scartter_plot( X, Y, n_samples, s_size , colors, grid = False, formato = 'png'):
   
    # Y : class list
    # X : numpy array (n, k)

    fig, ax = plt.subplots(1, 1, figsize= (10, 10))
    samples, dim = X.shape

    le = LabelEncoder()
    le.fit(Y)
    clases = le.classes_

    # only first 2 dimmension

    dic_f = {'x1' : X[:,0],'x2' : X[:,1], 'target' : Y}
    df = pd.DataFrame(dic_f) 
    
    
    for i, clase in enumerate(clases):
        
        df_aux = df.loc[df.target == clase].values

        x = df_aux[:n_samples,0]
        y = df_aux[:n_samples,1]

        if clase == 'SLSN':
            ax.scatter(x, y,  label=clase, alpha=0.9, s= 50, color = colors[i])

        elif clase == 'SNIbc':
            ax.scatter(x, y,  label=clase, alpha=0.9, s= 20, color = colors[i])
        else:
            ax.scatter(x, y,  label=clase, alpha=0.5, s= s_size,  color =colors[i] )


    ax.legend( fontsize= 12)
    ax.grid(grid)

    ax.set_facecolor("white")
    #plt.savefig( name_fig + '.' + formato, format = formato)
    return fig, ax




def custom_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          plot_size = None,
                          cmap=plt.cm.Blues):

    """ 
    # y_true : [y_true_sim_1, y_true_sim_2, .... y_pred_sim_k]    
    # y_pred : [y_pred_sim_1, y_pred_sim_2, .... y_pred_sim_k]
    # classes : ['label_1', label_2, ...] len(classes) = n_classes
    # title : -----
    # plot_size : (h, w)
    # cmap : plt.cm.Blues,  [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    """
                              
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    n_sim = len(y_true)  

    cnt = 0
    cm_list = []

    for y_true_idx, y_pred_idx in zip(y_true, y_pred):
        if cnt == 0:
            cm = confusion_matrix(y_true_idx, y_pred_idx)
        else:
            cm += confusion_matrix(y_true_idx, y_pred_idx)

        cm_list.append(confusion_matrix(y_true_idx, y_pred_idx))

        cnt += 1

    cm = np.mean(cm_list, axis = 0)
    cm_std = np.std(cm_list, axis = 0)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true_idx, y_pred_idx)]

    if normalize:

        normalize_factor = cm.sum(axis=1)[:, np.newaxis]

        cm = cm.astype('float') / normalize_factor
        cm_std = cm_std.astype('float') / normalize_factor

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

        cm = np.floor(cm).astype('int')

    fig, ax = plt.subplots(figsize=plot_size,  dpi=80)
    
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin= 0, vmax=1)
    
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            if i != j :
                ax.text(j, i, format(cm[i, j], fmt) ,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            elif i == j:

                ax.text(j, i, format(cm[i, j], fmt) + r'$\pm$' + format(cm_std[i,j], '.2f') ,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                
             
    fig.tight_layout()
    
    return ax, fig




def feature_histogram(fig, ax, x, param_idx, name, bins):

    colors = ['g', 'r']
    param = [r'$A$', r'$\beta$', r'$\tau_{rise}$', r'$\tau_{fall}$', r'$\gamma$', r'$t_{0}$']
    #plt.style.use('bmh')
    
    for i in range(len(x)):
        data = x[i]
        h, edges = np.histogram(data, bins= np.linspace( min(data), max(data), bins )) 
        ax.stairs(h, edges,  color = colors[i], label= '{}'.format(colors[i]), lw = 3)
        ax.set_title(param[param_idx], fontsize = 20)
    ax.legend()

    return fig, ax

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if (isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid)):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = (gridspec.GridSpecFromSubplotSpec(r + 1, r + 1,
                        subplot_spec=self.subplot))

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())