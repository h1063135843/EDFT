import numpy as np
import matplotlib.pyplot as plt
import os


def norm(mat):
    confusion_mat_N = mat.copy()
    confusion_mat_N = confusion_mat_N.astype('float') / confusion_mat_N.sum(
        axis=1)[:, np.newaxis]
    return confusion_mat_N


def show_conf_mat(data, classes_name, decimal=False):
    num = len(data)

    fig, ax = plt.subplots(1, num)
    ax = ax.flatten() if num > 1 else [ax]
    cmap = plt.cm.get_cmap('BuPu')

    xlocations = np.array(range(len(classes_name)))
    for i in range(num):
        majorFormatter = plt.FormatStrFormatter('%1.1f')
        for j in range(data[i].shape[0]):
            for k in range(data[i].shape[0]):
                if decimal:
                    txt = '%.3f' % data[i][k, j]
                else:
                    txt = format(data[i][k, j], ',').replace("-", u"\u2212")
                ax[i].annotate(
                    txt,
                    xy=(j, k),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='red')
        im = ax[i].imshow(data[i], cmap=cmap)
        ax[i].xaxis.set_ticks_position('top')
        ax[i].set_xticks(xlocations)
        ax[i].set_xticklabels(classes_name, rotation=60)
        ax[i].set_yticks(xlocations)
        ax[i].set_yticklabels(classes_name)

    fig.colorbar(im, ax=ax, fraction=0.045 / num, pad=0.05)
    plt.show()


def visualize_three_mats():
    segformer = np.array([[21075636, 447976, 849189, 183406, 92624, 22837],
                          [799558, 20719389, 371593, 38434, 15779, 4655],
                          [649032, 203086, 14621448, 1887284, 4500, 5435],
                          [149387, 45380, 1501524, 16811874, 1009, 2924],
                          [106488, 16553, 5764, 1426, 664722, 480],
                          [183227, 148438, 5197, 910, 19850, 334929]])

    two_stream = np.array([[21237527, 422226, 774217, 171876, 57830, 7992],
                           [548330, 21131585, 212216, 44021, 9822, 3434],
                           [669343, 190216, 14381328, 2127453, 2389, 56],
                           [164992, 27690, 1271714, 17045566, 533, 1603],
                           [91922, 16639, 3882, 1876, 680915, 199],
                           [243447, 130210, 8483, 1004, 7799, 301608]])

    edft = np.array([[21213732, 455588, 748926, 178289, 65023, 10110],
                     [497497, 21232005, 164006, 43077, 11223, 1600],
                     [730607, 227176, 14492509, 1912634, 3506, 4353],
                     [172472, 37015, 1327723, 16971992, 861, 2035],
                     [78748, 11804, 2126, 1464, 700958, 333],
                     [200536, 187836, 3441, 592, 10572, 289574]])
    data = [segformer, two_stream, edft]
    norm_data = [norm(x) for x in data]
    dif_data = [data[1] - data[0], data[2] - data[0]]
    classes_name = [
        'imp surf', 'building', 'low_veg', 'tree', 'car', 'clutter'
    ]

    show_conf_mat(norm_data, classes_name, True)
    show_conf_mat([data[2]], classes_name)
    show_conf_mat(dif_data, classes_name)


def visualize_scanet2d():
    data = [[
        111941090, 39305, 219221, 34627, 65072, 1441, 13197, 677, 11161,
        304306, 1156923, 0, 228673, 0, 0, 1081468, 0, 31, 0, 1246174
    ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                550071, 465, 34818, 1493281, 0, 271, 6995, 516, 145810, 0, 556,
                0, 1699002, 0, 23012, 13599, 0, 2, 30253, 5266141
            ],
            [
                670892, 6220, 875129, 7037864, 6124317, 266391, 92, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 92247, 4318939
            ],
            [
                1766669, 22206, 0, 202, 326, 895717, 2097, 0, 0, 0, 269145, 0,
                79452, 0, 1032, 3072, 4105, 2798, 2226, 5149185
            ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                95452, 3577, 0, 0, 0, 0, 4927, 0, 609663, 0, 0, 0, 10450393,
                9867, 299272, 0, 0, 0, 0, 901817
            ],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 157220, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                409305, 1314663, 0, 0, 0, 116520, 8421975, 1748219, 84, 6353,
                10916, 0, 484641, 0, 0, 637074, 0, 581, 662449, 10336002
            ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                2703124, 11146, 0, 0, 0, 102125, 13883, 33666, 0, 748, 61296,
                0, 565373, 0, 740199, 135, 0, 4, 1492, 5467908
            ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    data = np.array(data)
    
    norm_data=[norm(data)]
    data=[data]
    # print(data.shape)

    classes_name = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
        'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
        'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
        'otherfurniture'
    ]

    show_conf_mat(norm_data, classes_name, True)
    # show_conf_mat(data, classes_name)


visualize_scanet2d()
# visualize_three_mats()
