import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1.0):
    # y_true_no_background = y_true[:, :, :, 1:]
    # y_pred_no_background = y_pred[:, :, :, 1:]
    # y_true_f = K.flatten(y_true_no_background)
    # y_pred_f = K.flatten(y_pred_no_background)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
