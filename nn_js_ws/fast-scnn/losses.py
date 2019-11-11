from metrics import dice_coef
from keras.losses import categorical_crossentropy, binary_crossentropy


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def cce_dice_loss(y_true, y_pred, cce_weight=0.2):
    return cce_weight * categorical_crossentropy(y_true, y_pred) + (1 - cce_weight) * dice_coef_loss(y_true, y_pred)


def bce_dice_loss(y_true, y_pred, bce_weight=0.2):
    return bce_weight * binary_crossentropy(y_true, y_pred) + (
                1 - bce_weight) * dice_coef_loss(y_true, y_pred)
