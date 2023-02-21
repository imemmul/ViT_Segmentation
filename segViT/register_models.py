from mmseg.models.decode_heads.atm_head import ATMHead
from ViT_Segmentation.utils.dataset_parser import EddyDatasetREGISTER


def register_modules():
    try:
        ATMHead() # initiliaze and add register to mmseg module
    except:
        pass
    try:
        EddyDatasetREGISTER()
    except:
        pass


if __name__ == '__main__':
    register_modules()