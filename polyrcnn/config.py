from detectron2.config import CfgNode as CN

def add_polyrcnn_config(cfg):
    """
    Add config for PolyRCNN.
    """
    cfg.MODEL.PolyRCNN = CN()
    cfg.MODEL.PolyRCNN.NUM_CLASSES = 1
    cfg.MODEL.PolyRCNN.NUM_PROPOSALS = 100  # the number of proposal polygons per image
    cfg.MODEL.PolyRCNN.NUM_CORNERS = 96  # the number of proposal vertices per polygon
    # RCNN Head.
    cfg.MODEL.PolyRCNN.NHEADS = 8  # multi-head self-attention
    cfg.MODEL.PolyRCNN.DROPOUT = 0.0
    cfg.MODEL.PolyRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.PolyRCNN.ACTIVATION = 'relu'
    cfg.MODEL.PolyRCNN.HIDDEN_DIM = 256
    cfg.MODEL.PolyRCNN.NUM_CLS = 1
    cfg.MODEL.PolyRCNN.NUM_REG = 3
    cfg.MODEL.PolyRCNN.NUM_COR = 3
    cfg.MODEL.PolyRCNN.NUM_POL = 3
    cfg.MODEL.PolyRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.PolyRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.PolyRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.PolyRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.PolyRCNN.GIOU_WEIGHT = 2.0
    cfg.MODEL.PolyRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.PolyRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.PolyRCNN.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.PolyRCNN.USE_FOCAL = True
    cfg.MODEL.PolyRCNN.ALPHA = 0.25
    cfg.MODEL.PolyRCNN.GAMMA = 2.0
    cfg.MODEL.PolyRCNN.PRIOR_PROB = 0.01

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])

    # Initial Proposal Polygon.
    # 96 uniform sampled vertices from the contour of a 0.5*0.5 rectangle in the center of a 1*1 rectangle.
    cfg.INIT_POLYGON = [0.2500, 0.2500, 0.2500, 0.2679, 0.2500, 0.2857, 0.2500, 0.3125, 0.2500,
                        0.3304, 0.2500, 0.3482, 0.2500, 0.3661, 0.2500, 0.3929, 0.2500, 0.4107,
                        0.2500, 0.4286, 0.2500, 0.4464, 0.2500, 0.4732, 0.2500, 0.4911, 0.2500,
                        0.5089, 0.2500, 0.5268, 0.2500, 0.5446, 0.2500, 0.5714, 0.2500, 0.5893,
                        0.2500, 0.6071, 0.2500, 0.6250, 0.2500, 0.6518, 0.2500, 0.6696, 0.2500,
                        0.6875, 0.2500, 0.7054, 0.2500, 0.7321, 0.2500, 0.7500, 0.2696, 0.7500,
                        0.2892, 0.7500, 0.3088, 0.7500, 0.3382, 0.7500, 0.3578, 0.7500, 0.3775,
                        0.7500, 0.3971, 0.7500, 0.4265, 0.7500, 0.4461, 0.7500, 0.4657, 0.7500,
                        0.4853, 0.7500, 0.5049, 0.7500, 0.5343, 0.7500, 0.5539, 0.7500, 0.5735,
                        0.7500, 0.5931, 0.7500, 0.6225, 0.7500, 0.6422, 0.7500, 0.6618, 0.7500,
                        0.6814, 0.7500, 0.7108, 0.7500, 0.7304, 0.7500, 0.7500, 0.7500, 0.7500,
                        0.7321, 0.7500, 0.7143, 0.7500, 0.6875, 0.7500, 0.6696, 0.7500, 0.6518,
                        0.7500, 0.6339, 0.7500, 0.6071, 0.7500, 0.5893, 0.7500, 0.5714, 0.7500,
                        0.5536, 0.7500, 0.5268, 0.7500, 0.5089, 0.7500, 0.4911, 0.7500, 0.4732,
                        0.7500, 0.4554, 0.7500, 0.4286, 0.7500, 0.4107, 0.7500, 0.3929, 0.7500,
                        0.3750, 0.7500, 0.3482, 0.7500, 0.3304, 0.7500, 0.3125, 0.7500, 0.2946,
                        0.7500, 0.2768, 0.7500, 0.2500, 0.7304, 0.2500, 0.7108, 0.2500, 0.6912,
                        0.2500, 0.6618, 0.2500, 0.6422, 0.2500, 0.6225, 0.2500, 0.6029, 0.2500,
                        0.5735, 0.2500, 0.5539, 0.2500, 0.5343, 0.2500, 0.5147, 0.2500, 0.4951,
                        0.2500, 0.4657, 0.2500, 0.4461, 0.2500, 0.4265, 0.2500, 0.4069, 0.2500,
                        0.3775, 0.2500, 0.3578, 0.2500, 0.3382, 0.2500, 0.3186, 0.2500, 0.2892,
                        0.2500, 0.2696, 0.2500]
    # 60 uniform sampled vertices from the contour of a 0.5*0.5 rectangle in the center of a 1*1 rectangle.
    # cfg.INIT_POLYGON = [0.2475, 0.2475, 0.2475, 0.2809, 0.2475, 0.3144, 0.2475, 0.3478, 0.2475, 0.3813, 0.2475,
    #                     0.4147, 0.2475, 0.4482, 0.2475, 0.4816, 0.2475, 0.5151, 0.2475, 0.5485, 0.2475, 0.5819,
    #                     0.2475, 0.6154, 0.2475, 0.6488, 0.2475, 0.6823, 0.2475, 0.7157, 0.2475, 0.7492, 0.2809,
    #                     0.7492, 0.3144, 0.7492, 0.3478, 0.7492, 0.3813, 0.7492, 0.4147, 0.7492, 0.4482, 0.7492,
    #                     0.4816, 0.7492, 0.5151, 0.7492, 0.5485, 0.7492, 0.5819, 0.7492, 0.6154, 0.7492, 0.6488,
    #                     0.7492, 0.6823, 0.7492, 0.7157, 0.7492, 0.7492, 0.7492, 0.7492, 0.7157, 0.7492, 0.6823,
    #                     0.7492, 0.6488, 0.7492, 0.6154, 0.7492, 0.5819, 0.7492, 0.5485, 0.7492, 0.5151, 0.7492,
    #                     0.4816, 0.7492, 0.4482, 0.7492, 0.4147, 0.7492, 0.3813, 0.7492, 0.3478, 0.7492, 0.3144,
    #                     0.7492, 0.2809, 0.7492, 0.2475, 0.7157, 0.2475, 0.6823, 0.2475, 0.6488, 0.2475, 0.6154,
    #                     0.2475, 0.5819, 0.2475, 0.5485, 0.2475, 0.5151, 0.2475, 0.4816, 0.2475, 0.4482, 0.2475,
    #                     0.4147, 0.2475, 0.3813, 0.2475, 0.3478, 0.2475, 0.3144, 0.2475, 0.2809, 0.2475]