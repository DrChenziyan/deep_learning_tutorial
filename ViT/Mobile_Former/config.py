# ! -*- coding: utf-8 -*-

config_52 = {
    'name': 'Mobile-Former_52M',
    'token': 3,  # num tokens
    'embed_dim': 128,  # embed dim
    'stem_out_dim': 8,
    'bneck': {'bneck_exp': 24, 'bneck_out': 12, 'stride': 2},  # exp out stride
    'block': [
        # stage2
        {'inp': 12, 'exp': 36, 'out': 12, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 12, 'exp': 72, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 72, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 48, 'exp': 288, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 64, 'exp': 384, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
    ],
    "fc_dimension": 1920,
}


config_294 = {
    "name": "Mobile-Former_294M",
    "token": 6,
    "embed_dim": 192,
    "stem_out_dim": 16,
    # stage 1
    "bneck": {"bneck_exp":32, 
                "bneck_out": 16, 
                "stride":1
            },
    "block": [
         # stage2
        {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 1},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 1},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 1},
    ],
    "fc_dimension": 1920,    
}

config_508 = {
    'name': 'Mobile-Former_508M',
    'token': 6,  # tokens and embed_dim
    'embed_dim': 192,
    'stem_out_dim': 24,
    'bneck': {'bneck_exp': 48, 'bneck_out': 24, 'stride': 1},
    'block': [
        {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 1},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 1},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 1},
    ],
    "fc_dimension": 1920,
}