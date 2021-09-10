# ! -*- coding: utf-8 -*-
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