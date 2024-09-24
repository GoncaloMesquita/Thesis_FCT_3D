from Models.Created.UNETR_3D import UNETR
import torch.nn as nn
import torch 
from Models.Created.Unet_3D import Unet_3D
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from Models.Created.FCT_3D_Study_1_256 import  FCT_Mod_1_256


def create_model(model_name, in_channels, out_channels, pretrained, model_checkpoint, patch_size, evaluate, device, patch_vit, maps):

    if model_name == '3DUnet':

        model = Unet_3D(in_channels, out_channels)
        
        if pretrained:
            
            checkpoints = torch.load(model_checkpoint, map_location = device)
            
            for k in ["ll.weight", "ll.bias"]:
                if k in checkpoints:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoints[k]
                    
            model.load_state_dict(checkpoints, strict=False)
            
        if evaluate:

            checkpoints = torch.load(model_checkpoint, map_location = device)
            model.load_state_dict(checkpoints, strict=True)

    elif model_name == 'UNETR':

        model = UNETR(in_channels=in_channels, out_channels=out_channels, img_size=patch_size, save_attn=maps)

        if pretrained:

            checkpoints = torch.load(model_checkpoint, map_location = device)

            for k in [ 'out.conv.conv.bias', 'out.conv.conv.weight',"vit.patch_embedding.cls_token",'encoder1.layer.conv1.conv.weight', 'encoder1.layer.conv3.conv.weight', "vit.patch_embedding.position_embeddings"]:
                    if k in checkpoints:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoints[k]

            model.load_state_dict(checkpoints, strict=False)

            model.out = UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=out_channels)

            model.encoder1.layer.conv1 = get_conv_layer(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    dropout=0.0,
                    act=None,
                    norm=None,
                    conv_only=False,
                )
            
            model.encoder1.layer.conv3 = get_conv_layer(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=16,
                    kernel_size=1,
                    stride=1,
                    dropout=0.0,
                    act=None,
                    norm=None,
                    conv_only=False,
                )
            
        if evaluate:

            checkpoints = torch.load(model_checkpoint, map_location = device)
            model.load_state_dict(checkpoints, strict=True) 
            
    elif model_name == 'FCT':

        model = FCT_Mod_1_256(in_channels, out_channels)
        
        if pretrained:
            
            checkpoints = torch.load(model_checkpoint, map_location = device)
            
            for k in ["ds9.conv3.0.bias", "ds9.conv3.0.weight",
                      'block_2.trans.conv1.0.weight',"block_2.trans.conv1.0.bias",
                    'block_3.trans.conv1.0.weight',"block_3.trans.conv1.0.bias",
                    'block_4.trans.conv1.0.weight',"block_4.trans.conv1.0.bias",
                    'block_5.trans.conv1.0.weight',"block_5.trans.conv1.0.bias",
                    'block_6.trans.conv1.0.weight',"block_6.trans.conv1.0.bias",
                    'block_7.trans.conv1.0.weight',"block_7.trans.conv1.0.bias",
                    'block_8.trans.conv1.0.weight',"block_8.trans.conv1.0.bias",
                    'block_9.trans.conv1.0.weight',"block_9.trans.conv1.0.bias"]:
                
                if k in checkpoints:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoints[k]
                    
            model.load_state_dict(checkpoints, strict=False)
        
            model.ds9.conv3 = nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding="same")

        if evaluate:
            
            checkpoints = torch.load(model_checkpoint, map_location = device)
            
            key_mapping = {
            "ds9.conv3.weight": "ds9.conv3.0.weight",
            "ds9.conv3.bias": "ds9.conv3.0.bias"}

            new_state_dict = {}
            for old_key, value in checkpoints.items():
                new_key = key_mapping.get(old_key, old_key)
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict, strict=True)
    

       
    return model


# from Models.Created.UNETR_3D import UNETR
# import torch.nn as nn
# import torch 
# from Models.Created.Unet_3D import Unet_3D
# from monai.networks.blocks.dynunet_block import UnetOutBlock
# from monai.networks.blocks.dynunet_block import get_conv_layer
# from Models.Created.FCT_3D import FCT
# from Models.Created.FCT_3D_Study import FCT_Mod
# from Models.Created.FCT_3D_Study_1 import FCT_Mod_1
# from Models.Created.FCT_3D_Study_3 import FCT_Mod_3
# from Models.Created.FCT_3D_Study_1_256 import  FCT_Mod_1_256


# def create_model(model_name, in_channels, out_channels, pretrained, model_checkpoint, patch_size, evaluate, device, patch_vit, maps, fold):

#     if model_name == '3DUnet':

#         model = Unet_3D(in_channels, out_channels)
        
#         if pretrained:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             for k in ["ll.weight", "ll.bias"]:
#                 if k in checkpoints:
#                     print(f"Removing key {k} from pretrained checkpoint")
#                     del checkpoints[k]
                    
#             model.load_state_dict(checkpoints, strict=False)
            
#         if evaluate:

#             checkpoints = torch.load(model_checkpoint, map_location = device)
#             model.load_state_dict(checkpoints, strict=True)

#     elif model_name == 'UNETR':

#         model = UNETR(in_channels=in_channels, out_channels=out_channels, img_size=patch_size, save_attn=maps)

#         if pretrained:

#             checkpoints = torch.load(model_checkpoint, map_location = device)

#             for k in [ 'out.conv.conv.bias', 'out.conv.conv.weight',"vit.patch_embedding.cls_token",'encoder1.layer.conv1.conv.weight', 'encoder1.layer.conv3.conv.weight', "vit.patch_embedding.position_embeddings"]:
#                     if k in checkpoints:
#                         print(f"Removing key {k} from pretrained checkpoint")
#                         del checkpoints[k]

#             model.load_state_dict(checkpoints, strict=False)

#             model.out = UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=out_channels)

#             model.encoder1.layer.conv1 = get_conv_layer(
#                     spatial_dims=3,
#                     in_channels=in_channels,
#                     out_channels=16,
#                     kernel_size=3,
#                     stride=1,
#                     dropout=0.0,
#                     act=None,
#                     norm=None,
#                     conv_only=False,
#                 )
            
#             model.encoder1.layer.conv3 = get_conv_layer(
#                     spatial_dims=3,
#                     in_channels=in_channels,
#                     out_channels=16,
#                     kernel_size=1,
#                     stride=1,
#                     dropout=0.0,
#                     act=None,
#                     norm=None,
#                     conv_only=False,
#                 )
            
#         if evaluate:

#             checkpoints = torch.load(model_checkpoint, map_location = device)
#             model.load_state_dict(checkpoints, strict=True)


#     elif model_name == 'FCT':

#         model = FCT(in_channels, out_channels, patch_vit)
        
#         if pretrained:

#             check = torch.load("Models/Pre_traneid_BTCV/Pretrained_FCT_96_last.pth", map_location = device)
#             checkpoints = torch.load(model_checkpoint, map_location = device)

#             for k in ["ds9.conv3.0.bias", "ds9.conv3.0.weight"]:
#                 if k in checkpoints:
#                     print(f"Removing key {k} from pretrained checkpoint")
#                     del checkpoints[k]
                    
#             names_layers = [
#             "trans.conv1.0.weight",
#             ]

#             for block_num in range(1, 10):  # Assuming blocks are numbered from 1 to 9
#                 # Construct the full layer names for the current block
#                 layer_names_to_replace = [f"block_{block_num}." + name for name in names_layers]
#                 for layer_name, tensor in checkpoints.items():
#                     if layer_name in layer_names_to_replace:
#                         print(f"Replace the layer: {layer_name}")
#                         checkpoints[layer_name] = check[layer_name]
                        
#             model.load_state_dict(checkpoints, strict=False)
    
#             model.ds9.conv3 = nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding="same")

#         if evaluate:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             key_mapping = {
#             "ds9.conv3.weight": "ds9.conv3.0.weight",
#             "ds9.conv3.bias": "ds9.conv3.0.bias"}

#             new_state_dict = {}
#             for old_key, value in checkpoints.items():
#                 new_key = key_mapping.get(old_key, old_key)
#                 new_state_dict[new_key] = value

#             model.load_state_dict(new_state_dict, strict=True)
#             # model.load_state_dict(checkpoints, strict=True)
            
#     elif model_name == 'FCT_Mod':

#         model = FCT_Mod(in_channels, out_channels)
        
#         if pretrained:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             for k in ["ds9.conv3.0.bias", "ds9.conv3.0.weight"]:
#                 if k in checkpoints:
#                     print(f"Removing key {k} from pretrained checkpoint")
#                     del checkpoints[k]
                    
#             model.load_state_dict(checkpoints, strict=False)
        
#             model.ds9.conv3 = nn.Conv3d(48, out_channels, kernel_size=1, stride=1, padding="same")

#         if evaluate:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             key_mapping = {
#             "ds9.conv3.weight": "ds9.conv3.0.weight",
#             "ds9.conv3.bias": "ds9.conv3.0.bias"}

#             new_state_dict = {}
#             for old_key, value in checkpoints.items():
#                 new_key = key_mapping.get(old_key, old_key)
#                 new_state_dict[new_key] = value

#             model.load_state_dict(new_state_dict, strict=True)
            
#     elif model_name == 'FCT_Mod_1':

#         model = FCT_Mod_1(in_channels, out_channels)
        
#         if pretrained:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             for k in ["ds9.conv3.0.bias", "ds9.conv3.0.weight"]:
#                 if k in checkpoints:
#                     print(f"Removing key {k} from pretrained checkpoint")
#                     del checkpoints[k]
                    
#             model.load_state_dict(checkpoints, strict=False)
        
#             model.ds9.conv3 = nn.Conv3d(96, out_channels, kernel_size=1, stride=1, padding="same")

#         if evaluate:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             key_mapping = {
#             "ds9.conv3.weight": "ds9.conv3.0.weight",
#             "ds9.conv3.bias": "ds9.conv3.0.bias"}

#             new_state_dict = {}
#             for old_key, value in checkpoints.items():
#                 new_key = key_mapping.get(old_key, old_key)
#                 new_state_dict[new_key] = value

#             model.load_state_dict(new_state_dict, strict=True)
            
#     elif model_name == 'FCT_Mod_1_256':

#         model = FCT_Mod_1_256(in_channels, out_channels)
        
#         if pretrained:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
#             if fold != 0:
            
#                 for k in ["ds9.conv3.0.bias", "ds9.conv3.0.weight",
#                           'block_2.trans.conv1.0.weight',"block_2.trans.conv1.0.bias",
#                         'block_3.trans.conv1.0.weight',"block_3.trans.conv1.0.bias",
#                         'block_4.trans.conv1.0.weight',"block_4.trans.conv1.0.bias",
#                         'block_5.trans.conv1.0.weight',"block_5.trans.conv1.0.bias",
#                         'block_6.trans.conv1.0.weight',"block_6.trans.conv1.0.bias",
#                         'block_7.trans.conv1.0.weight',"block_7.trans.conv1.0.bias",
#                         'block_8.trans.conv1.0.weight',"block_8.trans.conv1.0.bias",
#                         'block_9.trans.conv1.0.weight',"block_9.trans.conv1.0.bias"]:

#                     if k in checkpoints:
#                         print(f"Removing key {k} from pretrained checkpoint")
#                         del checkpoints[k]
                        
#             if fold == 0:
#                 key_mapping = {
#                             "ds9.conv3.weight": "ds9.conv3.0.weight",
#                                 "ds9.conv3.bias": "ds9.conv3.0.bias"}

#                 new_state_dict = {}
#                 for old_key, value in checkpoints.items():
#                     new_key = key_mapping.get(old_key, old_key)
#                     new_state_dict[new_key] = value
                    
#             model.load_state_dict(checkpoints, strict=False)
#             if fold != 0:
#                 model.ds9.conv3 = nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding="same")

#         if evaluate:
            
#             checkpoints = torch.load(model_checkpoint, map_location = device)
            
#             key_mapping = {
#             "ds9.conv3.weight": "ds9.conv3.0.weight",
#             "ds9.conv3.bias": "ds9.conv3.0.bias"}

#             new_state_dict = {}
#             for old_key, value in checkpoints.items():
#                 new_key = key_mapping.get(old_key, old_key)
#                 new_state_dict[new_key] = value

#             model.load_state_dict(new_state_dict, strict=True)
    

       
#     return model











            # if bool(args.parsial_train[i]):
                
            #     print("Warm up...")
            #     # Freeze the whole model
            #     for param in model.parameters():
            #             param.requires_grad = False
                        
            #     # Unfreeze specific layers
            #     unfreeze_layers = [
            #                         'block_2.trans.conv1.0.weight', 'block_2.trans.conv1.0.bias',
            #                         'block_3.trans.conv1.0.weight', 'block_3.trans.conv1.0.bias',
            #                         'block_4.trans.conv1.0.weight', 'block_4.trans.conv1.0.bias',
            #                         'block_5.trans.conv1.0.weight', 'block_5.trans.conv1.0.bias',
            #                         'block_6.trans.conv1.0.weight', 'block_6.trans.conv1.0.bias',
            #                         'block_7.trans.conv1.0.weight', 'block_7.trans.conv1.0.bias',
            #                         'block_8.trans.conv1.0.weight', 'block_8.trans.conv1.0.bias',
            #                         'block_9.trans.conv1.0.weight', 'block_9.trans.conv1.0.bias']

            #     for name, param in model.named_parameters():
            #         if name in unfreeze_layers:
            #             param.requires_grad = True
                
            #     for g in range (0,10):
                            
            #         train_loss = train(model, train_dataloader, optimizer, criterion, device, class_weights_norm)
            #         print(f"Epoch {g}/{10}, Train Loss: {train_loss:.4f}")
                    
            #     for param in model.parameters():
            #         param.requires_grad = True