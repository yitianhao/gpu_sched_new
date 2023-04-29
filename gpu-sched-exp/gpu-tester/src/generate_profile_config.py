import copy
import os

from utils import write_json_file

MODEL_NAMES = {
    "detection": [
        "fasterrcnn_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
        "maskrcnn_resnet50_fpn_v2",
        "retinanet_resnet50_fpn",
        "fcos_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ],
    "segmentation": [
        "deeplabv3_mobilenet_v3_large",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "fcn_resnet50",
        "fcn_resnet101",
        "lraspp_mobilenet_v3_large",
    ],
    "classification": [
        "alexnet",
        "convnext_base",
        "convnext_large",
        "convnext_small",
        "convnext_tiny",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_v2_l",
        "efficientnet_v2_m",
        "efficientnet_v2_s",
        "googlenet",
        "inception_v3",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "regnet_x_16gf",
        "regnet_x_1_6gf",
        "regnet_x_32gf",
        "regnet_x_3_2gf",
        "regnet_x_400mf",
        "regnet_x_800mf",
        "regnet_x_8gf",
        "regnet_y_128gf",
        "regnet_y_16gf",
        "regnet_y_1_6gf",
        "regnet_y_32gf",
        "regnet_y_3_2gf",
        "regnet_y_400mf",
        "regnet_y_800mf",
        "regnet_y_8gf",
        "resnet101",
        "resnet152",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "resnext50_32x4d",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
        "squeezenet1_0",
        "squeezenet1_1",
        "swin_b",
        "swin_s",
        "swin_t",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "vit_b_16",
        "vit_b_32",
        "vit_h_14",
        "vit_l_16",
        "vit_l_32",
        "wide_resnet101_2",
        "wide_resnet50_2",
    ],
}


MODEL_WEIGHTS = {
    "detection": [
        "FasterRCNN_ResNet50_FPN_Weights",
        "KeypointRCNN_ResNet50_FPN_Weights",
        "MaskRCNN_ResNet50_FPN_V2_Weights",
        "RetinaNet_ResNet50_FPN_Weights",
        "FCOS_ResNet50_FPN_Weights",
        "SSD300_VGG16_Weights",
        "SSDLite320_MobileNet_V3_Large_Weights",
    ],
    "segmentation": [
        "DeepLabV3_MobileNet_V3_Large_Weights",
        "DeepLabV3_ResNet50_Weights",
        "DeepLabV3_ResNet101_Weights",
        "FCN_ResNet50_Weights",
        "FCN_ResNet101_Weights",
        "LRASPP_MobileNet_V3_Large_Weights",
    ],
    "classification": [
        "AlexNet_Weights",
        "ConvNeXt_Base_Weights",
        "ConvNeXt_Large_Weights",
        "ConvNeXt_Small_Weights",
        "ConvNeXt_Tiny_Weights",
        "DenseNet121_Weights",
        "DenseNet161_Weights",
        "DenseNet169_Weights",
        "DenseNet201_Weights",
        "EfficientNet_B0_Weights",
        "EfficientNet_B1_Weights",
        "EfficientNet_B2_Weights",
        "EfficientNet_B3_Weights",
        "EfficientNet_B4_Weights",
        "EfficientNet_B5_Weights",
        "EfficientNet_B6_Weights",
        "EfficientNet_B7_Weights",
        "EfficientNet_V2_L_Weights",
        "EfficientNet_V2_M_Weights",
        "EfficientNet_V2_S_Weights",
        "GoogLeNet_Weights",
        "Inception_V3_Weights",
        "MNASNet0_5_Weights",
        "MNASNet0_75_Weights",
        "MNASNet1_0_Weights",
        "MNASNet1_3_Weights",
        "MobileNet_V2_Weights",
        "MobileNet_V3_Large_Weights",
        "MobileNet_V3_Small_Weights",
        "RegNet_X_16GF_Weights",
        "RegNet_X_1_6GF_Weights",
        "RegNet_X_32GF_Weights",
        "RegNet_X_3_2GF_Weights",
        "RegNet_X_400MF_Weights",
        "RegNet_X_800MF_Weights",
        "RegNet_X_8GF_Weights",
        "RegNet_Y_128GF_Weights",
        "RegNet_Y_16GF_Weights",
        "RegNet_Y_1_6GF_Weights",
        "RegNet_Y_32GF_Weights",
        "RegNet_Y_3_2GF_Weights",
        "RegNet_Y_400MF_Weights",
        "RegNet_Y_800MF_Weights",
        "RegNet_Y_8GF_Weights",
        "ResNet101_Weights",
        "ResNet152_Weights",
        "ResNet18_Weights",
        "ResNet34_Weights",
        "ResNet50_Weights",
        "ResNeXt101_32X8D_Weights",
        "ResNeXt101_64X4D_Weights",
        "ResNeXt50_32X4D_Weights",
        "ShuffleNet_V2_X0_5_Weights",
        "ShuffleNet_V2_X1_0_Weights",
        "ShuffleNet_V2_X1_5_Weights",
        "ShuffleNet_V2_X2_0_Weights",
        "SqueezeNet1_0_Weights",
        "SqueezeNet1_1_Weights",
        "Swin_B_Weights",
        "Swin_S_Weights",
        "Swin_T_Weights",
        "VGG11_Weights",
        "VGG11_BN_Weights",
        "VGG13_Weights",
        "VGG13_BN_Weights",
        "VGG16_Weights",
        "VGG16_BN_Weights",
        "VGG19_Weights",
        "VGG19_BN_Weights",
        "ViT_B_16_Weights",
        "ViT_B_32_Weights",
        "ViT_H_14_Weights",
        "ViT_L_16_Weights",
        "ViT_L_32_Weights",
        "Wide_ResNet101_2_Weights",
        "Wide_ResNet50_2_Weights",
    ],
}

TEMPLATE = {
    "model_name": "fasterrcnn_resnet50_fpn",
    "model_weight": "FasterRCNN_ResNet50_FPN_Weights",
    "sleep_time": 0,
    "input_file_path": "../data-set/rene/0000000099.png",
    "output_file_path": "./profiles/detection/",
    "output_file_name": "fasterrcnn_resnet50_fpn_720x1280_sleep_time_0",
    "priority": 0,
    "resize": False,
    "resize_size": [
        720,
        1280
    ],
    "control": {
        "control": False,
        "controlsync": False,
        "controlEvent": False,
        "queue_limit": {
            "sync": 1,
            "event_group": 2
        }
    },
    "batch_size": 1
}


def main():
    # workload = 'detection'
    workload = 'classification'
    for model_name, model_weight in zip(
        MODEL_NAMES[workload], MODEL_WEIGHTS[workload]):
        # print(model_name, model_weight)
        config = copy.deepcopy(TEMPLATE)
        config["model_name"] = model_name
        config["model_weight"] = model_weight
        w, h = TEMPLATE["resize_size"]
        # w, h = 1440, 2560
        config["resize_size"] = [w, h]
        config["resize"] = True
        sleep_time = 0  # template['sleep_time']
        config["sleep_time"] = sleep_time
        batch = 1
        config['batch_size'] = batch
        name = f"{model_name}_{w}x{h}_sleep_time_{sleep_time}_batch_{batch}"
        config["output_file_name"] = name
        config["output_file_path"] = f"./nsys_profiles/{workload}/{name}"

        folder = f"./nsys_profile_configs/{workload}/"
        os.makedirs(folder, exist_ok=True)

        write_json_file(os.path.join(folder, f"{name}.json"), config)


if __name__ == '__main__':
    main()
