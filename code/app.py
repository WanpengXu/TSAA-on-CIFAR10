import os
import gradio as gr

import torch
from torch.backends import cudnn

import cv2

from utils import set_seeds, get_device, _normalize, _denormalize, get_run_time

from generators import GeneratorResnet

from pytorch_cifar.models import *

set_seeds(42)


def normalize(t):
    # CIFAR-10 test dataset
    mean = (0.4940, 0.4850, 0.4504)
    std = (0.2467, 0.2429, 0.2616)
    return _normalize(t, mean, std)


def denormalize(t):
    mean = (0.4940, 0.4850, 0.4504)
    std = (0.2467, 0.2429, 0.2616)
    return _denormalize(t, mean, std)


label_to_name = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车")


def generate_and_eval_adv_image(original_image, target_class, eps, generator_model_name, target_model_name,
                                generate_adv_images_dir):
    generator_weight_path = os.path.join(os.path.dirname(__file__), '..', 'generator_weights', generator_model_name,
                                         target_class, f'{eps}.pth')
    target_model_weight = os.path.join(os.path.dirname(__file__), '..', 'classifier_weights',
                                       f'{target_model_name}.pth')

    # GPU
    device = get_device()

    netG = GeneratorResnet(eps=eps / 255., evaluate=True, data_dim='high')
    netG.load_state_dict(torch.load(generator_weight_path, map_location=torch.device(device)))
    netG = netG.to(device)
    if device != 'cpu':
        netG = torch.nn.DataParallel(netG)
    netG.eval()

    if target_model_name[:3] == "VGG":
        target_model = VGG(target_model_name)
    else:
        target_model = eval(target_model_name + "()")

    target_model = target_model.to(device)
    if device != 'cpu':
        target_model = torch.nn.DataParallel(target_model)
        cudnn.benchmark = True
    checkpoint = torch.load(target_model_weight, map_location=torch.device(device))

    target_model.load_state_dict(checkpoint['net_state_dict'])

    target_model.eval()
    print(f"target_model_acc={checkpoint['acc']}")

    img = original_image
    # label = torch.tensor(label)
    img = img / 255
    img = torch.from_numpy(img.copy())
    img = img.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    print(img.shape)

    # img, label = img.to(device), label.to(device)
    img = img.to(device)

    clean_out = target_model(normalize(img.clone().detach()))
    predict_clean_class = clean_out.argmax(dim=-1)
    time, (adv, _, adv_0, adv_00) = get_run_time(netG, img)
    adv_out = target_model(normalize(adv.clone().detach()))
    predict_adv_class = adv_out.argmax(dim=-1)

    # 打印对抗样本
    adv_image_path = os.path.join(generate_adv_images_dir, f'{target_model_name}.png')
    adv_image = adv.clone().detach().squeeze(0).permute(1, 2, 0).cpu() * 255
    print(adv_image.shape)
    cv2.imwrite(adv_image_path, cv2.cvtColor(adv_image.numpy(), cv2.COLOR_BGR2RGB))

    norm = torch.norm(adv_0.clone().detach(), 0)

    print('L0 norm:', norm)
    print('time:', time)

    # 返回生成图片的路径和欺骗状态
    return adv_image_path, predict_adv_class.item()


# 后续可以把生成和测试解耦试试
# def eval_adv_image(adv_image_path, classifier_model):


def generate_adv_images(original_image, target, eps, generator_model, classifier_models):
    generate_adv_images_dir = os.path.join(os.path.dirname(__file__), '..', 'generate_adv_images_dir')
    os.makedirs(generate_adv_images_dir, exist_ok=True)
    adv_images_and_fool_state = []
    # print(type(original_image))
    # <class 'numpy.ndarray'>
    # print(image_label)
    # image_label = image_label.split(',')[0]
    target = target.split(',')[0]
    for classifier_model in classifier_models:
        adv_image_path, state = generate_and_eval_adv_image(original_image, target, eps, generator_model,
                                                            classifier_model, generate_adv_images_dir)
        adv_images_and_fool_state.append((adv_image_path, f'{classifier_model}: {label_to_name[state]}'))

    print(adv_images_and_fool_state)
    return adv_images_and_fool_state


with gr.Blocks() as demo:
    # block是左右的Tab
    gr.Markdown("# 模式识别课程设计：对抗样本生成系统")
    # 这里面的element按Column排列
    with gr.Column(variant="panel"):
        # 这里面的element按Row排列
        with gr.Row(variant="compact"):
            original_image = gr.components.Image(label="原始图像")
            with gr.Column(variant="compact"):
                # image_label = gr.Dropdown(
                #     choices=["0, 飞机", "1, 汽车", "2, 鸟", "3, 猫",
                #              "4, 鹿", "5, 狗", "6, 蛙", "7, 马", "8, 船", "9, 卡车"],
                #     label="图像标签",
                #     info="该图像的标签是哪种类型？"
                # )
                target = gr.Dropdown(
                    choices=["-1, 任意", "0, 飞机", "1, 汽车", "2, 鸟", "3, 猫",
                             "4, 鹿", "5, 狗", "6, 蛙", "7, 马", "8, 船", "9, 卡车"],
                    label="攻击目标",
                    info="想诱导分类模型识别为哪种类型？"
                )
                eps = gr.Slider(0, 255,
                                label='eps',
                                info="想设置多少扰动预算？")
                generate_btn = gr.Button("生成对抗样本").style(full_width=False)

        with gr.Row(variant="compact"):
            # 展示迁移效果应该用一个欺骗多个吧？
            generator_model = gr.Radio(["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                                        "VGG11", "VGG13", "VGG16",
                                        "GoogLeNet", "MobileNet", "MobileNetV2"
                                        ],
                                       label="生成器模型",
                                       info="想用哪些生成器生成？")
            classifier_models = gr.CheckboxGroup(["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                                                  "VGG11", "VGG13", "VGG16",
                                                  "GoogLeNet", "MobileNet", "MobileNetV2"
                                                  ],
                                                 label="分类模型",
                                                 info="想尝试欺骗哪种分类模型？")
        adv_images = gr.Gallery(label="生成的对抗样本").style(grid=6)
        pwd = os.path.dirname(__file__)
        gr.Markdown("## 原始图像示例")
        gr.Examples(
            # examples=os.path.join(os.path.dirname(__file__), "examples"),
            examples=[
                [os.path.join(pwd, "examples", '0.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '1.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '2.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '3.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '4.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '5.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '6.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '7.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '8.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices],
                [os.path.join(pwd, "examples", '9.jpg'), "-1, 任意", 20, "ResNet50", classifier_models.choices]
            ],
            inputs=[original_image, target, eps, generator_model, classifier_models],
            outputs=adv_images,
            fn=generate_adv_images,
            # cache_examples=True,
            label="示例"
        )

    generate_btn.click(fn=generate_adv_images,
                       # inputs=[original_image, image_label, target, eps, generator_model, classifier_models],
                       inputs=[original_image, target, eps, generator_model, classifier_models],
                       outputs=adv_images)

if __name__ == "__main__":
    demo.launch()
