

import random
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

# 加载YOLOv3-tiny模型（可改成你的 yolov3-tiny.pt）
model = YOLO('yolov8n.pt')

# 初始化玩家分数
players_score = {"玩家A": 0, "玩家B": 0}
current_player = "玩家A"

# 获取COCO类别列表
coco_classes = model.names  # dict形式，例如 {0: 'person', 1: 'bicycle', ...}

# 游戏状态
game_state = {
    "target": random.choice(list(coco_classes.values())),
    "round": 1,
}

def switch_player():
    global current_player
    current_player = "玩家B" if current_player == "玩家A" else "玩家A"

def get_current_target():
    """显示当前轮玩家和目标物体"""
    return f"轮到玩家: {current_player}\n目标物体: {game_state['target']}"

def game_round(image):
    global players_score, current_player, game_state

    if image is None:
        return None, "请上传图片", players_score

    # YOLO检测
    results = model(image)
    detected_classes = []
    for r in results:
        for cls_id in r.boxes.cls:
            detected_classes.append(coco_classes[int(cls_id)])

    # 判断是否拍对目标
    target = game_state["target"]
    if target in detected_classes:
        players_score[current_player] += 1
        feedback = f"✅ {current_player} 找到 {target}！得1分"
    else:
        players_score[current_player] -= 1
        feedback = f"❌ {current_player} 拍错了！图片中没有 {target}，扣1分"

    # 绘制检测框
    annotated_image = results[0].plot()

    # 切换玩家 & 更新目标
    switch_player()
    game_state["target"] = random.choice(list(coco_classes.values()))
    game_state["round"] += 1

    feedback += f"\n下一轮目标: {game_state['target']}\n轮到玩家: {current_player}"
    return annotated_image, feedback, players_score

# Gradio界面
title = "📸 拍错物体惩罚 / 对战小游戏"
description = """
游戏规则：
1. 系统随机指定一个目标物体。
2. 当前玩家上传图片。
3. 如果图片中包含目标物体 → 得分，否则 → 扣分。
4. 两个玩家轮流进行。
"""

with gr.Blocks() as demo:
    gr.Markdown("# 📸 拍错物体惩罚 / 对战小游戏")
    gr.Markdown(description)
    
    target_text = gr.Textbox(label="当前目标物体", value=get_current_target(), interactive=False)
    uploaded_img = gr.Image(type="numpy", label="上传图片")
    output_img = gr.Image(type="numpy", label="检测结果")
    feedback_box = gr.Textbox(label="游戏反馈")
    score_box = gr.JSON(label="玩家分数")

    def update_target_text():
        return get_current_target()

    btn = gr.Button("开始本轮")
    btn.click(fn=update_target_text, outputs=target_text)
    uploaded_img.change(fn=game_round, inputs=uploaded_img, outputs=[output_img, feedback_box, score_box])

demo.launch()
