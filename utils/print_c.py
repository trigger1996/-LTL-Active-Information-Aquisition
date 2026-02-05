import os
import logging
import datetime
import random

from collections import Counter
from colorama import Fore, Back, Style, init
from typing import Union, List
import re

# 创建日志文件夹（如果不存在）
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 生成全局日志文件名
#LOG_FILE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
LOG_FILE = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))


# 配置日志（仅初始化一次）
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(message)s',     # 日志格式
    handlers=[
        logging.FileHandler(LOG_FILE),      # 输出到文件
        #logging.StreamHandler()            # 输出到控制台
    ]
)

def print_c(
        data,
        color: Union[int, str] = "green",
        style: Union[str, List[str]] = None,
        bg_color: Union[int, str] = None,
        is_logging=True,
        log_file: str = LOG_FILE,
        **kwargs
):
    """
    增强版颜色样式打印输出功能，支持多种文本样式和背景色

    参数:
        data: 要打印的内容
        color: 颜色，可以是数字(30-37)或字符串名称(如'red','green')
        style: 文本样式，支持: 'bold','italic','underline','blink','reverse'
        bg_color: 背景色，可以是数字(40-47)或字符串名称(如'bg_red','bg_green')
        log_file: 日志文件路径，如果为None则不记录日志
        kwargs: 其他日志配置参数

    颜色代码:
        前景色: 黑色(30), 红色(31), 绿色(32), 黄色(33),
               蓝色(34), 紫色(35), 青色(36), 白色(37)
        背景色: 黑色(40), 红色(41), 绿色(42), 黄色(43),
               蓝色(44), 紫色(45), 青色(46), 白色(47)
    """
    # 颜色和样式映射表
    color_map = {
        'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
        'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
        'bg_black': 40, 'bg_red': 41, 'bg_green': 42, 'bg_yellow': 43,
        'bg_blue': 44, 'bg_magenta': 45, 'bg_cyan': 46, 'bg_white': 47
    }

    style_map = {
        'bold': 1, 'italic': 3, 'underline': 4,
        'blink': 5, 'reverse': 7
    }

    # 处理颜色参数
    if isinstance(color, str):
        color_code = color_map.get(color.lower(), 32)  # 默认绿色
    else:
        color_code = color if 30 <= color <= 37 else 32

    # 处理样式参数
    style_codes = []
    if style:
        if isinstance(style, str):
            style = [style]
        for s in style:
            if s.lower() in style_map:
                style_codes.append(str(style_map[s.lower()]))

    # 处理背景色参数
    bg_code = ""
    if bg_color:
        if isinstance(bg_color, str):
            bg_code = str(color_map.get(f"bg_{bg_color.lower()}", 40))
        elif 40 <= bg_color <= 47:
            bg_code = str(bg_color)

    # 构建ANSI转义序列
    codes = style_codes + [str(color_code)]
    if bg_code:
        codes.append(bg_code)
    ansi_start = "\033[" + ";".join(codes) + "m"
    ansi_end = "\033[0m"

    # 打印带样式的文本
    print(f"{ansi_start}{data}{ansi_end}")

    # 记录日志
    if is_logging and log_file:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            **kwargs
        )
        data_str = str(data)
        logging.info(data_str)