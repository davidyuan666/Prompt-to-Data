#!/usr/bin/env python3
"""
不可去除水印生成器
将文字水印嵌入到JPG图片的多个通道中，使其难以完全去除
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
from pathlib import Path

class WatermarkGenerator:
    def __init__(self, watermark_text="版权所有", font_path=None):
        """
        初始化水印生成器
        
        Args:
            watermark_text: 水印文字
            font_path: 字体文件路径，如果为None则使用默认字体
        """
        self.watermark_text = watermark_text
        
        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            if font_path and os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, 60)  # 增大字体大小到60
            else:
                # 尝试常见的中文字体
                common_fonts = [
                    "/System/Library/Fonts/PingFang.ttc",  # macOS
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                    "C:/Windows/Fonts/simhei.ttf",  # Windows
                    "C:/Windows/Fonts/msyh.ttc",  # Windows
                ]
                for font in common_fonts:
                    if os.path.exists(font):
                        self.font = ImageFont.truetype(font, 60)  # 增大字体大小到60
                        break
                else:
                    self.font = ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
    
    def add_robust_watermark(self, image_path, output_path=None, opacity=0.5, watermark_text=None):  # 添加watermark_text参数
        """
        添加不可去除的水印
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径，如果为None则覆盖原文件
            opacity: 水印透明度 (0-1)，默认0.5
            watermark_text: 水印文字，如果为None则使用初始化时的文字
        
        Returns:
            处理后的图片路径
        """
        # 使用指定的水印文字或默认文字
        current_text = watermark_text if watermark_text else self.watermark_text
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 获取图片尺寸
        height, width = img.shape[:2]
        
        # 创建多个水印层
        watermarked = img.copy()
        
        # 方法1: 在多个位置添加半透明水印（增强版）
        self._add_multiple_watermarks(watermarked, opacity, current_text)
        
        # 方法2: 在频域添加水印（更难以去除）
        watermarked = self._add_frequency_domain_watermark(watermarked, opacity)
        
        # 方法3: 在颜色通道添加轻微扰动
        watermarked = self._add_channel_perturbation(watermarked)
        
        # 保存图片
        if output_path is None:
            output_path = image_path.replace('.jpg', '_watermarked.jpg')
            if output_path == image_path:  # 如果没有.jpg后缀
                output_path = f"{image_path}_watermarked.jpg"
        
        # 使用高质量压缩
        cv2.imwrite(output_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"水印已添加，保存到: {output_path}")
        return output_path
    
    def _add_multiple_watermarks(self, image, opacity, watermark_text):
        """在多个位置添加水印（增强版）"""
        height, width = image.shape[:2]
        
        # 创建水印文字图片
        watermark_img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark_img)
        
        # 计算水印大小
        text_bbox = draw.textbbox((0, 0), watermark_text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 在多个位置添加水印
        positions = []
        spacing_x = int(text_width * 1.5)  # 减小间距，让水印更密集
        spacing_y = int(text_height * 1.5)
        
        for y in range(0, height, spacing_y):
            for x in range(0, width, spacing_x):
                positions.append((x, y))
        
        # 随机旋转角度
        angles = [-30, -15, 0, 15, 30]
        
        for pos in positions:
            x, y = pos
            # 随机选择旋转角度
            angle = random.choice(angles)
            
            # 创建单个水印 - 使用更深的颜色
            single_watermark = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
            single_draw = ImageDraw.Draw(single_watermark)
            
            # 添加文字阴影效果，增强可见性
            shadow_opacity = int(255 * opacity * 0.8)
            main_opacity = int(255 * opacity)
            
            # 添加阴影
            single_draw.text((2, 2), watermark_text, font=self.font, 
                           fill=(0, 0, 0, shadow_opacity))
            # 添加主文字
            single_draw.text((0, 0), watermark_text, font=self.font, 
                           fill=(255, 255, 255, main_opacity))
            
            # 旋转水印
            if angle != 0:
                single_watermark = single_watermark.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))
            
            # 粘贴到水印图层
            watermark_img.paste(single_watermark, (x, y), single_watermark)
        
        # 将水印叠加到原图
        watermark_np = cv2.cvtColor(np.array(watermark_img), cv2.COLOR_RGBA2BGRA)
        
        # 分离通道
        alpha = watermark_np[:, :, 3] / 255.0
        
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha) + watermark_np[:, :, c] * alpha
    
    def _add_frequency_domain_watermark(self, image, opacity):
        """在频域添加水印（难以去除）"""
        height, width = image.shape[:2]
        
        # 对每个颜色通道进行处理
        for channel in range(3):
            # 获取通道数据
            channel_data = image[:, :, channel].astype(np.float32)
            
            # 傅里叶变换
            dft = cv2.dft(channel_data, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # 创建水印模式 - 增强水印强度
            watermark_pattern = np.zeros_like(channel_data)
            rows, cols = channel_data.shape
            
            # 在频域添加水印模式 - 增加密度和强度
            for i in range(0, rows, 15):  # 减小间隔，增加密度
                for j in range(0, cols, 15):
                    if random.random() > 0.5:  # 增加概率
                        watermark_pattern[i, j] = opacity * 20  # 增加强度
            
            # 将水印模式添加到频域
            dft_shift[:, :, 0] += watermark_pattern
            
            # 逆傅里叶变换
            dft_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(dft_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            
            # 归一化并更新通道
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            image[:, :, channel] = img_back.astype(np.uint8)
        
        return image
    
    def _add_channel_perturbation(self, image):
        """在颜色通道添加轻微扰动"""
        height, width = image.shape[:2]
        
        # 创建微小的扰动模式 - 稍微增强
        perturbation = np.random.normal(0, 1.0, (height, width, 3)).astype(np.float32)  # 增加扰动强度
        
        # 将扰动添加到图像
        perturbed = image.astype(np.float32) + perturbation
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        
        return perturbed
    
    def add_visible_watermark(self, image_path, output_path=None, opacity=0.7, font_size=80, watermark_text=None):
        """
        添加更明显的水印（针对需要高可见性的场景）
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            opacity: 水印透明度 (0-1)，默认0.7
            font_size: 字体大小，默认80
            watermark_text: 水印文字，如果为None则使用初始化时的文字
        """
        # 使用指定的水印文字或默认文字
        current_text = watermark_text if watermark_text else self.watermark_text
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 创建PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        
        # 创建临时字体（更大）
        try:
            # 尝试加载字体
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 计算图片中心位置
        width, height = pil_img.size
        
        # 在图片中心添加大号水印
        text_bbox = draw.textbbox((0, 0), current_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算位置（居中）
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # 添加带阴影的大水印
        shadow_color = (0, 0, 0, int(255 * opacity * 0.6))
        text_color = (255, 255, 255, int(255 * opacity))
        
        # 阴影
        draw.text((x+3, y+3), current_text, font=font, fill=shadow_color)
        # 主文字
        draw.text((x, y), current_text, font=font, fill=text_color)
        
        # 在四个角落也添加水印
        positions = [
            (50, 50),  # 左上
            (width - text_width - 50, 50),  # 右上
            (50, height - text_height - 50),  # 左下
            (width - text_width - 50, height - text_height - 50)  # 右下
        ]
        
        for pos_x, pos_y in positions:
            # 阴影
            draw.text((pos_x+2, pos_y+2), current_text, font=font, fill=shadow_color)
            # 主文字
            draw.text((pos_x, pos_y), current_text, font=font, fill=text_color)
        
        # 转换回OpenCV格式
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 保存图片
        if output_path is None:
            output_path = image_path.replace('.jpg', '_visible_watermark.jpg')
            if output_path == image_path:
                output_path = f"{image_path}_visible_watermark.jpg"
        
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"明显水印已添加，保存到: {output_path}")
        return output_path
    
    def batch_process(self, input_dir, output_dir=None, opacity=0.5, visible=False, watermark_text=None):
        """
        批量处理文件夹中的所有JPG图片
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径，如果为None则创建"watermarked"子文件夹
            opacity: 水印透明度
            visible: 是否使用明显水印模式
            watermark_text: 水印文字，如果为None则使用初始化时的文字
        
        Returns:
            处理后的图片数量
        """
        input_path = Path(input_dir)
        
        if output_dir is None:
            if visible:
                output_dir = input_path / "visible_watermarked"
            else:
                output_dir = input_path / "watermarked"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有JPG文件
        jpg_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        
        processed_count = 0
        for jpg_file in jpg_files:
            try:
                if visible:
                    output_path = output_dir / f"{jpg_file.stem}_visible.jpg"
                    self.add_visible_watermark(str(jpg_file), str(output_path), opacity, watermark_text=watermark_text)
                else:
                    output_path = output_dir / f"{jpg_file.stem}_watermarked.jpg"
                    self.add_robust_watermark(str(jpg_file), str(output_path), opacity, watermark_text=watermark_text)
                processed_count += 1
                print(f"已处理: {jpg_file.name}")
            except Exception as e:
                print(f"处理失败 {jpg_file.name}: {e}")
        
        return processed_count

# python watermark_generator.py input.jpg -t "我的专属水印"

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="不可去除水印生成器")
    parser.add_argument("input", help="输入图片路径或文件夹路径")
    parser.add_argument("-o", "--output", help="输出路径（文件或文件夹）")
    parser.add_argument("-t", "--text", default="版权所有", help="水印文字，默认'版权所有'")
    parser.add_argument("--opacity", type=float, default=0.5, help="水印透明度 (0-1)，默认0.5")
    parser.add_argument("--font", help="字体文件路径")
    parser.add_argument("--batch", action="store_true", help="批量处理文件夹")
    parser.add_argument("--visible", action="store_true", help="使用明显水印模式（更大更清晰）")
    parser.add_argument("--font-size", type=int, default=80, help="明显水印模式的字体大小")
    
    args = parser.parse_args()
    
    # 创建水印生成器，使用命令行指定的水印文字
    generator = WatermarkGenerator(args.text, args.font)
    
    if args.batch or os.path.isdir(args.input):
        # 批量处理
        processed = generator.batch_process(args.input, args.output, args.opacity, args.visible, args.text)
        print(f"\n批量处理完成！共处理 {processed} 张图片")
    else:
        # 单文件处理
        if args.visible:
            output_path = generator.add_visible_watermark(args.input, args.output, args.opacity, args.font_size, args.text)
        else:
            output_path = generator.add_robust_watermark(args.input, args.output, args.opacity, args.text)
        print(f"\n处理完成！输出文件: {output_path}")
        print(f"使用的水印文字: '{args.text}'")


if __name__ == "__main__":
    main()