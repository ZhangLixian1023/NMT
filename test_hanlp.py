from data.preprocessor import Preprocessor

# 创建preprocessor实例
preprocessor = Preprocessor()

# 测试英文文本处理（添加小写转换）
en_text = "Records indicate that HMX1 inquired about whether the event might violate the provision."
processed_en = preprocessor.process_text(en_text, 'en')
print(f"原始英文文本: {en_text}")
print(f"处理后英文文本: {processed_en}")

# 测试中文文本处理（从jieba迁移到HanLP）
zh_text = "记录指出HMX1曾询问此次活动是否违反了该法案"
processed_zh = preprocessor.process_text(zh_text, 'zh')
print(f"\n原始中文文本: {zh_text}")
print(f"处理后中文文本: {processed_zh}")

# 测试更复杂的英文文本（包含大小写和特殊字符）
complex_en = "\"One question we asked was if it was a violation of the Hatch Act and were informed it was not,\" the commander wrote."
processed_complex_en = preprocessor.process_text(complex_en, 'en')
print(f"\n原始复杂英文文本: {complex_en}")
print(f"处理后复杂英文文本: {processed_complex_en}")

print("\n✅ 测试完成！")
