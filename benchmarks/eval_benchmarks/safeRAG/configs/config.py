# config.py

# 这个部分保持默认或留空，因为你使用的是代理URL，而不是DeepSeek官方URL
DeepSeek_key = ''
DeepSeek_base = 'https://api.deepseek.com'

# --- 重点：在这里填入你获得的API Key和Base URL ---
# 即使你要调用DeepSeek模型，但因为是通过兼容OpenAI的代理URL，所以要填在这里
GPT_api_key = 'sk-AVsyIUmeAjeyEyhBA981E921C5304b079540091115430e97'
GPT_api_base = 'https://aihubmix.com/v1'

# 如果你不使用本地模型，这些路径可以留空
Qwen_7B_local_path = ''
Qwen_14B_local_path = ''
Baichuan2_13b_local_path = ''
ChatGLM3_local_path = ''