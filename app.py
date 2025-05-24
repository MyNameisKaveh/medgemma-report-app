import gradio as gr
from transformers import pipeline
from PIL import Image
import requests
import torch
import os

# --- مدیریت توکن Hugging Face برای مدل‌های نیازمند مجوز ---
# برای مدل‌هایی مثل MedGemma که نیاز به پذیرش شرایط دارند،
# توکن Hugging Face شما باید در دسترس باشد.
# در Hugging Face Spaces، بهترین راه تنظیم آن به عنوان یک "Secret" با نام HF_TOKEN است.
# به تب "Settings" در Space خود بروید و یک "New secret" با نام HF_TOKEN
# و مقدار توکن دسترسی خود از Hugging Face (از بخش Access Tokens در تنظیمات پروفایلتان) اضافه کنید.
# این توکن باید به حسابی تعلق داشته باشد که شرایط مدل MedGemma را پذیرفته است.
# huggingface_token = os.getenv("HF_TOKEN") # خواندن توکن از Secret

# --- بارگذاری مدل ---
# این بخش ممکن است زمان‌بر باشد و حافظه زیادی مصرف کند.
# انتخاب دستگاه (CPU/GPU) بسیار مهم است.

# به دلیل پیچیدگی‌های مربوط به توکن در بارگذاری اولیه مدل در اسکریپت،
# فعلاً فرض می‌کنیم اگر شرایط را پذیرفته‌اید و از یک محیط متصل به اینترنت استفاده می‌کنید،
# کتابخانه transformers بتواند مدل را دانلود کند.
# اگر با مشکل مجوز مواجه شدید، استفاده از HF_TOKEN به عنوان Secret ضروری است
# و ممکن است نیاز باشد در فراخوانی pipeline از پارامتر token=huggingface_token استفاده کنید.

model_pipeline = None
model_load_error = None
model_loaded_successfully = False

try:
    print("Attempting to load model...")
    # انتخاب دستگاه: اولویت با GPU
    if torch.cuda.is_available():
        device_to_use = 0  # اولین GPU در دسترس
        dtype_to_use = torch.bfloat16
        print(f"GPU found ({torch.cuda.get_device_name(device_to_use)}). Using GPU.")
    else:
        device_to_use = "cpu"
        dtype_to_use = torch.float32 # bfloat16 برای همه CPU ها مناسب نیست
        print("Warning: GPU not found. Using CPU. This will be very slow and might lead to memory issues.")

    model_pipeline = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=dtype_to_use,
        device=device_to_use,
        # token=huggingface_token # در صورت نیاز به توکن برای دانلود مدل
    )
    model_loaded_successfully = True
    print("Model loaded successfully.")
except Exception as e:
    model_load_error = str(e)
    print(f"Error loading model: {model_load_error}")
    # در صورت بروز خطا هنگام بارگذاری مدل، در رابط کاربری نمایش داده می‌شود.


# --- تابع اصلی برای پردازش با Gradio ---
def generate_medical_report(input_image, user_prompt):
    if not model_loaded_successfully:
        return f"خطا در بارگذاری مدل: {model_load_error}. لطفاً لاگ‌های Space و سخت‌افزار انتخاب شده را بررسی کنید. مطمئن شوید شرایط استفاده از مدل را پذیرفته‌اید و در صورت نیاز، HF_TOKEN را به عنوان Secret تنظیم کرده‌اید."
    
    if input_image is None:
        return "لطفاً یک تصویر آپلود کنید."
    if not user_prompt or not user_prompt.strip():
        return "لطفاً یک سوال یا توضیح وارد کنید."

    # ساختار پیام‌ها مطابق با Model Card
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": input_image}, # input_image از Gradio به صورت PIL Image می‌آید
            ]
        }
    ]

    try:
        print("Generating report...")
        # استفاده از فراخوانی pipeline مطابق با نمونه کد Model Card
        # output = model_pipeline(text=messages, max_new_tokens=512)
        # با توجه به اینکه `messages` شامل تصویر هم هست، و `image-text-to-text` pipeline
        # ممکن است ورودی‌ها را به شکل دیگری انتظار داشته باشد،
        # بهتر است از فرمتی استفاده کنیم که مستقیماً تصویر و متن را جداگانه بگیرد
        # یا مطمئن شویم pipeline به درستی `messages` را پردازش می‌کند.
        # نمونه کد model card برای pipeline از `text=messages` استفاده کرده که کمی غیرمعمول است.
        # اجازه دهید همان را امتحان کنیم.
        
        output = model_pipeline(messages, max_new_tokens=512) # مطابق مثال pipeline عمومی‌تر
        # اگر با خطا مواجه شد، به فرمت Model Card برگردید:
        # output = model_pipeline(text=messages, max_new_tokens=512)

        # پردازش خروجی مطابق با ساختار مورد انتظار از Model Card
        # print(output[0]["generated_text"][-1]["content"])
        generated_output_list = output[0]["generated_text"]
        final_text = ""
        if isinstance(generated_output_list, list) and len(generated_output_list) > 0:
            last_message_part = generated_output_list[-1]
            if isinstance(last_message_part, dict) and "content" in last_message_part:
                final_text = last_message_part["content"]
            elif isinstance(last_message_part, str):
                final_text = last_message_part
            else:
                final_text = "ساختار خروجی مدل قابل پردازش نبود."
        else:
            final_text = f"خروجی غیرمنتظره از مدل: {str(output)}"
        
        print("Report generated.")
        return final_text
    except Exception as e:
        error_message = f"خطا در هنگام تولید گزارش: {str(e)}"
        print(error_message)
        return error_message

# --- تعریف رابط کاربری Gradio ---
# متن سلب مسئولیت (Disclaimer)
disclaimer_text = """
**سلب مسئولیت (Disclaimer):**
این اپلیکیشن از مدل MedGemma فقط برای اهداف نمایشی و اطلاع‌رسانی استفاده می‌کند.
گزارش‌های تولید شده توسط این اپلیکیشن **جایگزین مشاوره، تشخیص یا درمان پزشکی حرفه‌ای نیستند.**
همیشه در مورد هرگونه سوالی که در رابطه با یک وضعیت پزشکی دارید، از پزشک خود یا سایر ارائه‌دهندگان خدمات بهداشتی واجد شرایط مشاوره بگیرید.
هرگز به دلیل چیزی که از این اپلیکیشن خوانده یا دیده‌اید، مشاوره پزشکی حرفه‌ای را نادیده نگیرید یا در جستجوی آن تأخیر نکنید.
استفاده از مدل MedGemma تحت شرایط و قوانین "Health AI Developer Foundations" است.
"""

# ایجاد رابط کاربری
interface = gr.Interface(
    fn=generate_medical_report,
    inputs=[
        gr.Image(type="pil", label="آپلود تصویر پزشکی (مانند عکس رادیولوژی)"),
        gr.Textbox(label="سوال یا درخواست شما (مثال: این عکس رادیولوژی را توصیف کن، یافته‌ها چیست؟)")
    ],
    outputs=gr.Textbox(label="اطلاعات تولید شده توسط مدل"),
    title=" دستیار گزارش پزشکی MedGemma (نسخه آزمایشی)",
    description="یک تصویر پزشکی آپلود کنید و سوال خود را بپرسید. این ابزار از مدل google/medgemma-4b-it برای تولید پاسخ استفاده می‌کند.",
    article=disclaimer_text, # نمایش متن سلب مسئولیت
    allow_flagging="never" # غیرفعال کردن قابلیت پرچم‌گذاری (اختیاری)
)

# --- راه‌اندازی اپلیکیشن ---
if __name__ == "__main__":
    interface.launch()
