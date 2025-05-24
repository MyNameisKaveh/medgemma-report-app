.
import gradio as gr
from transformers import pipeline
from PIL import Image
# import requests # برای این کد فعلاً نیازی به requests نیست مگر اینکه بخواهید تصویر را از URL بخوانید
import torch
import os

# --- توکن Hugging Face برای دسترسی به مدل‌های Gated توسط app.py ---
# این کد، وقتی در Hugging Face Space شما اجرا می‌شود، انتظار دارد HF_TOKEN
# به عنوان یک "Secret" در تنظیمات Space شما تنظیم شده باشد.
# این توکن باید متعلق به حسابی باشد که شرایط استفاده از google/medgemma-4b-it را پذیرفته است.
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")

# --- متغیرهای سراسری برای مدل و وضعیت بارگذاری ---
model_pipeline = None
model_load_error_message = None
model_loaded_successfully = False

# --- منطق بارگذاری مدل ---
# این بخش در ابتدای اجرای Space یک بار انجام می‌شود.
try:
    print("MedGemma App: در حال تلاش برای بارگذاری مدل google/medgemma-4b-it...")
    # کاربر مشخص کرده که فعلاً از CPU استفاده شود.
    device_to_use = "cpu"
    # برای CPU بهتر است از torch.float32 استفاده شود.
    # torch.bfloat16 بیشتر برای GPU/TPU یا CPUهای جدیدتر با پشتیبانی خاص است.
    dtype_to_use = torch.float32
    print(f"MedGemma App: دستگاه مورد استفاده: {device_to_use}. این فرآیند برای MedGemma 4B بسیار کند خواهد بود و ممکن است با مشکل حافظه مواجه شوید.")

    # برای مدل‌های gated، اگر HF_TOKEN به عنوان متغیر محیطی (Space secret) تنظیم شده باشد،
    # توابع from_pretrained و pipeline باید به طور خودکار از آن استفاده کنند.
    # ارسال صریح token=HUGGING_FACE_TOKEN نیز برای اطمینان بیشتر خوب است.
    model_pipeline = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=dtype_to_use,
        device=device_to_use,
        token=HUGGING_FACE_TOKEN # ارسال صریح توکن
    )
    model_loaded_successfully = True
    print("MedGemma App: مدل با موفقیت بارگذاری شد.")
except Exception as e:
    model_load_error_message = str(e)
    print(f"MedGemma App: خطای بسیار مهم در بارگذاری مدل: {model_load_error_message}")
    # این خطا در رابط کاربری Gradio نمایش داده خواهد شد اگر مدل بارگذاری نشود.

# --- تابع اصلی برای پردازش با Gradio ---
def generate_medical_report(input_image_pil, user_prompt_text):
    if not model_loaded_successfully:
        return f"خطا: مدل قابل بارگذاری نبود: {model_load_error_message}. لطفاً لاگ‌های Space، تنظیمات سخت‌افزار، و اطمینان از تنظیم صحیح HF_TOKEN به عنوان Secret در تنظیمات Space و پذیرش شرایط MedGemma را بررسی کنید."
    
    if input_image_pil is None:
        return "خطا: لطفاً یک تصویر آپلود کنید."
    if not user_prompt_text or not user_prompt_text.strip():
        return "خطا: لطفاً یک سوال یا توضیح وارد کنید."

    print(f"MedGemma App: درخواست جدید دریافت شد. پرامپت: '{user_prompt_text}'. نوع تصویر: {type(input_image_pil)}")

    # ساختار پیام‌ها مطابق با نمونه‌های Model Card برای MedGemma
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist. Provide descriptive and informative insights based on the image and query."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_text},
                {"type": "image", "image": input_image_pil}, # Gradio تصویر را به صورت آبجکت PIL می‌دهد
            ]
        }
    ]

    try:
        print("MedGemma App: در حال تولید گزارش با پایپ‌لاین مدل...")
        # استفاده از ساختار فراخوانی pipeline مطابق با Model Card:
        # کد نمونه Model Card: output = pipe(text=messages, max_new_tokens=200)
        # این پارامتر text=messages برای این پایپ‌لاین خاص است.
        raw_output = model_pipeline(text=messages, max_new_tokens=512) 
        print(f"MedGemma App: خروجی خام از پایپ‌لاین: {raw_output}")

        # پردازش خروجی مطابق با ساختار نمونه Model Card:
        # print(output[0]["generated_text"][-1]["content"])
        final_text = "MedGemma App: خطای پیش‌بینی نشده در پردازش خروجی مدل." # مقدار پیش‌فرض در صورت بروز مشکل
        if raw_output and isinstance(raw_output, list) and len(raw_output) > 0:
            generated_text_output = raw_output[0].get("generated_text") # استفاده از .get برای جلوگیری از KeyError
            if generated_text_output and isinstance(generated_text_output, list) and len(generated_text_output) > 0:
                last_message_part = generated_text_output[-1]
                if isinstance(last_message_part, dict) and "content" in last_message_part:
                    final_text = last_message_part["content"]
                elif isinstance(last_message_part, str): # حالت جایگزین اگر ساختار ساده‌تر باشد
                    final_text = last_message_part
                else:
                    final_text = "MedGemma App: امکان پردازش بخش 'content' از آخرین پیام مدل وجود نداشت. ممکن است ساختار خروجی تغییر کرده باشد."
                    print(f"MedGemma App: ساختار غیرمنتظره در last_message_part: {last_message_part}")
            else:
                final_text = "MedGemma App: لیست 'generated_text' در خروجی مدل وجود ندارد، خالی است یا از نوع لیست نیست."
                print(f"MedGemma App: ساختار غیرمنتظره در generated_text_output: {generated_text_output}")
        else:
            final_text = "MedGemma App: خروجی مدل خالی است یا در فرمت لیست مورد انتظار نیست."
            print(f"MedGemma App: خروجی خام غیرمنتظره: {raw_output}")
        
        print(f"MedGemma App: گزارش تولید شد. متن نهایی (۱۰۰ کاراکتر اول): '{final_text[:100]}...'")
        return final_text
    except Exception as e:
        error_message = f"MedGemma App: خطا در هنگام تولید گزارش: {str(e)}"
        print(error_message) # لاگ کردن خطای کامل
        import traceback
        print(traceback.format_exc()) # لاگ کردن stack trace برای اشکال‌زدایی بهتر
        return error_message

# --- تعریف رابط کاربری Gradio ---
# متن سلب مسئولیت (Disclaimer) بر اساس شرایط استفاده MedGemma
disclaimer_markdown = """
---
**⚠️ سلب مسئولیت مهم (Disclaimer):**

این اپلیکیشن از مدل هوش مصنوعی MedGemma (`google/medgemma-4b-it`) فقط برای اهداف نمایشی، آموزشی و اطلاع‌رسانی استفاده می‌کند.
اطلاعات و گزارش‌های تولید شده توسط این ابزار **به هیچ عنوان نباید به عنوان مشاوره، تشخیص، یا توصیه درمانی پزشکی حرفه‌ای تلقی شوند.**

* همیشه برای هرگونه سوال یا نگرانی در مورد وضعیت پزشکی خود یا دیگران، با پزشک یا متخصص واجد شرایط مشورت کنید.
* هرگز اطلاعات یا توصیه‌های پزشکی حرفه‌ای را به دلیل چیزی که از این اپلیکیشن دریافت کرده‌اید، نادیده نگیرید یا در پیگیری آن تأخیر نکنید.
* توسعه‌دهندگان این اپلیکیشن و مدل MedGemma هیچ مسئولیتی در قبال تصمیمات یا اقداماتی که بر اساس خروجی این اپلیکیشن انجام می‌شود، ندارند.
* استفاده از مدل MedGemma تحت شرایط و قوانین "Health AI Developer Foundations" گوگل است. این یک مدل تحقیقاتی است و برای استفاده بالینی مستقیم تایید نشده است.

**این ابزار جایگزین قضاوت بالینی یک متخصص پزشکی نیست.**
"""

# ساخت رابط کاربری با Gradio Blocks برای کنترل بیشتر روی چیدمان
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo_interface:
    gr.Markdown("# 🩺 MedGemma - دستیار گزارش‌دهی پزشکی (نسخه آزمایشی)")
    gr.Markdown(disclaimer_markdown) # نمایش سلب مسئولیت در بالای صفحه

    # نمایش وضعیت بارگذاری مدل
    if not model_loaded_successfully:
        gr.Warning(f"هشدار جدی: مدل بارگذاری نشده است یا در بارگذاری با خطا مواجه شده است: {model_load_error_message}")
    elif device_to_use == "cpu":
        gr.Info("توجه: مدل روی CPU بارگذاری شده است. این فرآیند بسیار کند خواهد بود (هم بارگذاری و هم تولید گزارش) و ممکن است حافظه زیادی مصرف کند.")
    else: # model_loaded_successfully and not CPU (i.e., GPU)
        gr.Success("مدل با موفقیت روی GPU بارگذاری شد و آماده استفاده است.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image_component = gr.Image(type="pil", label="۱. تصویر پزشکی را آپلود کنید (مثلاً X-ray, CT Scan)")
            input_prompt_component = gr.Textbox(lines=3, label="۲. سوال یا درخواست خود را بنویسید:", placeholder="مثال: این تصویر را توصیف کن. آیا یافته غیرطبیعی وجود دارد؟ علائم اصلی چیست؟")
            submit_button_component = gr.Button("🚀 تولید گزارش", variant="primary")
        with gr.Column(scale=1):
            output_report_component = gr.Textbox(lines=15, label="نتیجه تحلیل مدل:", interactive=False, show_copy_button=True)
    
    # بخش مثال‌ها (اختیاری، می‌توانید حذف کنید اگر تصویر نمونه ندارید)
    # برای استفاده از مثال‌ها، باید فایل‌های تصویر نمونه را در ریپازیتوری خود قرار دهید.
    # gr.Examples(
    #     examples=[
    #         ["example_images/chest_xray_example.jpg", "Describe this chest X-ray focusing on the lungs and heart."],
    #         ["example_images/skin_lesion_example.png", "What are the characteristics of this skin lesion?"]
    #     ],
    #     inputs=[input_image_component, input_prompt_component],
    #     outputs=output_report_component,
    #     fn=generate_medical_report,
    #     cache_examples=False # یا True اگر مثال‌ها ثابت هستند
    # )
            
    submit_button_component.click(
        fn=generate_medical_report,
        inputs=[input_image_component, input_prompt_component],
        outputs=output_report_component
    )

# --- راه‌اندازی اپلیکیشن Gradio ---
if __name__ == "__main__":
    # این بخش برای اجرای محلی است. در Hugging Face Spaces، خود پلتفرم launch() را مدیریت می‌کند.
    # اگر می‌خواهید به صورت محلی تست کنید، می‌توانید تصاویر نمونه را بسازید یا مسیر صحیح به آن‌ها بدهید.
    # if not os.path.exists("example_images"):
    #    os.makedirs("example_images", exist_ok=True)
    # if not os.path.exists("example_images/chest_xray_example.jpg"):
    #    try:
    #        Image.new('RGB', (512, 512), color = 'grey').save("example_images/chest_xray_example.jpg")
    #    except Exception as e: print(f"Could not create placeholder image: {e}")
            
    demo_interface.launch() # می‌توانید برای تست محلی share=True را اضافه کنید تا لینک عمومی موقت بدهد
