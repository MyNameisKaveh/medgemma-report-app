import gradio as gr
from gradio_client import Client # برای اتصال به سایر Space ها
from PIL import Image # برای کار با فرمت تصویر ورودی از Gradio
import os
import uuid # برای ساخت نام فایل موقت و یکتا برای تصاویر
import traceback # برای نمایش بهتر خطاها در لاگ

# --- تنظیمات اولیه ---
RAD_EXPLAIN_SPACE_ID = "google/rad_explain"  # شناسه Space مقصد
# یا می‌توانید از URL کامل استفاده کنید: "https://google-rad-explain.hf.space/"
REPORT_GENERATION_FN_INDEX = 4  # ایندکس تابع تولید گزارش در Rad-Explain

# نام پوشه برای ذخیره موقت تصاویر آپلود شده توسط کاربر
# این پوشه در ریشه Space شما ساخته خواهد شد (اگر وجود نداشته باشد)
TEMP_IMAGE_DIR = "temp_uploaded_images" 

# --- متغیرهای سراسری برای وضعیت کلاینت ---
gradio_client_instance = None
gradio_client_error_message = None
gradio_client_loaded_successfully = False

# --- تلاش برای ساخت کلاینت هنگام شروع به کار اپلیکیشن ---
try:
    print(f"RadExplain API App: در حال ساخت کلاینت برای اتصال به Space ID: {RAD_EXPLAIN_SPACE_ID}...")
    gradio_client_instance = Client(RAD_EXPLAIN_SPACE_ID)
    gradio_client_loaded_successfully = True
    print("RadExplain API App: کلاینت Gradio با موفقیت ساخته شد.")

    # ساخت پوشه موقت برای تصاویر در صورت عدم وجود
    if not os.path.exists(TEMP_IMAGE_DIR):
        os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
        print(f"RadExplain API App: پوشه موقت تصاویر در '{TEMP_IMAGE_DIR}' ساخته شد.")

except Exception as e:
    gradio_client_error_message = str(e)
    print(f"RadExplain API App: خطای بسیار مهم در هنگام ساخت کلاینت Gradio: {gradio_client_error_message}")
    print(traceback.format_exc()) # چاپ کامل خطا در لاگ برای بررسی بیشتر


# --- تابع اصلی که با کلیک روی دکمه در Gradio فراخوانی می‌شود ---
def get_report_from_rad_explain_api(pil_image_input, user_prompt_text):
    # ورودی user_prompt_text فعلاً برای این تابع خاص Rad-Explain (fn_index=4) استفاده نمی‌شود،
    # اما آن را در رابط کاربری نگه می‌داریم شاید کاربر بخواهد زمینه‌ای برای سوال خود وارد کند.
    
    if not gradio_client_loaded_successfully:
        error_msg = f"خطا: کلاینت برای ارتباط با سرویس Rad-Explain در دسترس نیست: {gradio_client_error_message}"
        print(f"RadExplain API App: {error_msg}")
        return error_msg  # نمایش خطا به کاربر

    if pil_image_input is None:
        error_msg = "خطا: لطفاً ابتدا یک تصویر آپلود کنید."
        print(f"RadExplain API App: {error_msg}")
        return error_msg

    print(f"RadExplain API App: درخواست جدید دریافت شد. پرامپت کاربر (ممکن است استفاده نشود): '{user_prompt_text}'. نوع تصویر: {type(pil_image_input)}")

    temp_image_path = None  # برای نگهداری مسیر فایل تصویر موقت
    try:
        # کتابخانه gradio_client برای ارسال تصویر، اغلب با مسیر فایل بهتر کار می‌کند.
        # بنابراین، تصویر PIL دریافت شده از کاربر را به صورت موقت ذخیره می‌کنیم.
        unique_filename = f"{uuid.uuid4()}.png" # یک نام یکتا برای فایل موقت
        temp_image_path = os.path.join(TEMP_IMAGE_DIR, unique_filename)
        
        # ذخیره تصویر آپلود شده در مسیر موقت
        pil_image_input.save(temp_image_path)
        print(f"RadExplain API App: تصویر موقت در مسیر زیر ذخیره شد: {temp_image_path}")

        print(f"RadExplain API App: در حال فراخوانی API سرویس Rad-Explain (fn_index={REPORT_GENERATION_FN_INDEX}) با تصویر: {temp_image_path}...")
        
        # فراخوانی تابع predict از کلاینت با ایندکس مربوط به تولید گزارش
        # ورودی برای fn_index=4 در rad_explain فقط خود تصویر است.
        api_result = gradio_client_instance.predict(
            temp_image_path,  # ارسال مسیر فایل تصویر
            fn_index=REPORT_GENERATION_FN_INDEX
        )
        
        print(f"RadExplain API App: نتیجه خام دریافت شده از API: {api_result}")
        
        # نتیجه باید متن گزارش باشد
        report_text = str(api_result) if api_result is not None else "گزارشی تولید نشد یا پاسخ خالی بود."
        
        print(f"RadExplain API App: گزارش با موفقیت از API دریافت شد. طول گزارش: {len(report_text)}")
        return report_text

    except Exception as e:
        error_message = f"RadExplain API App: خطا در هنگام فراخوانی API سرویس Rad-Explain: {str(e)}"
        print(error_message)
        print(traceback.format_exc()) # چاپ کامل خطا برای اشکال‌زدایی
        return error_message # نمایش خطا به کاربر
    finally:
        # پاک کردن فایل تصویر موقت پس از اتمام کار (چه موفقیت‌آمیز چه با خطا)
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"RadExplain API App: فایل تصویر موقت پاک شد: {temp_image_path}")
            except Exception as e_clean:
                # اگر در پاک کردن فایل هم خطا رخ داد، فقط در لاگ می‌نویسیم و ادامه می‌دهیم
                print(f"RadExplain API App: خطا در پاک کردن فایل تصویر موقت {temp_image_path}: {e_clean}")

# --- تعریف رابط کاربری Gradio ---
# متن سلب مسئولیت (Disclaimer)
disclaimer_markdown_text = """
---
**⚠️ سلب مسئولیت مهم (Disclaimer):**

این اپلیکیشن از API سرویس `google/rad_explain` برای تولید اطلاعات استفاده می‌کند.
اطلاعات و گزارش‌های تولید شده **به هیچ عنوان نباید به عنوان مشاوره، تشخیص، یا توصیه درمانی پزشکی حرفه‌ای تلقی شوند.**

* همیشه برای هرگونه سوال یا نگرانی در مورد وضعیت پزشکی خود یا دیگران، با پزشک یا متخصص واجد شرایط مشورت کنید.
* توسعه‌دهندگان این اپلیکیشن هیچ مسئولیتی در قبال دقت یا کامل بودن اطلاعات دریافتی از سرویس `google/rad_explain` یا تصمیمات و اقداماتی که بر اساس خروجی این اپلیکیشن انجام می‌شود، ندارند.
* استفاده از سرویس `google/rad_explain` و مدل‌های زیربنایی آن (مانند MedGemma) تحت شرایط و قوانین تعیین شده توسط ارائه‌دهندگان آن سرویس‌ها است.

**این ابزار جایگزین قضاوت بالینی یک متخصص پزشکی نیست.**
"""

# استفاده از gr.Blocks برای کنترل بیشتر روی چیدمان رابط کاربری
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.green)) as app_interface:
    gr.Markdown("# 🩺 دستیار گزارش‌دهی پزشکی (با استفاده از API سرویس Rad-Explain)")
    gr.Markdown(disclaimer_markdown_text) # نمایش متن سلب مسئولیت

    # نمایش وضعیت اتصال به سرویس Rad-Explain
    if not gradio_client_loaded_successfully:
        gr.Warning(f"هشدار: امکان برقراری ارتباط با سرویس Rad-Explain برای تحلیل تصاویر وجود ندارد. لطفاً لاگ‌ها را برای جزئیات خطا بررسی کنید: {gradio_client_error_message}")
    else:
        gr.Success("سرویس تحلیل تصویر Rad-Explain آماده استفاده است.")

    with gr.Row(): # ایجاد یک ردیف برای چیدمان کنار هم
        with gr.Column(scale=1): # ستون اول برای ورودی‌ها
            image_input_component = gr.Image(type="pil", label="۱. تصویر پزشکی را آپلود کنید (مثلاً X-ray)")
            prompt_input_component = gr.Textbox(
                lines=2, 
                label="۲. سوال یا زمینه (اختیاری - ممکن است توسط API فعلی Rad-Explain برای تولید گزارش مستقیم استفاده نشود):", 
                placeholder="مثال: یافته‌های اصلی در این تصویر ریه چیست؟"
            )
            generate_button = gr.Button("🚀 دریافت گزارش از Rad-Explain", variant="primary")
        
        with gr.Column(scale=1): # ستون دوم برای خروجی
            report_output_component = gr.Textbox(
                lines=15, 
                label="گزارش تولید شده توسط Rad-Explain:", 
                interactive=False, # کاربر نتواند متن خروجی را ویرایش کند
                show_copy_button=True # دکمه کپی برای متن خروجی
            )
            
    # اتصال دکمه به تابع پردازشگر
    generate_button.click(
        fn=get_report_from_rad_explain_api,
        inputs=[image_input_component, prompt_input_component],
        outputs=report_output_component
    )

# --- راه‌اندازی اپلیکیشن Gradio ---
# این بخش برای اجرای مستقیم فایل app.py (مثلاً روی کامپیوتر شخصی) است.
# وقتی در Hugging Face Spaces اجرا می‌شود، خود پلتفرم اپلیکیشن را launch می‌کند.
if __name__ == "__main__":
    # قبل از لانچ، پوشه تصاویر موقت را می‌سازیم اگر وجود ندارد (برای اجرای محلی)
    if not os.path.exists(TEMP_IMAGE_DIR):
        os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
        print(f"Directory {TEMP_IMAGE_DIR} created for local run.")
    
    app_interface.launch() 
    # برای اشتراک‌گذاری لینک عمومی هنگام اجرای محلی، می‌توانید از share=True استفاده کنید: app_interface.launch(share=True)
