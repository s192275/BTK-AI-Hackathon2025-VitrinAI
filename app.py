import streamlit as st
import google.generativeai as genai
import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
import logging 
import warnings 
from PIL import Image
import io
from ai_modules import real_esrgan_and_clahe
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from dotenv import load_dotenv
import os

# Uyarıları ve loglama ayarlarını yapılandır
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename = "app.log", filemode='a', encoding = "utf-8")
load_dotenv()
def whisper_model(model_size: str = "turbo", device: str = "cpu", fs: int = 16000, 
                  chunk_duration: int = 9, total_duration: int = 10) -> str:
    """
    Ses kaydı alır ve whisper modelini kullanarak metne dönüştürür. Bu işlem, faster-whisper kütüphanesi ile normalden daha hızlı yapılır.

    Args:
        model_size (str, optional): _description_. Default olarak turbo modeli kullanılır ancak faster-whisper kütüphanesindeki diğer model çeşitleri de kullanılır.
        device (str, optional): _description_. Cihaz tipi default olarak "cpu" dadır. Ancak cuda cihazı varsa "gpu" ya çekilebilir.
        fs (int, optional): _description_. Örnekleme frekansıdır. Default olarak 16000'dedir. Yani 16kHz olarak ayarlanmıştır.
        chunk_duration (int, optional): _description_. Parçalara ayrılacak süredir. Default olarak 9'dur.
        total_duration (int, optional): _description_. Ses kaydı yapılacak süredir. Default olarak 10'dur.

    Returns:
        text: _description_. Model tarafından üretilen metin çıktısı.
    """
    model = WhisperModel(model_size, device = device)  
    buffer = np.zeros(0, dtype=np.float32)

    def callback(indata: np.ndarray, frames: int, time: sd.CallbackStop, status: sd.CallbackFlags) -> None:
        """
        Mikrofon akışından gelen ses verisini işler.

        Args:
            indata (np.ndarray): _description_. Şekli (frames, channels) olan ses verisi dizisi (örnekler).
            frames (int): _description_. Bu callback çağrısında alınan örnek sayısı.
            time (sd.CallbackStop): _description_. Zaman bilgilerini içeren bir yapı.
            status (sd.CallbackFlags): _description_. Giriş akışının durumu hakkında bilgi.
        Returns:
            None
        """
        nonlocal buffer
        buffer = np.concatenate((buffer, indata[:, 0]))

    with sd.InputStream(callback=callback, channels=1, samplerate=fs):
        st.info("Ses kaydı dinleniyor...")
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= total_duration:
                break

            if len(buffer) >= fs * chunk_duration:
                chunk = buffer[:fs * chunk_duration]
                buffer = buffer[int(fs * (chunk_duration - 0.5)):]  # overlap: 0.5s
                segments, _ = model.transcribe(chunk, language="tr")
                text = " ".join([seg.text for seg in segments])
                logging.info(f"Seslendirilen icerik : {text}")
                return text
            # CPU yükünü düşür
            time.sleep(0.1)  

def gemini_edit_product_description(description:str) -> str:
    """
    Text girdisinden ürün açıklamasını düzenler. 
    Args:
        description (str): _description_. Ürün açıklaması.
    Returns:
        response.text (str): _description_. Düzenlenmiş ürün açıklaması.
    """
    genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Sen, e-ticaret platformları için ürün açıklamalarını SEO dostu ve dikkat çekici bir şekilde yazan deneyimli bir metin yazarısın. Ürün açıklamaları e-ticaret platformlarında yayınlanmaya uygun olmalıdır. Aşağıda verilen ürün açıklamasını değerlendirip sadece açıklamanın son halini yaz:
    Ürün Açıklaması:
    "{description}"
    """
    response = model.generate_content(prompt)
    logging.info(f"Girilen Açıklama: {description} Model Çıktısı: {response.text}")
    return response.text

def gemini_edit_product_description_with_image(image_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """
    Resim girdisinden ürün açıklamasını düzenler.
    Args:
        image_file (st.runtime.uploaded_file_manager.UploadedFile): _description_. Ürün resmi dosyası.
    Returns:
        response.text (str): _description_. Düzenlenmiş ürün açıklaması.
    """
    genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    with Image.open(image_file).convert("RGB") as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
    # Gemini'nin beklediği format
    image_dict = {
        "mime_type": "image/jpeg",
        "data": img_bytes
    }
    response = model.generate_content(
    [ 
        "Sen, e-ticaret platformları için ürün açıklamalarını SEO dostu ve dikkat çekici bir şekilde yazan deneyimli bir metin yazarısın. Görseller e-ticaret platformlarında yayınlanmaya uygun olmalıdır. Verilen görseli değerlendirip sadece ürün açıklamasını yaz:", 
        image_dict 
    ])
    logging.info(f"Girilen Görsel Sonucu Model Çıktısı: {response.text}")
    return response.text

def enhance_image(image_file:st.runtime.uploaded_file_manager.UploadedFile):
    """
    Ürün resmini Real-ESRGAN ve CLAHE algoritmaları ile iyileştirir.
    Args:
        image_file (st.runtime.uploaded_file_manager.UploadedFile): _description_. Sisteme yüklenen ürün resmi dosyası.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            İlk değer: Yüklenen görüntünün sadece GAN modeli ile süper çözünürlük uygulanmış hali (RGB formatında NumPy dizisi).
            İkinci değer: GAN modeli + CLAHE uygulanmış görüntü (RGB formatında NumPy dizisi).
    """
    Vismodel = real_esrgan_and_clahe.VisModel
    img = Image.open(image_file).convert("RGB")
    vis_model = Vismodel(img)
    upscaled_image = vis_model.pred_gan(img)
    clahe_image = vis_model.pred_gan_with_clahe(img)
    return np.array(upscaled_image), np.array(clahe_image)

def request_try_on_api(person_img: st.runtime.uploaded_file_manager.UploadedFile, clothe_img: st.runtime.uploaded_file_manager.UploadedFile, url:str = "https://try-on-diffusion.p.rapidapi.com/try-on-file"):
    """
    Virtual Try-On API'sine istek gönderir ve manken üzerine oturtulmuş ürün görselini döndürür.
    Args:
        person_img (st.runtime.uploaded_file_manager.UploadedFile): _description_. Ürün denenecek kişinin resmi.
        clothe_img (st.runtime.uploaded_file_manager.UploadedFile): _description_. Üzerine oturtulacak kıyafet resmi.
        url (_type_, optional): _description_. API "https://try-on-diffusion.p.rapidapi.com/try-on-file".

    Returns:
        io.BytesIO(response.content): _description_. API cevabından dönen manken üzerine oturtulmuş ürün görseli.
    """
    headers = {
    "x-rapidapi-host": "try-on-diffusion.p.rapidapi.com",
    "x-rapidapi-key": os.getenv("X_RAPID_API_KEY"),
    }
    
    multipart_data = MultipartEncoder(
        fields={
            "avatar_image": ("person.png", person_img, "image/png"),
            "clothing_image": ("clothe.png", clothe_img, "image/png"),
        }
    )
    headers["Content-Type"] = multipart_data.content_type
    response = requests.post(url, headers=headers, data=multipart_data)
    response.raise_for_status()  # Hata kontrolü
    logging.info("Virtual Try-On API çağrısı başarılı.")
    return io.BytesIO(response.content) 
    
page = st.sidebar.selectbox("Sayfa Seçin", ["Ana Sayfa", "Girdi ile Ürün Açıklaması Düzeltme", "Ürün Resmi ile Ürün Açıklaması Oluşturma", "Ürün Görselini İyileştirme", "Virtual Try-On"])

def page1() -> None:
    st.title("Girdi ile Ürün Açıklaması Düzenleme")
    st.write("Bu hizmetle birlikte kullanıcılar aşağıdaki textbox'a girdiği açıklama ya da Seslendir butonu sayesinde üzerinde düzenleme yapılmasını istediği ürün açıklamasını girebilir.")
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Düzenlenmesini istediğiniz ürün açıklamasını yazın.", label_visibility="collapsed")

    with col2:
        send_clicked = st.button("Gönder")

    seslendir_clicked = st.button("Sesli Komut")

    if send_clicked and user_input:
        st.write(f"Girilen açıklama: {user_input}")
        st.write(f"Düzenlenmiş açıklama: {gemini_edit_product_description(user_input)}")

    if seslendir_clicked:
        st.write("🔊 Seslendirme işlemi başlatılıyor...")
        audio_desc = whisper_model()
        st.write(f"Sesli girilen açıklama: {audio_desc}")
        st.write(f"Düzenlenmiş açıklama: {gemini_edit_product_description(audio_desc)}")

def page2() -> None:
    st.title("Resim Girdisi ile Ürün Açıklaması Düzenleme")
    st.write("Bu hizmetle birlikte kullanıcılar ürün resmini girerek otomatik olarak oluşturulmuş bir açıklama elde ederler.")
    image = st.file_uploader("Ürün resmini yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if image:
        st.image(image, caption="Yüklenen Ürün Resmi")
        st.write(f"Ürün Açıklaması: {gemini_edit_product_description_with_image(image)}")

def page3() -> None:
    st.title("Ürün Görsellerini İyileştirme")
    st.write("Bu hizmetle birlikte kullanıcıların girmiş olduğu ürün resmi CLAHE ve Real-ESRGAN algoritmaları ile iyileştirilir.")
    image = st.file_uploader("İyileştirilecek ürün resmini yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if image:
        gan_img, clahe_image = enhance_image(image)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Yüklenen Ürün Resmi")
        with col2:
            st.image(gan_img, caption="Real-ESRGAN ile İyileştirilmiş Resim")
        with col3:
            st.image(clahe_image, caption="Real-ESRGAN + CLAHE ile İyileştirilmiş Resim")

def page4() -> None:
    st.title("Ürün Görseli ile Manken Üzerine Oturtma")
    st.write("Bu hizmetle birlikte kullanıcılar ürün resmini yükleyerek manken üzerine oturtulmuş bir görsel elde ederler.")
    person_img = st.file_uploader(label = "Lütfen üzerinde ürün denenecek kişi resmini yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    clothe_img = st.file_uploader(label = "Lütfen üzerine oturtulacak kıyafet resmini yükleyin", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if person_img and clothe_img:
        fitted_img = request_try_on_api(person_img=person_img, clothe_img=clothe_img)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(Image.open(person_img), caption="Kişi Resmi")
        with col2:
            st.image(Image.open(clothe_img), caption="Kıyafet Resmi")
        with col3:
            st.image(fitted_img, caption="Manken Üzerine Oturtulmuş Ürün Görseli")        

if page == "Ana Sayfa":
    st.set_page_config(page_title = "BTK Akademi AI Hackathon Projesi")
    st.title("BTK Akademi AI Hackathon Projesi")
    st.write("Bu uygulama, BTK Akademi AI Hackathon Projesi kapsamında geliştirilmiştir.")
    st.write("E-Ticaret platformlarının yaygınlaşması günümüzde inkar edilemez bir gerçek haline gelmiştir. Bu da ürün satıcılarına profesyonel çekim, manken çekimleri gibi çeşitli masraflar ortaya çıkarmıştır. Bu platformun oluşturulma amacı ürün satıcılarının arasındaki rekabet uçurumunu ortadan kaldırarak ürün satışının kolaylaştırılmasıdır. ")
    st.write("Sunulan Hizmetler:")
    st.write("1. Sesli ya da manuel olarak girilen ürün açıklamalarını e-ticaret platformlarına uygun hale getirip yeni açıklamalar oluşturmak.")
    st.write("2. Ürün görsellerinden e-ticaret platformlarına uygun açıklamalar oluşturmak.")
    st.write("3. Ürün görsellerini Real-ESRGAN ve CLAHE algoritmaları ile iyileştirmek.")
    st.write("4. E-Ticaret platformlarında sıkça satılan kıyafet kategorisi için ilgili kıyafetin örnek görseldeki mankene yapay zeka ile üzerine oturtulması ve bu sayede ürün satıcılarına kolaylık sağlamak.")

elif page == "Girdi ile Ürün Açıklaması Düzeltme":
    page1()        

elif page == "Ürün Resmi ile Ürün Açıklaması Oluşturma":
    page2()
elif page == "Ürün Görselini İyileştirme":
    page3()
else:
    page4()
