import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image
import argparse
import os
import time

def generate_video(prompt, image_path=None, output_path="output.mp4", num_frames=81, height=480, width=832, offload=False):
    is_i2v = image_path is not None
    
    # Wybór modelu w zależności od tego, czy podano obrazek startowy
    if is_i2v:
        # Wan2.1 oficjalnie wspiera Image-to-Video (I2V) w modelu 14B-480P
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        print(f"[{time.strftime('%H:%M:%S')}] Wykryto obraz startowy: {image_path}")
        print(f"[{time.strftime('%H:%M:%S')}] Używam modelu Image-to-Video (14B): {model_id}")
    else:
        # Dla samego tekstu używamy wybranego wcześniej lżejszego modelu 1.3B
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        print(f"[{time.strftime('%H:%M:%S')}] Brak obrazu startowego.")
        print(f"[{time.strftime('%H:%M:%S')}] Używam modelu Text-to-Video (1.3B): {model_id}")
        
    print("Przy pierwszym uruchomieniu zostaną pobrane wagi modelu z HuggingFace.")
    
    # Ładowanie pipeline'u (DiffusionPipeline automatycznie dobierze właściwą klasę np. WanPipeline / WanI2VPipeline)
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return

    # Zarządzanie pamięcią GPU
    if offload:
        print("Włączono optymalizację pamięci (CPU offload).")
        pipeline.enable_model_cpu_offload()
    else:
        print("Przenoszenie modelu bezpośrednio do pamięci VRAM (Karty Graficznej).")
        pipeline.to("cuda")

    print(f"[{time.strftime('%H:%M:%S')}] Generowanie wideo dla promptu: '{prompt}'")
    print(f"Ustawienia: klatki={num_frames}, rozdzielczość={width}x{height}")
    
    # Generowanie klatek
    start_time = time.time()
    try:
        if is_i2v:
            # Wczytywanie obrazka
            image = load_image(image_path)
            
            # W modelu I2V przekazujemy parametr image
            output = pipeline(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=5.0, # Zgodnie z zaleceniami Wan
            ).frames[0]
        else:
            output = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=5.0,
            ).frames[0]
            
    except Exception as e:
        print(f"Błąd podczas generowania wideo (prawdopodobnie brak pamięci VRAM): {e}")
        return
        
    duration = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Zakończono generowanie. Czas trwania: {duration:.2f} sekund.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Zapisywanie pliku wideo do '{output_path}'...")
    try:
        export_to_video(output, output_path, fps=16)
        print(f"[{time.strftime('%H:%M:%S')}] Sukces! Wideo zostało wygenerowane i zapisane.")
    except Exception as e:
        print(f"Błąd podczas zapisu wideo. Upewnij się, że masz poprawnie zainstalowane 'imageio-ffmpeg': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generuj wideo używając modelu Wan 2.1 (T2V 1.3B lub I2V 14B)")
    parser.add_argument("--prompt", type=str, required=True, help="Tekst (prompt) do wygenerowania wideo (najlepiej po angielsku)")
    parser.add_argument("--image", type=str, default=None, help="Opcjonalne: Ścieżka do zdjęcia początkowego. Przełącza skrypt w tryb Image-to-Video.")
    parser.add_argument("--output", type=str, default="wan_video_output.mp4", help="Ścieżka do zapisanego pliku wyjściowego .mp4")
    parser.add_argument("--frames", type=int, default=81, help="Liczba klatek (domyślnie: 81)")
    parser.add_argument("--offload", action="store_true", help="Użyj tej flagi, jeśli masz mało pamięci VRAM i chcesz oszczędzać zasoby")
    
    args = parser.parse_args()
    
    # Sprawdzenie dostępności CUDA
    if not torch.cuda.is_available():
        print("UWAGA: Nie wykryto karty graficznej z CUDA! Generowanie na CPU będzie ekstremalnie wolne i może nie działać prawidłowo.")
        print("Jeśli jesteś podłączony do serwera NVIDIA, upewnij się, że masz poprawne sterowniki i PyTorch z obsługą CUDA.")
        
    generate_video(args.prompt, args.image, args.output, args.frames, offload=args.offload)
