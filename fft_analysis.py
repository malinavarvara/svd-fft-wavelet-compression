import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import fftpack

def fft_compression_analysis(image_path):
    """Анализ сжатия изображения с помощью FFT"""
    
    # Загрузка и преобразование в ч/б
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image, dtype=float)
    
    print(f"Размер изображения: {image_matrix.shape}")
    
    # Выполняем 2D FFT
    fft_transform = np.fft.fft2(image_matrix)
    fft_shifted = np.fft.fftshift(fft_transform)  # Сдвигаем нулевую частоту в центр
    
    # Вычисляем амплитуду (модуль) для визуализации
    magnitude_spectrum = np.log(1 + np.abs(fft_shifted))
    
    # Визуализируем исходное изображение и спектр Фурье
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_matrix, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='hot')
    plt.title('Спектр Фурье (логарифм амплитуды)')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.angle(fft_shifted), cmap='hsv')
    plt.title('Фаза Фурье')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return image_matrix, fft_transform, fft_shifted

def compress_with_fft(fft_coeffs, keep_ratio):
    """
    Сжатие FFT коэффициентов - сохранение только максимальных по модулю коэффициентов
    """
    # Получаем амплитуды коэффициентов
    magnitudes = np.abs(fft_coeffs)
    
    # Вычисляем порог для сохранения указанного процента коэффициентов
    total_coeffs = fft_coeffs.size
    keep_count = int(total_coeffs * keep_ratio)
    
    # Находим пороговое значение
    threshold = np.partition(magnitudes.ravel(), -keep_count)[-keep_count]
    
    # Создаем маску для коэффициентов выше порога
    mask = magnitudes >= threshold
    
    # Применяем маску - обнуляем малые коэффициенты
    compressed_fft = fft_coeffs * mask
    
    return compressed_fft, mask

def reconstruct_from_compressed_fft(compressed_fft, original_shape):
    """Восстановление изображения из сжатых FFT коэффициентов"""
    # Обратное FFT преобразование
    reconstructed = np.fft.ifft2(compressed_fft).real
    
    # Ограничиваем значения в диапазоне 0-255
    reconstructed = np.clip(reconstructed, 0, 255)
    
    return reconstructed.astype(np.uint8)

def main_fft_analysis():
    """Основная функция анализа FFT сжатия"""
    
    # Проценты сохранения коэффициентов
    compression_ratios = [0.10, 0.05, 0.01, 0.002]
    ratios_names = ['10%', '5%', '1%', '0.2%']
    
    # Анализ для портрета
    print("=" * 80)
    print("АНАЛИЗ FFT СЖАТИЯ - ПОРТРЕТ")
    print("=" * 80)
    
    try:
        image_matrix, fft_transform, fft_shifted = fft_compression_analysis('portret.jpg')
        
        # Результаты сжатия
        print(f"\n{'Уровень':<10} {'Коэфф.':<8} {'Норма ошибки':<15} {'Сохраняемые':<12}")
        print("-" * 60)
        
        # Создаем subplot для визуализации результатов
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Исходное изображение
        axes[0].imshow(image_matrix, cmap='gray')
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        for i, (ratio, name) in enumerate(zip(compression_ratios, ratios_names)):
            # Сжимаем FFT коэффициенты
            compressed_fft, mask = compress_with_fft(fft_transform, ratio)
            
            # Восстанавливаем изображение
            reconstructed = reconstruct_from_compressed_fft(compressed_fft, image_matrix.shape)
            
            # Вычисляем норму Фробениуса ошибки
            error_norm = np.linalg.norm(image_matrix - reconstructed, 'fro')
            
            # Подсчитываем количество сохраняемых коэффициентов
            kept_coeffs = np.sum(mask)
            total_coeffs = mask.size
            kept_percentage = (kept_coeffs / total_coeffs) * 100
            
            print(f"{name:<10} {ratio:<8.3f} {error_norm:<15.2f} {kept_coeffs:<12} ({kept_percentage:.1f}%)")
            
            # Визуализируем результат
            axes[i+1].imshow(reconstructed, cmap='gray')
            axes[i+1].set_title(f'Сохранено {name} коэфф.\nОшибка: {error_norm:.1f}')
            axes[i+1].axis('off')
        
        # Скрываем последний subplot если не используется
        if len(compression_ratios) + 1 < len(axes):
            for j in range(len(compression_ratios) + 1, len(axes)):
                axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительный анализ: визуализация маски сжатия
        visualize_compression_mask(fft_transform, compression_ratios[0])  # Для 10%
        
    except FileNotFoundError:
        print("Файл 'portret.jpg' не найден!")
        return

def visualize_compression_mask(fft_coeffs, ratio):
    """Визуализация того, какие коэффициенты сохраняются"""
    compressed_fft, mask = compress_with_fft(fft_coeffs, ratio)
    
    # Сдвигаем для визуализации
    mask_shifted = np.fft.fftshift(mask.astype(float))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(mask_shifted, cmap='gray')
    plt.title(f'Маска сохранения коэффициентов ({ratio*100:.1f}%)')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    # Показываем спектр с сохраненными коэффициентами
    compressed_shifted = np.fft.fftshift(compressed_fft)
    magnitude = np.log(1 + np.abs(compressed_shifted))
    plt.imshow(magnitude, cmap='hot')
    plt.title(f'Спектр после сжатия ({ratio*100:.1f}%)')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def advanced_fft_analysis(image_path):
    """Расширенный анализ FFT с различными стратегиями сжатия"""
    
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image, dtype=float)
    fft_transform = np.fft.fft2(image_matrix)
    
    ratios = [0.10, 0.05, 0.01, 0.002]
    
    print("\n" + "="*80)
    print("РАСШИРЕННЫЙ АНАЛИЗ FFT СЖАТИЯ")
    print("="*80)
    print(f"{'Метод':<15} {'Уровень':<10} {'Коэфф.':<8} {'Ошибка':<12} {'PSNR':<10}")
    print("-" * 70)
    
    for ratio in ratios:
        # Метод 1: Сохранение максимальных коэффициентов (как в основном задании)
        compressed_fft1, _ = compress_with_fft(fft_transform, ratio)
        reconstructed1 = reconstruct_from_compressed_fft(compressed_fft1, image_matrix.shape)
        error1 = np.linalg.norm(image_matrix - reconstructed1, 'fro')
        psnr1 = 20 * np.log10(255.0 / (error1 / np.sqrt(image_matrix.size)))
        
        print(f"{'Макс. коэфф.':<15} {ratio*100:<9.1f}% {ratio:<8.3f} {error1:<12.2f} {psnr1:<10.2f}")
        
        # Метод 2: Сохранение низкочастотных коэффициентов
        compressed_fft2 = keep_low_frequencies(fft_transform, ratio)
        reconstructed2 = reconstruct_from_compressed_fft(compressed_fft2, image_matrix.shape)
        error2 = np.linalg.norm(image_matrix - reconstructed2, 'fro')
        psnr2 = 20 * np.log10(255.0 / (error2 / np.sqrt(image_matrix.size)))
        
        print(f"{'Низкие частоты':<15} {ratio*100:<9.1f}% {ratio:<8.3f} {error2:<12.2f} {psnr2:<10.2f}")

def keep_low_frequencies(fft_coeffs, ratio):
    """Альтернативный метод: сохранение низкочастотных коэффициентов"""
    rows, cols = fft_coeffs.shape
    center_r, center_c = rows // 2, cols // 2
    
    # Вычисляем радиус для сохранения указанного процента коэффициентов
    total_coeffs = rows * cols
    target_coeffs = int(total_coeffs * ratio)
    
    # Находим радиус, который содержит нужное количество коэффициентов
    radius = 1
    while True:
        area = np.pi * radius * radius
        if area >= target_coeffs:
            break
        radius += 1
    
    # Создаем маску для низкочастотных коэффициентов
    mask = np.zeros_like(fft_coeffs, dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_r)**2 + (j - center_c)**2)
            if dist <= radius:
                mask[i, j] = True
    
    return fft_coeffs * mask

# Запуск анализа
if __name__ == "__main__":
    main_fft_analysis()
    
    # Дополнительный расширенный анализ
    try:
        advanced_fft_analysis('portret.jpg')
    except FileNotFoundError:
        print("Файл 'portret.jpg' не найден для расширенного анализа")