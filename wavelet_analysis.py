import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
import os

def wavelet_compression_analysis(image_path, wavelet_name='db4', level=4):
    """Анализ сжатия изображения с помощью вейвлет-преобразования"""
    
    # Загрузка и преобразование в ч/б
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image, dtype=float)
    
    print(f"Размер изображения: {image_matrix.shape}")
    print(f"Используемый вейвлет: {wavelet_name}")
    print(f"Уровень глубины: {level}")
    
    # Выполняем многоуровневое вейвлет-разложение
    coeffs = pywt.wavedec2(image_matrix, wavelet=wavelet_name, level=level)
    
    # Преобразуем коэффициенты в один массив для обработки
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Визуализируем исходное изображение и коэффициенты вейвлета
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_matrix, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Визуализация коэффициентов вейвлета (логарифм для лучшего отображения)
    coeff_vis = np.log(1 + np.abs(coeff_arr))
    plt.imshow(coeff_vis, cmap='hot')
    plt.title('Коэффициенты вейвлет-преобразования')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return image_matrix, coeffs, coeff_arr, coeff_slices

def compress_wavelet_coeffs(coeff_arr, keep_ratio):
    """
    Сжатие вейвлет-коэффициентов - сохранение только максимальных по модулю
    """
    # Получаем амплитуды коэффициентов
    magnitudes = np.abs(coeff_arr)
    
    # Вычисляем порог для сохранения указанного процента коэффициентов
    total_coeffs = coeff_arr.size
    keep_count = int(total_coeffs * keep_ratio)
    
    # Находим пороговое значение
    if keep_count > 0:
        threshold = np.partition(magnitudes.ravel(), -keep_count)[-keep_count]
    else:
        threshold = np.max(magnitudes) + 1  # Ничего не сохранять
    
    # Создаем маску для коэффициентов выше порога
    mask = magnitudes >= threshold
    
    # Применяем маску - обнуляем малые коэффициенты
    compressed_coeffs = coeff_arr * mask
    
    # Подсчитываем статистику
    kept_coeffs = np.sum(mask)
    compression_ratio = (1 - kept_coeffs / total_coeffs) * 100
    
    return compressed_coeffs, mask, kept_coeffs, compression_ratio

def reconstruct_from_compressed_wavelet(compressed_coeffs, coeff_slices, wavelet_name, original_shape):
    """Восстановление изображения из сжатых вейвлет-коэффициентов"""
    # Преобразуем обратно в структуру коэффициентов
    coeffs_fill = pywt.array_to_coeffs(compressed_coeffs, coeff_slices, output_format='wavedec2')
    
    # Обратное вейвлет-преобразование
    reconstructed = pywt.waverec2(coeffs_fill, wavelet=wavelet_name)
    
    # Обрезаем до исходного размера (на случай округления)
    reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    
    # Ограничиваем значения в диапазоне 0-255
    reconstructed = np.clip(reconstructed, 0, 255)
    
    return reconstructed.astype(np.uint8)

def main_wavelet_analysis():
    """Основная функция анализа вейвлет-сжатия"""
    
    # Проценты сохранения коэффициентов
    compression_ratios = [0.10, 0.05, 0.01, 0.005]
    ratios_names = ['10%', '5%', '1%', '0.5%']
    
    # Поиск изображения
    image_path = find_image()
    
    if not image_path:
        print("Создаю тестовое изображение...")
        create_test_image()
        image_path = 'test_portret.jpg'
    
    print("=" * 80)
    print(f"АНАЛИЗ ВЕЙВЛЕТ-СЖАТИЯ - {image_path}")
    print("=" * 80)
    
    try:
        # Анализ с уровнем глубины 4
        image_matrix, coeffs, coeff_arr, coeff_slices = wavelet_compression_analysis(
            image_path, wavelet_name='db4', level=4)
        
        # Результаты сжатия
        print(f"\n{'Уровень':<10} {'Коэфф.':<8} {'Норма ошибки':<15} {'Сохраняемые':<12} {'Сжатие':<10}")
        print("-" * 70)
        
        # Создаем subplot для визуализации результатов
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Исходное изображение
        axes[0].imshow(image_matrix, cmap='gray')
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        for i, (ratio, name) in enumerate(zip(compression_ratios, ratios_names)):
            # Сжимаем вейвлет-коэффициенты
            compressed_coeffs, mask, kept_coeffs, comp_ratio = compress_wavelet_coeffs(coeff_arr, ratio)
            
            # Восстанавливаем изображение
            reconstructed = reconstruct_from_compressed_wavelet(
                compressed_coeffs, coeff_slices, 'db4', image_matrix.shape)
            
            # Вычисляем норму Фробениуса ошибки
            error_norm = np.linalg.norm(image_matrix - reconstructed, 'fro')
            
            print(f"{name:<10} {ratio:<8.3f} {error_norm:<15.2f} {kept_coeffs:<12} {comp_ratio:<9.1f}%")
            
            # Визуализируем результат
            axes[i+1].imshow(reconstructed, cmap='gray')
            axes[i+1].set_title(f'Сохранено {name} коэфф.\nОшибка: {error_norm:.1f}')
            axes[i+1].axis('off')
            
            # Сохраняем результат
            result_image = Image.fromarray(reconstructed)
            result_image.save(f'wavelet_compressed_{name}.jpg')
        
        # Скрываем последний subplot если не используется
        if len(compression_ratios) + 1 < len(axes):
            for j in range(len(compression_ratios) + 1, len(axes)):
                axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Экспериментируем с разным числом уровней
        experiment_with_levels(image_path, compression_ratios[1])  # Для 5% коэффициентов
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        import traceback
        traceback.print_exc()

def experiment_with_levels(image_path, ratio=0.05):
    """Эксперимент с разным числом уровней разложения"""
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ С РАЗНЫМ ЧИСЛОМ УРОВНЕЙ")
    print("="*80)
    
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image, dtype=float)
    
    levels_to_test = [1, 2, 3, 4, 5, 6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    axes[0].imshow(image_matrix, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    print(f"{'Уровень':<8} {'Норма ошибки':<15} {'Сохраняемые':<12}")
    print("-" * 50)
    
    for i, level in enumerate(levels_to_test):
        try:
            # Выполняем вейвлет-разложение с текущим уровнем
            coeffs = pywt.wavedec2(image_matrix, wavelet='db4', level=level)
            coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            
            # Сжимаем коэффициенты
            compressed_coeffs, mask, kept_coeffs, _ = compress_wavelet_coeffs(coeff_arr, ratio)
            
            # Восстанавливаем изображение
            reconstructed = reconstruct_from_compressed_wavelet(
                compressed_coeffs, coeff_slices, 'db4', image_matrix.shape)
            
            # Вычисляем ошибку
            error_norm = np.linalg.norm(image_matrix - reconstructed, 'fro')
            
            print(f"{level:<8} {error_norm:<15.2f} {kept_coeffs:<12}")
            
            # Визуализируем результат
            axes[i+1].imshow(reconstructed, cmap='gray')
            axes[i+1].set_title(f'Уровень: {level}\nОшибка: {error_norm:.1f}')
            axes[i+1].axis('off')
            
        except Exception as e:
            print(f"Ошибка для уровня {level}: {e}")
            axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

def find_image():
    """Поиск доступных изображений"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            return file
    
    return None

def create_test_image():
    """Создание тестового изображения если нет реального"""
    # Создаем тестовое изображение с текстурой для демонстрации вейвлетов
    test_image = np.zeros((512, 512), dtype=np.uint8)
    
    # Добавляем различные структуры для демонстрации вейвлет-анализа
    # Гладкая область
    test_image[100:200, 100:200] = 150
    
    # Резкие границы
    test_image[300:350, 300:450] = 200
    test_image[350:400, 300:450] = 100
    
    # Текстурная область (имитация травы/листьев)
    for i in range(400, 512, 4):
        for j in range(50, 150, 4):
            test_image[i:i+2, j:j+2] = np.random.randint(80, 120)
    
    # Вертикальные и горизонтальные линии
    test_image[250:255, 50:450] = 220  # Горизонтальная
    test_image[50:450, 250:255] = 220  # Вертикальная
    
    # Сохраняем тестовое изображение
    test_img = Image.fromarray(test_image)
    test_img.save('test_portret.jpg')
    print("Создано тестовое изображение 'test_portret.jpg'")

def compare_wavelet_families(image_path):
    """Сравнение разных семейств вейвлетов"""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РАЗНЫХ ВЕЙВЛЕТОВ")
    print("="*80)
    
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image, dtype=float)
    
    wavelets_to_test = ['db1', 'db4', 'db8', 'haar', 'sym4', 'coif2']
    ratio = 0.05  # 5% коэффициентов
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    axes[0].imshow(image_matrix, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    print(f"{'Вейвлет':<8} {'Норма ошибки':<15}")
    print("-" * 30)
    
    for i, wavelet in enumerate(wavelets_to_test):
        try:
            # Выполняем вейвлет-разложение
            coeffs = pywt.wavedec2(image_matrix, wavelet=wavelet, level=4)
            coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            
            # Сжимаем коэффициенты
            compressed_coeffs, _, _, _ = compress_wavelet_coeffs(coeff_arr, ratio)
            
            # Восстанавливаем изображение
            reconstructed = reconstruct_from_compressed_wavelet(
                compressed_coeffs, coeff_slices, wavelet, image_matrix.shape)
            
            # Вычисляем ошибку
            error_norm = np.linalg.norm(image_matrix - reconstructed, 'fro')
            
            print(f"{wavelet:<8} {error_norm:<15.2f}")
            
            # Визуализируем результат
            axes[i+1].imshow(reconstructed, cmap='gray')
            axes[i+1].set_title(f'Вейвлет: {wavelet}\nОшибка: {error_norm:.1f}')
            axes[i+1].axis('off')
            
        except Exception as e:
            print(f"Ошибка для вейвлета {wavelet}: {e}")
            axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Запуск анализа
if __name__ == "__main__":
    # Убедимся, что установлен PyWavelets
    try:
        import pywt
        print("PyWavelets установлен успешно")
    except ImportError:
        print("Установите PyWavelets: pip install PyWavelets")
        exit(1)
    
    main_wavelet_analysis()
    
    # Дополнительное сравнение вейвлетов
    try:
        image_path = find_image() or 'test_portret.jpg'
        compare_wavelet_families(image_path)
    except Exception as e:
        print(f"Ошибка при сравнении вейвлетов: {e}")