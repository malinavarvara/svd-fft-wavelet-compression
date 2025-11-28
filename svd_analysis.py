import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка изображения и преобразование в ч/б
image = Image.open('images\original\nature.jpg').convert('L')
image_array = np.array(image)

print(f"Размер матрицы: {image_array.shape}")
print(f"Тип данных: {image_array.dtype}")
print(f"Диапазон значений: [{image_array.min()}, {image_array.max()}]")
print(f"Пример значений:\n{image_array[:5, :5]}")

# === СИНГУЛЯРНОЕ РАЗЛОЖЕНИЕ ===

# Выполняем SVD
U, S, VT = np.linalg.svd(image_array, full_matrices=False)

# Общее количество сингулярных чисел
total_singular_values = len(S)
print(f"\nВсего сингулярных чисел: {total_singular_values}")

# Вычисляем кумулятивную энергию
cumulative_energy = np.cumsum(S) / np.sum(S)

# График сингулярных чисел и кумулятивной энергии
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(S, 'b-', linewidth=1)
plt.title('Сингулярные числа (логарифмическая шкала)')
plt.xlabel('Индекс r')
plt.ylabel('σ_r (log scale)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cumulative_energy, 'r-', linewidth=2)
plt.title('Кумулятивная энергия')
plt.xlabel('Количество компонент r')
plt.ylabel('Доля общей энергии')
plt.grid(True)
plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()

# Проценты для усечения
compression_ratios = [0.10, 0.05, 0.01, 0.002]
ratios_names = ['10%', '5%', '1%', '0.2%']

print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ СЖАТИЯ С ПОМОЩЬЮ SVD")
print("="*80)
print(f"{'Уровень':<10} {'r':<8} {'% энергии':<12} {'Ошибка Фробениуса':<20}")
print("-" * 60)

# Визуализация результатов - исправленная часть
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()  # Преобразуем в одномерный массив для простой индексации

# Исходное изображение
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('Исходное изображение')
axes[0].axis('off')

# Восстановленные изображения для разных уровней сжатия
for i, (ratio, name) in enumerate(zip(compression_ratios, ratios_names)):
    # Вычисляем количество компонент
    r = max(1, int(total_singular_values * ratio))
    
    # Восстанавливаем изображение с первыми r компонентами
    reconstructed = U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # Вычисляем процент энергии
    energy_percentage = cumulative_energy[r-1] * 100
    
    # Вычисляем норму Фробениуса ошибки
    error_norm = np.linalg.norm(image_array - reconstructed, 'fro')
    
    print(f"{name:<10} {r:<8} {energy_percentage:<10.2f}% {error_norm:<20.2f}")
    
    # Исправленная индексация - используем i+1
    axes[i+1].imshow(reconstructed, cmap='gray')
    axes[i+1].set_title(f'r = {r} ({name})\nЭнергия: {energy_percentage:.1f}%')
    axes[i+1].axis('off')

# Скрываем последний subplot если он не используется
if len(compression_ratios) + 1 < len(axes):
    for j in range(len(compression_ratios) + 1, len(axes)):
        axes[j].axis('off')

plt.tight_layout()
plt.show()

# Дополнительная информация
print("\n" + "="*80)
print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
print("="*80)
print(f"Размер исходной матрицы: {image_array.shape}")
print(f"Общее количество элементов: {image_array.size}")
print(f"Размер U: {U.shape}")
print(f"Размер S: {S.shape}")
print(f"Размер VT: {VT.shape}")

# Экономия памяти для разных уровней сжатия
print("\nЭКОНОМИЯ ПАМЯТИ:")
print(f"{'Уровень':<10} {'r':<8} {'Исходный размер':<18} {'Сжатый размер':<16} {'Коэф. сжатия':<12}")
print("-" * 70)

m, n = image_array.shape
for ratio, name in zip(compression_ratios, ratios_names):
    r = max(1, int(total_singular_values * ratio))
    
    # Исходный размер (в элементах)
    original_size = m * n
    
    # Размер сжатого представления: U(m×r) + S(r) + VT(r×n)
    compressed_size = m * r + r + r * n
    
    compression_ratio = original_size / compressed_size
    
    print(f"{name:<10} {r:<8} {original_size:<18} {compressed_size:<16} {compression_ratio:<10.2f}x")

# Сохранение восстановленных изображений
print("\nСохранение изображений...")
for ratio, name in zip(compression_ratios, ratios_names):
    r = max(1, int(total_singular_values * ratio))
    reconstructed = U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # Сохраняем изображение
    reconstructed_image = Image.fromarray(reconstructed)
    filename = f"nature_svd_{name.replace('%', '')}.jpg"
    reconstructed_image.save(filename)
    print(f"Сохранено: {filename}")

print("\nАнализ завершен!")