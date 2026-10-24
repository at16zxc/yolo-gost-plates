import csv
from pathlib import Path

def convert_meta_to_yolo(dataset_root: str = 'dataset') -> None:
    root = Path(dataset_root)
    meta_path = root / 'meta.csv'
    labels_dir = root / 'labels' / 'synthetic'
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        return

    class_map = {'type1': 0, 'type1a': 1, 'type1b': 2, 'other': 3}

    with meta_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        count = 0
        for row in reader:
            img_name = Path(row['image']).name
            stem = Path(img_name).stem
            class_id = class_map.get(row['plate_type'], 0)

            bbox_parts = [float(x) for x in row['bbox'].split(',')]
            if len(bbox_parts) == 4:
                x, y, w, h = bbox_parts
                x_center = (x + w / 2.0) / 1280.0
                y_center = (y + h / 2.0) / 720.0
                norm_w = w / 1280.0
                norm_h = h / 720.0

                label_file = labels_dir / f'{stem}.txt'
                label_file.write_text(f'{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n', encoding='utf-8')
                count += 1

    print(f'[OK] Создано {count} файлов разметки в {labels_dir}!')

if __name__ == '__main__':
    convert_meta_to_yolo()
