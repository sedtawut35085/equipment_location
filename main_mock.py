from rembg import remove
from PIL import Image

def main():
    input_path = 'dataset/rotten/x_20250601_223327.jpg'
    output_path = 'temp_file/output_no_bg.png'

    input_image = Image.open(input_path)

    # ลบ background
    output_image = remove(input_image)

    # บันทึกไฟล์ PNG (transparent background)
    output_image.save(output_path)

if __name__ == "__main__":
    main()
